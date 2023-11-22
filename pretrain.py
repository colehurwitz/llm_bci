import os
import argparse
from tqdm import tqdm
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from datasets import load_from_disk
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models.neural_encoder import NeuralEncoder
from models.neural_pretrainer import NeuralPretrainer

from utils.config_utils import update_config, config_from_kwargs, ParseKwargs
from utils.data_utils import NeuralPretrainerDataset, pt_pad_collate_fn
from utils.eval_utils import format_ctc, word_error_count, smoothed_RMS
from utils.optim_utils import WarmupCosineScheduler

DEFAULT_CONFIG_FILE = "configs/default_pretrain_config.yaml"

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False

def main(args):
    torch.set_printoptions(profile="full")
    
    # Get configs
    config = update_config(DEFAULT_CONFIG_FILE, args.config_file if args.config_file != "none" else None) 
    config = update_config(config, config_from_kwargs(args.kwargs))

    accelerator = Accelerator()
    reset_seeds(config.seed)
    accelerator.print(f"Starting run {config.savestring}")

    # Create saving paths
    checkpoint_dir = os.path.join(config.checkpoint_dir,config.savestring)
    pt_dir = os.path.join(config.pt_dir,config.savestring)
    log_dir = os.path.join(config.log_dir,config.savestring)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)
    # Init tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # Load preprocessed dataset
    print(f"Loading data from {os.path.join(config.data_dir, config.data_file)}")
    data = torch.load(os.path.join(config.data_dir, config.data_file))

    # Get vocabulary info
    vocab = data["train"]["info"]["vocab"] if config.neural_pretrainer.loss.type == "ctc" else None
    blank_id = vocab.index("BLANK") if config.neural_pretrainer.loss.type == "ctc" else None
    vocab_size = len(vocab)  if config.neural_pretrainer.loss.type == "ctc" else None

    # Created train and test dataloaders
    train_data = data["train"]
    test_data = data["test"]

    train_dataset = NeuralPretrainerDataset(train_data, loss_fn=config.neural_pretrainer.loss.type, len=config.trainer.train_len)
    test_dataset = NeuralPretrainerDataset(test_data, loss_fn=config.neural_pretrainer.loss.type, len=config.trainer.test_len)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=partial(pt_pad_collate_fn,blank_id), batch_size=config.trainer.train_batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=partial(pt_pad_collate_fn,blank_id), batch_size=config.trainer.test_batch_size, pin_memory=True
    )   


    # Create encoder model for pretraining
    config["neural_config"]["embedder"]["n_channels"] = train_data["model_inputs"]["features"][0].shape[-1]
    accelerator.print(config)
    
    encoder = NeuralEncoder(config.neural_config)
    model = NeuralPretrainer(encoder, config.neural_pretrainer, vocab_size, blank_id)
    if config.model_dir is not None:  
        print(f"Loading model from {config.model_dir}")
        model.encoder.load_state_dict(torch.load(os.path.join(config.model_dir,"encoder.pth")))
        model.decoder.load_state_dict(torch.load(os.path.join(config.model_dir,"decoder.pth")))
    else:
        print("Creating model from scratch")

    accelerator.print(model)
    accelerator.print(f"Encoder params: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad):,}")
    accelerator.print(f"Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)

    if config.optimizer.scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.optimizer.warmup_steps,
            num_training_steps=(len(train_dataloader) * config.trainer.num_epochs),
        )
    elif config.optimizer.scheduler == "cosine":
        lr_scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=config.optimizer.warmup_epochs,
            warmup_lr=0,
            num_epochs=config.trainer.num_epochs,
            base_lr=config.optimizer.lr,
            final_lr=0,
            iter_per_epoch=len(train_dataloader)
        )
    else:
        raise Exception(f"Scheduler '{config.optimizer.scheduler}' not implemented")

    # Prepare model for distributed training
    model, train_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, test_dataloader, optimizer, lr_scheduler
    )

    # Train
    print("Start pretraining")
    for epoch in range(1,config.trainer.num_epochs+1):
        
        print(f"Training epoch {epoch}")
        
        model.train()

        # Train metrics
        train_loss = []
        train_examples = []

        # Poisson metrics
        train_RMS = []
        train_all_RMS = []
        train_all_examples = []
        
        # CTC metrics
        train_errors = []
        train_phonemes = []

        # To save
        train_sentences = []
        train_phonograms = []
        train_logits = []
        train_preds = []
        train_features = []
        train_features_mask = []
        train_targets = []
        train_targets_mask = []

        # Schedule for the probability of expanding the mask
        if config.masker_scheduler.do:
            expand_prob = max(0, min(1, (epoch - config.masker_scheduler.start) / (config.masker_scheduler.end - config.masker_scheduler.start)))
            model.encoder.masker.expand_prob = expand_prob

        for step, (batch, phonograms, sentences) in enumerate(tqdm(train_dataloader)):
    
            outputs = model(**batch)
            loss = outputs.loss

            # Gather general train metrics
            train_loss.append(loss.detach().item())
            train_examples.append(outputs.n_examples)

            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Gather loss-specific train metrics
            if config.neural_pretrainer.loss.type == "ctc": # and (step+1)%10:
                logits = outputs.outputs.detach().cpu()
                preds = torch.argmax(logits, -1)
                preds = [" ".join(format_ctc(pred, vocab, blank_id)) for pred in preds]
                phonograms = [" ".join(p) for p in phonograms]
                errors, phonemes = word_error_count(preds, phonograms)
                train_errors.append(errors)
                train_phonemes.append(phonemes)
                
                preds = [pred.replace(" ","").replace("SIL"," SIL ") for pred in preds]
                phonograms = [p.replace(" ","").replace("SIL"," SIL ") for p in phonograms]
                train_preds += preds
                train_sentences += sentences
                train_phonograms += phonograms
                train_logits.append(logits)
                writer.add_scalar("PER/train_iter",errors/phonemes, 1+step+(epoch-1)*len(train_dataloader))
            elif config.neural_pretrainer.loss.type == "poisson":

                preds = outputs.outputs.detach().cpu()
                if config.neural_pretrainer.use_lograte:
                    preds = torch.exp(preds)

                features = batch["features"].detach().cpu()
                features_mask = batch["features_mask"].detach().cpu()
                targets = batch["targets"].detach().cpu()
                targets_mask = outputs.targets_mask.detach().cpu()
                
                RMS, all_RMS = smoothed_RMS(preds, features, targets_mask, config.eval.smoothing)
                train_RMS.append(RMS)
                train_all_RMS.append(all_RMS)
                train_all_examples.append(features.nelement())
                train_preds.append(preds)
                train_features.append(features)
                train_features_mask.append(features_mask)
                train_targets.append(targets)
                train_targets_mask.append(targets_mask)


                writer.add_scalar("RMS/train_iter",(train_RMS[-1]/train_examples[-1])**0.5, 1+step+(epoch-1)*len(train_dataloader))
                writer.add_scalar("AllRMS/train_iter",(train_all_RMS[-1]/train_all_examples[-1])**0.5, 1+step+(epoch-1)*len(train_dataloader))

            writer.add_scalar("Loss/train_iter",train_loss[-1]/train_examples[-1], 1+step+(epoch-1)*len(train_dataloader))
            
        # Log to tensorboard
        train_epoch_loss = sum(train_loss) / sum(train_examples)
        writer.add_scalar("Loss/train",train_epoch_loss,epoch)
        if config.neural_pretrainer.loss.type == "ctc":
            train_epoch_PER = sum(train_errors) / sum(train_phonemes)
            writer.add_scalar("PER/train",train_epoch_PER,epoch)
        elif config.neural_pretrainer.loss.type == "poisson":
            train_epoch_RMS = (sum(train_RMS) / sum(train_examples))**0.5
            writer.add_scalar("RMS/train",train_epoch_RMS,epoch)
            train_epoch_all_RMS = (sum(train_all_RMS) / sum(train_all_examples))**0.5
            writer.add_scalar("AllRMS/train",train_epoch_all_RMS,epoch)


        print(f"Evaluation epoch {epoch}")
        model.eval()

        # Test metrics
        test_loss = []
        test_examples = []

        # Poisson metrics
        test_RMS = []
        test_all_RMS = []
        test_all_examples = []
        
        # CTC metrics
        test_errors = []
        test_phonemes = []

        # To save
        test_sentences = []
        test_phonograms = []
        test_logits = []
        test_preds = []
        test_features = []
        test_features_mask = []
        test_targets = []
        test_targets_mask = []
        

        for step, (batch, phonograms, sentences) in enumerate(tqdm(test_dataloader)):

            # Forward pass
            with torch.no_grad():
                outputs = model(**batch)
            
            # Gather general test metrics
            test_loss.append(outputs.loss.detach().item())
            test_examples.append(outputs.n_examples)

            
            # Gather loss-specific test metrics
            if config.neural_pretrainer.loss.type == "ctc":
                logits = outputs.outputs.detach().cpu()
                preds = torch.argmax(logits,-1)
                preds = [" ".join(format_ctc(pred, vocab, blank_id)) for pred in preds]
                phonograms = [" ".join(p) for p in phonograms]
                errors, phonemes = word_error_count(preds, phonograms)
                test_errors.append(errors)
                test_phonemes.append(phonemes)
                preds = [pred.replace(" ","").replace("SIL"," SIL ") for pred in preds]
                phonograms = [p.replace(" ","").replace("SIL"," SIL ") for p in phonograms]
                test_preds += preds
                test_sentences += sentences
                test_phonograms += phonograms
                test_logits.append(logits)
                writer.add_scalar("PER/test_iter",errors/phonemes, 1+step+(epoch-1)*len(test_dataloader))
            elif config.neural_pretrainer.loss.type == "poisson":
                
                preds = outputs.outputs.detach().cpu()
                if config.neural_pretrainer.use_lograte:
                    preds = torch.exp(preds)

                features = batch["features"].detach().cpu()
                features_mask = batch["features_mask"].detach().cpu()
                targets = batch["targets"].detach().cpu()
                targets_mask = outputs.targets_mask.detach().cpu()

                RMS, all_RMS = smoothed_RMS(preds, targets, targets_mask, config.eval.smoothing)
                test_RMS.append(RMS)
                test_all_RMS.append(all_RMS)
                test_all_examples.append(features.nelement())
                test_preds.append(preds)
                test_features.append(features)
                test_features_mask.append(features_mask)
                test_targets.append(targets)
                test_targets_mask.append(targets_mask)

                writer.add_scalar("RMS/test_iter",(test_RMS[-1]/test_examples[-1])**0.5, 1+step+(epoch-1)*len(test_dataloader))
                writer.add_scalar("AllRMS/test_iter",(test_all_RMS[-1]/test_all_examples[-1])**0.5, 1+step+(epoch-1)*len(test_dataloader))

            writer.add_scalar("Loss/test_iter",test_loss[-1]/test_examples[-1],1+step+(epoch-1)*len(test_dataloader))
            
        # Log to tensorboard
        test_epoch_loss = sum(test_loss) / sum(test_examples)
        writer.add_scalar("Loss/test",test_epoch_loss,epoch)
        if config.neural_pretrainer.loss.type == "ctc":
            test_epoch_PER = sum(test_errors) / sum(test_phonemes)
            writer.add_scalar("PER/test",test_epoch_PER,epoch)
        elif config.neural_pretrainer.loss.type == "poisson":
            test_epoch_RMS = (sum(test_RMS) / sum(test_examples))**0.5
            writer.add_scalar("RMS/test",test_epoch_RMS,epoch)
            test_epoch_all_RMS = (sum(test_all_RMS) / sum(test_all_examples))**0.5
            writer.add_scalar("AllRMS/test",test_epoch_all_RMS,epoch)
        
        if config.neural_pretrainer.loss.type == "ctc":
            print("TRAIN: ")
            for pred, p, s in zip(train_preds[:20], train_phonograms[:20], train_sentences[:20]):
                print(f"Sentence: {s} \nPhonograms: {p} \nPrediction: {pred} \n")
            print("TEST: ")
            for pred, p, s in zip(test_preds[:20], test_phonograms[:20], test_sentences[:20]):
                print(f"Sentence: {s} \nPhonograms: {p} \nPrediction: {pred} \n")
            accelerator.print(f"{epoch=}: {train_epoch_loss=} {test_epoch_loss=} {train_epoch_PER=} {test_epoch_PER=}")
        elif config.neural_pretrainer.loss.type == "poisson":
            accelerator.print(f"{epoch=}: {train_epoch_loss=} {test_epoch_loss=} {train_epoch_RMS=} {test_epoch_RMS=}")

        # Save checkpoints 
        must_save = (config.trainer.save_every is not None and epoch%config.trainer.save_every == 0) or epoch in config.trainer.save_epochs
        if must_save:
            save_to_path = os.path.join(checkpoint_dir,f"EP{epoch}")
            if not os.path.exists(save_to_path):
                os.makedirs(save_to_path)
            accelerator.print(f"Saving checkpoint at epoch {epoch} to {save_to_path}")
            torch.save(model.encoder.state_dict(), os.path.join(save_to_path,"encoder.pth"))
            torch.save(model.decoder.state_dict(), os.path.join(save_to_path,"decoder.pth"))
            torch.save(dict(config), os.path.join(save_to_path,"config.pth"))
            if config.eval.save_data:
                if config.neural_pretrainer.loss.type == "poisson":
                    torch.save({
                        "train": {
                            "loss": train_loss,
                            "examples": train_examples,
                            "preds": train_preds,
                            "features": train_features,
                            "features_mask": train_features_mask,
                            "targets": train_targets,
                            "targets_mask": train_targets_mask,
                        },
                        "test": { 
                            "loss": test_loss,
                            "examples": test_examples,
                            "preds": test_preds,
                            "features": test_features,
                            "features_mask": test_features_mask,
                            "targets": test_targets,
                            "targets_mask": test_targets_mask,
                        },
                    }, os.path.join(save_to_path, "data.pth"))
                elif config.neural_pretrainer.loss.type == "ctc":
                    torch.save({
                        "train": {
                            "loss": train_loss,
                            "examples": train_examples,
                            "preds": train_preds,
                            "sentences": train_sentences,
                            "phonograms": train_phonograms,
                            "logits": train_logits,
                        },
                        "test": { 
                            "loss": test_loss,
                            "examples": test_examples,
                            "preds": test_preds,
                            "sentences": test_sentences,
                            "phonograms": test_phonograms,
                            "logits": test_logits,
                        },
                    }, os.path.join(save_to_path, "data.pth"))

            
        accelerator.wait_for_everyone()
        
    writer.flush()
    writer.close()

    # Save pretrained model
    torch.save(model.encoder.state_dict(), os.path.join(pt_dir,"encoder.pth"))
    torch.save(model.decoder.state_dict(), os.path.join(pt_dir,"decoder.pth"))
    torch.save(dict(config), os.path.join(pt_dir,"config.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)