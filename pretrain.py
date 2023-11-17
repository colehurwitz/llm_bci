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
from utils.eval_utils import format_ctc, word_error_count

DEFAULT_CONFIG_FILE = "configs/default_pretrain_config.yaml"

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False

def main(args):
    
    # Get configs
    config = update_config(DEFAULT_CONFIG_FILE, args.config_file if args.config_file != "none" else None) 
    config = update_config(config, config_from_kwargs(args.kwargs))

    accelerator = Accelerator()
    reset_seeds(config.seed)
    accelerator.print(config)

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
    vocab = data["train"]["eval"]["vocab"]
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
    encoder = NeuralEncoder(config.neural_config)
    model = NeuralPretrainer(encoder, config.neural_pretrainer, vocab_size, blank_id)
    if config.model_dir is not None:  
        print(f"Loading model from {config.model_dir}")
        model.encoder.load_state_dict(torch.load(os.path.join(config.model_dir,"encoder.bin")))
        model.decoder.load_state_dict(torch.load(os.path.join(config.model_dir,"decoder.bin")))
    else:
        print("Creating model from scratch")

    accelerator.print(model)
    accelerator.print(f"Encoder params: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad):,}")
    accelerator.print(f"Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=round(config.optimizer.warmup_ratio * len(train_dataloader) * config.trainer.num_epochs),
        num_training_steps=(len(train_dataloader) * config.trainer.num_epochs),
    )

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
        train_loss = 0.
        train_examples = 0
        train_errors = 0.
        train_phonemes = 0
 
        for step, (batch, phonograms, sentences) in enumerate(tqdm(train_dataloader)):
    
            outputs = model(**batch)
            loss = outputs.loss

            # Gather train metrics
            train_loss += loss.detach().float()
            train_examples += outputs.n_examples

            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            writer.add_scalar("Loss/train_iter",loss.detach().float()/outputs.n_examples, 1+step+(epoch-1)*len(train_dataloader))

            # Get phoneme error rate (every 15 steps to save time)
            if config.neural_pretrainer.loss.type == "ctc" and step%15 == 0:
                preds = torch.argmax(outputs.outputs, -1)
                preds = [" ".join(format_ctc(pred, vocab, blank_id)) for pred in preds]
                phonograms = [" ".join(p) for p in phonograms]
                errors, phonemes = word_error_count(preds, phonograms)
                train_errors += errors
                train_phonemes += phonemes
                
                preds = [pred.replace(" ","").replace("SIL"," SIL ") for pred in preds]
                phonograms = [p.replace(" ","").replace("SIL"," SIL ") for p in phonograms]
                for pred, p, s in zip(preds, phonograms, sentences):
                    print(f"Sentence: {s} \nPhonograms: {p} \nPrediction: {pred} \n")
                writer.add_scalar("PER/train_iter",errors/phonemes, 1+step+(epoch-1)*len(train_dataloader))


        # Log to tensorboard
        train_epoch_loss = train_loss / train_examples
        writer.add_scalar("Loss/train",train_epoch_loss,epoch)
        train_epoch_PER = train_errors / train_phonemes
        writer.add_scalar("PER/train",train_PER,epoch)


        print(f"Evaluation epoch {epoch}")
        model.eval()

        # Test metrics
        test_loss = 0.
        test_examples = 0
        test_errors = 0.
        test_phonemes = 0

        for step, (batch, phonograms, sentences) in enumerate(tqdm(test_dataloader)):

            # Forward pass
            with torch.no_grad():
                outputs = model(**batch)
            
            # Gather test metrics
            test_loss += outputs.loss.detach().float()
            test_examples += outputs.n_examples

            # Log to tensorboard
            writer.add_scalar("Loss/test_iter",loss.detach().float()/outputs.n_examples,1+step+(epoch-1)*len(test_dataloader))
            
            # Get phoneme error rate
            if config.neural_pretrainer.loss.type == "ctc":
                preds = torch.argmax(outputs.outputs,-1)
                preds = [" ".join(format_ctc(pred, vocab, blank_id)) for pred in preds]
                phonograms = [" ".join(p) for p in phonograms]
                errors, phonemes = word_error_count(preds, phonograms)
                test_errors += errors
                test_phonemes += phonemes
                
                for p, s in zip(preds, sentences):
                    print(f"Sentence: {s} \nPrediction: {p} \n")
                writer.add_scalar("PER/test_iter",errors/phonemes, 1+step+(epoch-1)*len(test_dataloader))
        
        # Log to tensorboard
        test_epoch_loss = test_loss / test_examples
        writer.add_scalar("Loss/test",test_epoch_loss,epoch)
        test_epoch_PER = test_errors / test_phonemes
        writer.add_scalar("PER/test",test_epoch_PER,epoch)

        accelerator.print(f"{epoch=}: {train_epoch_loss=} {test_epoch_loss=} {test_epoch_PER=} ")

        # Save checkpoints 
        must_save = (config.trainer.save_every is not None and epoch%config.trainer.save_every == 0) or epoch in config.trainer.save_epochs
        if must_save:
            accelerator.print(f"Saving checkpoint at epoch {epoch}")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir,f"EP{epoch}"))

        accelerator.wait_for_everyone()
        
    writer.flush()
    writer.close()

    # Save pretrained model
    torch.save(model.encoder.state_dict(), os.path.join(pt_dir,"encoder.bin"))
    torch.save(model.decoder.state_dict(), os.path.join(pt_dir,"decoder.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)