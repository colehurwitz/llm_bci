import os
import argparse
from tqdm import tqdm
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, LlamaForCausalLM

from peft import LoraConfig

from utils.config_utils import update_config, config_from_kwargs, ParseKwargs
from utils.data_utils import PhonemesFinetuneDataset, ft_pad_collate_fn
from utils.eval_utils import word_error_count

DEFAULT_CONFIG_FILE = "configs/default_finetune_config.yaml"

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False

def main(args):
    
    config = update_config(DEFAULT_CONFIG_FILE, args.config_file if args.config_file != "none" else None) 
    config = update_config(config, config_from_kwargs(args.kwargs))

    accelerator = Accelerator()
    accelerator.print(config)
    reset_seeds(config.seed)

    # Create saving paths
    checkpoint_dir = os.path.join(config.dirs.checkpoint_dir,config.savestring)
    ft_dir = os.path.join(config.dirs.ft_dir,config.savestring)
    log_dir = os.path.join(config.dirs.log_dir,config.savestring)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Init tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    peft_config = LoraConfig(
        inference_mode=False, r=config.lora.r, lora_alpha=config.lora.alpha, lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules
    )


    # Load model with peft adapter for decoder
    model = LlamaForCausalLM.from_pretrained(config.dirs.model_dir)  
    model = get_peft_model(model, peft_config) 

    accelerator.print(model)
    model.decoder.print_trainable_parameters()

    # Load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, padding_side='right')
    pad_id = tokenizer.eos_token_id

    # Load preprocessed dataset
    data = torch.load(os.path.join(config.dirs.data_dir, config.data_file))
    train_data = data["train"]
    test_data = data["test"]

    train_dataset = PhonemesDataset(train_data, split="train", len=config.trainer.train_len)
    test_dataset = PhonemesDataset(test_data, split="test", len=config.trainer.test_len)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=partial(phonemes_pad_collate_fn,pad_id), batch_size=config.trainer.train_batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=partial(phonemes_pad_collate_fn,pad_id), batch_size=config.trainer.test_batch_size, pin_memory=True
    )


    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)
    # for pn, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(pn)

    if config.optimizer.scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.optimizer.warmup_epochs*len(train_dataloader),
            num_training_steps=config.trainer.num_epochs*len(train_dataloader),
        )
    elif config.optimizer.scheduler == "cosine":
        lr_scheduler = OneCycleLR(
            optimizer=optimizer,
            total_steps=config.trainer.num_epochs*len(train_dataloader),
            max_lr=config.optimizer.lr,
            pct_start=config.optimizer.warmup_epochs / config.trainer.num_epochs
        )
    else:
        raise Exception(f"Scheduler '{config.optimizer.scheduler}' not implemented")


    # Prepare model for distributed training
    model, train_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, test_dataloader, optimizer, lr_scheduler
    )


    # Train
    for epoch in range(1,config.trainer.num_epochs+1):
        
        print(f"Epoch {epoch}")
        model.train()

        # Train metrics
        train_loss = []
        train_examples = []
        train_errors = []
        train_words = []
        
        # To save
        train_sentences = []
        train_phonograms = []
        train_preds = []

        for step, (batch, phonograms, sentences) in enumerate(tqdm(train_dataloader)):
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Gather train metrics
            train_loss.append(loss.detach().item())
            train_examples.append(len(outputs.logits))
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Gather other metrics
            logits = outputs.outputs.detach().cpu()
            preds = torch.argmax(logits, -1)
            preds = [tokenizer.decode(p.cpu().squeeze(), skip_special_tokens=True) for p in preds]
            
            phonograms = [" ".join(p) for p in phonograms]
            phonograms = [p.replace(" ","").replace("SIL"," SIL ") for p in phonograms]
            errors, words = word_error_count(preds, sentences)
            train_errors.append(errors)
            train_words.append(words)
            train_preds += preds
            train_sentences += sentences
            train_phonograms += phonograms
            train_logits.append(logits)
            
            # Log to tensorboard
            writer.add_scalar("Loss/train_iter",train_loss[-1]/train_examples[-1], 1+step+(epoch-1)*len(train_dataloader))
            writer.add_scalar("WER/train_iter",errors/words, 1+step+(epoch-1)*len(train_dataloader))

        # Log to tensorboard
        train_epoch_loss = sum(train_loss) / sum(train_examples)
        writer.add_scalar("Loss/train",train_epoch_loss,epoch)
        train_epoch_WER = sum(train_errors) / sum(train_words)
        writer.add_scalar("WER/train",train_epoch_WER,epoch)

        print(f"Evaluation epoch {epoch}")
        model.eval()

        # Test metrics
        test_loss = []
        test_examples = []
        test_errors = []
        test_words = []

        # To save
        test_sentences = []
        test_phonograms = []
        test_preds = []

        for step, (batch, phonograms, sentences) in enumerate(tqdm(test_dataloader)):

            # Forward pass
            with torch.no_grad():
                outputs = model(**batch)
            
            # Gather test metrics
            test_loss.append(outputs.loss.detach().item())
            test_examples.append(outputs.n_examples)

            # Gather other metrics
            logits = outputs.outputs.detach().cpu()
            preds = torch.argmax(logits, -1)
            preds = [tokenizer.decode(p.cpu().squeeze(), skip_special_tokens=True) for p in preds]
            
            phonograms = [" ".join(p) for p in phonograms]
            phonograms = [p.replace(" ","").replace("SIL"," SIL ") for p in phonograms]
            errors, words = word_error_count(preds, sentences)
            train_errors.append(errors)
            train_words.append(words)
            train_preds += preds
            train_sentences += sentences
            train_phonograms += phonograms
            train_logits.append(logits)

            # Log to tensorboard
            writer.add_scalar("Loss/test_iter",test_loss[-1]/test_examples[-1], 1+step+(epoch-1)*len(test_dataloader))
            writer.add_scalar("WER/test_iter",errors/words, 1+step+(epoch-1)*len(test_dataloader))

        # Log to tensorboard
        test_epoch_loss = sum(test_loss) / sum(test_examples)
        writer.add_scalar("Loss/test",test_epoch_loss,epoch)
        test_epoch_WER = sum(test_errors) / sum(test_words)
        writer.add_scalar("WER/test",test_epoch_WER,epoch)

        # Log to wandb
        if config.log_to_wandb:
            wandb.log({
                "epoch": epoch,
                "train_epoch_loss": train_epoch_loss,
                "test_epoch_loss": test_epoch_loss ,
                "train_epoch_WER": train_epoch_WER,
                "test_epoch_WER": test_epoch_WER,
            })

        # Log to terminal
        print("TRAIN: ")
        for pred, p, s in zip(train_preds[:20], train_phonograms[:20], train_sentences[:20]):
            print(f"Sentence: {s} \nPhonograms: {p} \nPrediction: {pred} \n")
        print("TEST: ")
        for pred, p, s in zip(test_preds[:20], test_phonograms[:20], test_sentences[:20]):
            print(f"Sentence: {s} \nPhonograms: {p} \nPrediction: {pred} \n")
        accelerator.print(f"{epoch=}: {train_epoch_loss=} {train_epoch_WER=} {test_epoch_loss=} {test_epoch_WER=}")

        # Save checkpoints (not merged because we don't want to delete the adapter yet)
        must_save = (config.trainer.save_every is not None and epoch%config.trainer.save_every == 0) or epoch in config.trainer.save_epochs
        if must_save:
            save_to_path = os.path.join(checkpoint_dir,f"EP{epoch}")
            if not os.path.exists(save_to_path):
                os.makedirs(save_to_path)
            accelerator.print(f"Saving checkpoint at epoch {epoch} to {save_to_path}")
            model.save_pretrained(save_to_path)
            if config.eval.save_data:
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

    # Save finetuned model
    model.save_pretrained(ft_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)