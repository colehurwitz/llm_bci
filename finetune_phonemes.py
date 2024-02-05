import os
import sys
from functools import partial

import wandb 
import argparse
from tqdm import tqdm
from g2p_en import G2p

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR, StepLR

import accelerate
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, LlamaConfig
from peft import LoraConfig, get_peft_model

from utils.config_utils import update_config, config_from_kwargs, ParseKwargs, DictConfig
from utils.data_utils import PhonemesFinetuneDataset, ft_pad_collate_fn, prepare_phonemes_data
from utils.eval_utils import word_error_count

from models.phoneme_llama import PhonemeLlama

DEFAULT_CONFIG_FILE = "configs/finetune.yaml"

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False

def main(args):
    
    # Get configs
    config = update_config(DEFAULT_CONFIG_FILE, args.config_file if args.config_file != "none" else None) 
    config = update_config(config, config_from_kwargs(args.kwargs))
    
    savestring = config.savestring
    
    # Load checkpoint
    load_checkpoint = False
    checkpoint_dir = os.path.join(config.dirs.checkpoint_dir,savestring)
    
    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        load_checkpoint=True
        load_ep = max([int(f.split("-")[0].replace("EP","")) for f in os.listdir(checkpoint_dir)])
        load_step = max([int(f.split("-")[1].replace("STEP","")) for f in os.listdir(checkpoint_dir) if int(f.split("-")[0].replace("EP","")) == load_ep])
        load_dir = os.path.join(checkpoint_dir,f"EP{load_ep}-STEP{load_step}")
        config = DictConfig(torch.load(os.path.join(load_dir, "config.pth")))

    # Initialize wand and biases
    if config.log_to_wandb:
        run = wandb.init(config.wandb_project)
        config = update_config(config, config_from_kwargs(wandb.config, convert=False))

    # Initialize accelerator
    accelerator = accelerate.Accelerator(
        step_scheduler_with_optimizer=config.optimizer.scheduler in ["linear","cosine"], 
        split_batches=True,
        gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps
    )
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
    reset_seeds(config.seed)
    accelerator.print(config)
    accelerator.print(f"Starting run {savestring}")
    
    # Create saving paths
    ft_dir = os.path.join(config.dirs.ft_dir,savestring)
    log_dir = os.path.join(config.dirs.log_dir,savestring)
    if not os.path.exists(checkpoint_dir) and accelerator.is_main_process:
        os.makedirs(checkpoint_dir)

    # Init tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # LoRA configuration
    peft_config = LoraConfig(
        inference_mode=False, r=config.lora.r, lora_alpha=config.lora.alpha, lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules
    )


    if config.debug:
        accelerator.print(f"Creating small model for debugging")
        llama_config = LlamaConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=32,  num_attention_heads=4)
        model = PhonemeLlama.from_config(llama_config, config.coupler)
    else:
        accelerator.print(f"Loading pretrained LLM")
        model = PhonemeLlama.from_pretrained(config.dirs.model_dir, config.coupler)  

    # Load LLM with LoRA adapter
    if load_checkpoint:
        accelerator.print(f"Loading checkpoint at epoch {load_ep}, step {load_step}")
        model.load_coupler(load_dir)
        adapter_file = os.path.join(load_dir, "adapter_config.json")
        if os.path.isfile(adapter_file):
            accelerator.print(f"Loading adapter from checkpoint at epoch {load_ep}, step {load_step}")
            model.load_adapter(load_dir, is_trainable=(not config.freeze_llm))        
    
    if config.freeze_llm:
        accelerator.print(model)
        accelerator.print(f"Coupler params: {sum(p.numel() for p in model.coupler.parameters() if p.requires_grad):,}")
        accelerator.print(f"Decoder is frozen")
        for param in model.decoder.parameters():
            param.requires_grad = False
    else:
        if not model._is_peft:
            accelerator.print("Creating LORA adapter")
            model.create_adapter(peft_config) 
        accelerator.print(model)
        accelerator.print(f"Coupler params: {sum(p.numel() for p in model.coupler.parameters() if p.requires_grad):,}")
        accelerator.print("Decoder: ")
        if accelerator.is_main_process:
            model.decoder.print_trainable_parameters()

    # Load tokenizer and grapheme to phoneme translator
    tokenizer = AutoTokenizer.from_pretrained(config.dirs.tokenizer_dir, padding_side='right', add_bos_token=False, add_eos_token=False)
    pad_id = tokenizer.eos_token_id
    config["generation"]["pad_token_id"] = pad_id
    g2p = G2p()
    
    # Load preprocessed dataset
    accelerator.print(f"Loading data from {os.path.join(config.dirs.data_dir, config.data_file)}")
    data = torch.load(os.path.join(config.dirs.data_dir, config.data_file))
    train_data = {k: v[:config.trainer.train_len] for k,v in data["train"].items()}
    train_data = prepare_phonemes_data(train_data, tokenizer, g2p, config.prompt, config.stack)
    test_data = {k: v[:config.trainer.test_len] for k,v in data["test"].items()}
    test_data = prepare_phonemes_data(test_data, tokenizer, g2p, config.prompt, config.stack)
    
    train_dataset = PhonemesFinetuneDataset(train_data)
    test_dataset = PhonemesFinetuneDataset(test_data)
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=partial(ft_pad_collate_fn,config.noise,config.mask,pad_id,"train"), batch_size=config.trainer.train_batch_size, pin_memory=True, drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=partial(ft_pad_collate_fn,config.noise,config.mask,pad_id,"test"), batch_size=config.trainer.test_batch_size, pin_memory=True, drop_last=True,
    )


    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)
    # for pn, p in model.named_parameters():
    #     if p.requires_grad:
    #         accelerator.print(pn)

    if config.optimizer.scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.optimizer.warmup_pct*config.trainer.num_epochs*len(train_dataloader),
            num_training_steps=config.trainer.num_epochs*len(train_dataloader),
        )
    elif config.optimizer.scheduler == "cosine":
        lr_scheduler = OneCycleLR(
            optimizer=optimizer,
            total_steps=config.trainer.num_epochs*len(train_dataloader),
            max_lr=config.optimizer.lr,
            pct_start=config.optimizer.warmup_pct,
            div_factor=10
        )
    elif config.optimizer.scheduler == "step":
        lr_scheduler = StepLR(
            optimizer, 
            step_size=1, 
            gamma=config.optimizer.gamma)
    else:
        raise Exception(f"Scheduler '{config.optimizer.scheduler}' not implemented")


    # Prepare model for distributed training
    model, train_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, test_dataloader, optimizer, lr_scheduler
    )

    # Train
    if load_checkpoint:
        accelerator.print(f"Resuming finetuning at epoch {load_ep}, step {load_step}")
        start_epoch = load_ep
        start_step = load_step+1
    else:
        accelerator.print("Start finetuning")
        start_epoch = 1
        start_step = 0

    for epoch in range(start_epoch, config.trainer.num_epochs+1):
        accelerator.print(f"Epoch {epoch}")
        model.train()

        # Train metrics
        train_loss = []
        train_examples = []
        train_errors = []
        train_words = []
        
        # To save
        train_sentences = []
        train_true_phonemes = []
        train_pred_phonemes = []
        train_preds = []

        for step, (model_inputs, sentences, true_phonemes, pred_phonemes) in enumerate(tqdm(train_dataloader)):
            # Skip first steps if checkpoint was loaded
            if step < start_step:
                continue

            # Forward pass
            with accelerator.accumulate(model):
                outputs = model(**model_inputs)
                loss = outputs.loss
                examples = torch.tensor(outputs.n_examples).to(loss)

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                if config.optimizer.scheduler in ["linear","cosine"]:
                    lr_scheduler.step()
                optimizer.zero_grad()

            # Gather train metrics
            # sentences = accelerate.utils.gather_object(sentences)
            # phonemes = accelerate.utils.gather_object(phonemes)
            # pred_phonemes = accelerate.utils.gather_object(pred_phonemes)

            # Loss
            train_loss.append(accelerator.gather(loss).mean().detach().item())
            train_examples.append(accelerator.gather(examples).sum().detach().item())

            # Word error rate (WER)
            # labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=-100, pad_first=True)
            # labels = accelerator.gather_for_metrics(labels)
            labels = outputs.labels
            labels_lens = [len(lab[lab != -100]) for lab in labels]
            
            preds = torch.argmax(outputs.logits, -1)
            # preds = accelerator.pad_across_processes(preds, dim=1, pad_index=tokenizer.eos_token_id, pad_first=True)
            # preds = accelerator.gather_for_metrics(preds)
            preds = [p[(-labels_lens[i]-1):-1].detach().cpu() for i, p in enumerate(preds)]
            preds = [tokenizer.decode(p.cpu().squeeze(), skip_special_tokens=True) for p in preds]
            errors, words = word_error_count(preds, sentences)

            train_errors.append(errors)
            train_words.append(words)
            train_sentences += sentences
            train_true_phonemes += true_phonemes
            train_pred_phonemes += pred_phonemes
            train_preds += preds
            
            # Log to tensorboard
            if accelerator.is_main_process:
                writer.add_scalar("Loss/train_iter",train_loss[-1], 1+step+(epoch-1)*len(train_dataloader))
                writer.add_scalar("WER/train_iter",errors/words, 1+step+(epoch-1)*len(train_dataloader))
            
            # Evaluation condition
            if (1+ len(train_dataloader)*(epoch-1) + step) % config.trainer.eval_every == 0:
                
                # Test metrics
                test_loss = []
                test_examples = []
                test_errors = []
                test_words = []
                test_aid_errors = []
                test_aid_words = []

                # To save
                test_sentences = []
                test_true_phonemes = []
                test_pred_phonemes = []
                test_preds = [] 
                test_aid_preds = []

                accelerator.print(f"Evaluation at step {(1+len(train_dataloader)*(epoch-1) + step)}")
                model.eval()

                for test_step, (model_inputs, prompt_inputs, sentences, true_phonemes, pred_phonemes) in enumerate(tqdm(test_dataloader)):
                    
                    with torch.no_grad():
                        # Forward pass
                        outputs = model(**model_inputs)
                        loss = outputs.loss
                        examples = torch.tensor(outputs.n_examples).to(loss)

                    labels = outputs.labels
                    labels_lens = [len(lab[lab != -100]) for lab in labels]
                    aid_preds = torch.argmax(outputs.logits, -1)
                    aid_preds = [p[(-labels_lens[i]-1):-1].detach().cpu() for i, p in enumerate(aid_preds)]
                    aid_preds = [tokenizer.decode(p.cpu().squeeze(), skip_special_tokens=True) for p in aid_preds]
                    aid_errors, aid_words = word_error_count(aid_preds, sentences)
                    
                    test_aid_errors.append(aid_errors)
                    test_aid_words.append(aid_words)
                    test_aid_preds += aid_preds

                    # Inference  
                    preds = model.generate(
                            **prompt_inputs,
                            **config.generation,
                            synced_gpus=is_ds_zero_3
                        )

                    # Gather test metrics
                    # sentences = accelerate.utils.gather_object(sentences)
                    # phonemes = accelerate.utils.gather_object(phonemes)
                    # pred_phonemes = accelerate.utils.gather_object(pred_phonemes)

                    # Loss
                    test_loss.append(accelerator.gather(loss).mean().detach().item())
                    test_examples.append(accelerator.gather(examples).sum().detach().item())

                    # Word error rate (WER)
                    # prompts = accelerator.pad_across_processes(prompt_batch["input_ids"], dim=1, pad_index=tokenizer.eos_token_id, pad_first=True)
                    # prompts = accelerator.gather_for_metrics(prompts)
                    prompt_lens = [len(prompt_inputs["input_ids"][i]) for i in range(len(prompt_inputs["input_ids"]))]*config.generation.num_return_sequences

                    # preds = accelerator.pad_across_processes(preds, dim=1, pad_index=tokenizer.eos_token_id, pad_first=True)
                    # preds = accelerator.gather_for_metrics(preds)
                    preds = [tokenizer.decode(p[prompt_lens[i]:].detach().cpu().squeeze(), skip_special_tokens=True) for i, p in enumerate(preds)]
                    errors, words = word_error_count(preds, sentences*config.generation.num_return_sequences)

                    test_errors.append(errors)
                    test_words.append(words)
                    test_sentences += sentences
                    test_true_phonemes += true_phonemes
                    test_pred_phonemes += pred_phonemes
                    test_preds += preds

                    # Log to tensorboard
                    # if accelerator.is_main_process:
                    #     writer.add_scalar("Loss/test_iter",test_loss[-1], 1+test_step+evals*len(test_dataloader))
                    #     writer.add_scalar("WER/test_iter",errors/words, 1+test_step+evals*len(test_dataloader))

                # Eval metrics
                train_epoch_loss = sum([l*n for l,n in zip(train_loss,train_examples)]) / sum(train_examples)
                train_epoch_WER = sum(train_errors) / sum(train_words)
                test_epoch_loss = sum([l*n for l,n in zip(test_loss,test_examples)]) / sum(test_examples)
                test_epoch_WER = sum(test_errors) / sum(test_words)   
                test_epoch_aid_WER = sum(test_aid_errors) / sum(test_aid_words)  
                eval_step = (1+len(train_dataloader)*(epoch-1) + step)
                accelerator.print(f"{savestring=} {eval_step=}: {train_epoch_loss=} {train_epoch_WER=} {test_epoch_loss=} {test_epoch_WER=} {test_epoch_aid_WER=}")  
                
                # Log to terminal
                accelerator.print("TRAIN: ")
                for pred, pred_p, p, s in zip(train_preds[:20], train_pred_phonemes[:20], train_true_phonemes[:20], train_sentences[:20]):
                    accelerator.print(f"Sentence: {s}\n True phonemes: {p}\n Predicted phonemes: {pred_p}\n Prediction: {pred}\n")
                accelerator.print("TEST: ")
                for pred, aid_pred, pred_p, p, s in zip(test_preds[:20], test_aid_preds[:20], test_pred_phonemes[:20], test_true_phonemes[:20], test_sentences[:20]):
                    accelerator.print(f"Sentence: {s}\n True phonemes: {p}\n Predicted phonemes: {pred_p}\n Aided prediction: {aid_pred}\n Prediction: {pred}\n")
                
                # Log to tensorboard/wandb
                if accelerator.is_main_process:
                    # Log to tensorboard
                    writer.add_scalar("Loss/train",train_epoch_loss,1+ len(train_dataloader)*(epoch-1) + step)
                    writer.add_scalar("WER/train",train_epoch_WER,1+ len(train_dataloader)*(epoch-1) + step)
                    writer.add_scalar("Loss/test",test_epoch_loss,1+ len(train_dataloader)*(epoch-1) + step)
                    writer.add_scalar("WER/test",test_epoch_WER,1+ len(train_dataloader)*(epoch-1) + step)
                    writer.add_scalar("AidWER/test",test_epoch_aid_WER,1+ len(train_dataloader)*(epoch-1) + step)

                    # Log to wandb
                    if config.log_to_wandb:
                        wandb.log({
                            "step": 1+ len(train_dataloader)*(epoch-1) + step,
                            "train_epoch_loss": train_epoch_loss,
                            "test_epoch_loss": test_epoch_loss ,
                            "train_epoch_WER": train_epoch_WER,
                            "test_epoch_WER": test_epoch_WER,
                            "test_epoch_aid_WER": test_epoch_aid_WER,
                        })
                
                # End evaluation
                model.train()     

            # Save checkpoints
            if (1+ len(train_dataloader)*(epoch-1) + step) % config.trainer.save_every == 0 and accelerator.is_main_process:
                save_to_path = os.path.join(checkpoint_dir,f"EP{epoch}-STEP{step}")
                if not os.path.exists(save_to_path):
                    os.makedirs(save_to_path)

                accelerator.print(f"Saving checkpoint at step {1+ len(train_dataloader)*(epoch-1) + step} to {save_to_path}")
                if model._is_peft:
                    accelerator.unwrap_model(model).save_adapter(save_to_path)
                accelerator.unwrap_model(model).save_coupler(save_to_path)
                torch.save(dict(config), os.path.join(save_to_path,"config.pth"))
                
                if config.trainer.save_data:
                    torch.save({
                        "train": {
                            "sentences": train_sentences,
                            "true_phonemes": train_true_phonemes,
                            "pred_phonemes": train_pred_phonemes,
                            "preds": train_preds,
                        },
                        "test": { 
                            "sentences": test_sentences,
                            "true_phonemes": test_true_phonemes,
                            "pred_phonemes": test_pred_phonemes,
                            "preds": test_preds,
                            "aid_preds": test_aid_preds,
                        },
                    }, os.path.join(save_to_path, "data.pth"))
                    
                # Reset train metrics
                train_loss = []
                train_errors = []
                train_words = []
                
                # Reset train to save
                train_sentences = []
                train_true_phonemes = []
                train_pred_phonemes = []
                train_preds = []

        if config.optimizer.scheduler in ["step"]:
            lr_scheduler.step()
        
        start_step = 0

    writer.flush()
    writer.close()

    # Save finetuned model¡
    if not config.freeze_llm:
        model.merge_adapter()
    model.save_pretrained(ft_dir)
    accelerator.print("Training done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)
