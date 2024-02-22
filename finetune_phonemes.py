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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig, LlamaConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from utils.config_utils import update_config, config_from_kwargs, ParseKwargs, DictConfig
from utils.data_utils import PhonemesFinetuneDataset, ft_pad_collate_fn, prepare_phonemes_data
from utils.eval_utils import word_error_count

from models.phoneme_llm import PhonemeLLM

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

    # Initialize accelerator
    accelerator = accelerate.Accelerator(
        step_scheduler_with_optimizer=config.optimizer.scheduler in ["linear","cosine"], 
        split_batches=True,
    )
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    reset_seeds(config.seed)
    accelerator.print(config)
    accelerator.print(f"Starting run {savestring}")


    # Prepare logging
    checkpoint_dir = os.path.join(config.dirs.checkpoint_dir,savestring)
    if not os.path.exists(checkpoint_dir) and accelerator.is_main_process:
        os.makedirs(checkpoint_dir)

    log_dir = os.path.join(config.dirs.log_dir,savestring)
    writer = SummaryWriter(log_dir=log_dir)

    if config.log_to_wandb:
        run = wandb.init(config.wandb_project)
        config = update_config(config, config_from_kwargs(wandb.config, convert=False))
    

    # LoRA configuration
    peft_config = LoraConfig(
        inference_mode=False, r=config.lora.r, lora_alpha=config.lora.alpha, lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules, modules_to_save=config.lora.modules_to_save,
    )

    if config.debug:
        accelerator.print(f"Creating small LLaMA model for debugging")
        llm_config = LlamaConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=32,  num_attention_heads=4)
        llm = AutoModelForCausalLM.from_config(llm_config)
    else:
        accelerator.print(f"Loading pretrained LLM")
        llm = AutoModelForCausalLM.from_pretrained(config.dirs.llm_dir) 
    model = PhonemeLLM(llm, config.coupler)   

    if config.freeze_llm:
        for param in model.llm.parameters():
            param.requires_grad = False
        accelerator.print(model)
        accelerator.print(f"Coupler params: {sum(p.numel() for p in model.coupler.parameters() if p.requires_grad):,}")
        accelerator.print(f"LLM is frozen")
    else:
        accelerator.print("Creating LORA adapter")
        model.create_lora_adapter(peft_config) 
        accelerator.print(model)
        accelerator.print(f"Coupler params: {sum(p.numel() for p in model.coupler.parameters() if p.requires_grad):,}")
        accelerator.print("LLM: ")
        if accelerator.is_main_process:
            model.llm.print_trainable_parameters()

    # Load tokenizer and grapheme to phoneme translator
    tokenizer = AutoTokenizer.from_pretrained(config.dirs.tokenizer_dir, padding_side='left', add_bos_token=False, add_eos_token=False)
    pad_id = tokenizer.eos_token_id
    config["generation"]["pad_token_id"] = pad_id
    g2p = G2p()
    
    # Load preprocessed dataset
    accelerator.print(f"Loading data from {os.path.join(config.dirs.data_dir, config.data_file)}")
    data = torch.load(os.path.join(config.dirs.data_dir, config.data_file))
    train_data = {k: v[:config.trainer.train_len] if config.trainer.train_len != -1 else v for k,v in data["train"].items()}
    train_data = prepare_phonemes_data(train_data, tokenizer, g2p, config.prompt)
    test_data = {k: v[:config.trainer.test_len] if config.trainer.test_len != -1 else v for k,v in data["test"].items()}
    test_data = prepare_phonemes_data(test_data, tokenizer, g2p, config.prompt)
    
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
    #     print(pn, p.requires_grad)

    if config.optimizer.scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=round(config.optimizer.warmup_pct*config.trainer.num_epochs*len(train_dataloader)/config.optimizer.gradient_accumulation_steps),
            num_training_steps=config.trainer.num_epochs*len(train_dataloader)//config.optimizer.gradient_accumulation_steps,
        )
    elif config.optimizer.scheduler == "cosine":
        lr_scheduler = OneCycleLR(
            optimizer=optimizer,
            total_steps=config.trainer.num_epochs*len(train_dataloader)//config.optimizer.gradient_accumulation_steps,
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
    accelerator.print("Start finetuning")
    global_step = 0
    
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

    for epoch in range(1, config.trainer.num_epochs+1):
        accelerator.print(f"Epoch {epoch}")
        model.train()

        for step, (model_inputs, sentences, true_phonemes, pred_phonemes) in enumerate(tqdm(train_dataloader)):

            # Perform gradient accumulation
            if (global_step + 1) % config.optimizer.gradient_accumulation_steps == 0:
                outputs = model(**model_inputs)
                loss = outputs.loss
                examples = torch.tensor(outputs.n_examples).to(loss)
                accelerator.backward(loss / config.optimizer.gradient_accumulation_steps)
                optimizer.step()
                if config.optimizer.scheduler in ["linear","cosine"]:
                    lr_scheduler.step()
                optimizer.zero_grad()
            else:
                with accelerator.no_sync(model):
                    outputs = model(**model_inputs)
                    loss = outputs.loss
                    examples = torch.tensor(outputs.n_examples).to(loss)
                    accelerator.backward(loss / config.optimizer.gradient_accumulation_steps)

            # Gather train metrics
            # sentences = accelerate.utils.gather_object(sentences)
            # phonemes = accelerate.utils.gather_object(phonemes)
            # pred_phonemes = accelerate.utils.gather_object(pred_phonemes)

            # Loss
            train_loss.append(accelerator.gather(loss).sum().detach().item())
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
            preds = [tokenizer.decode(p.cpu().squeeze(), skip_special_tokens=True).lower().strip() for p in preds]
            errors, words = word_error_count(preds, sentences)

            train_errors.append(errors)
            train_words.append(words)
            train_sentences += sentences
            train_true_phonemes += true_phonemes
            train_pred_phonemes += pred_phonemes
            train_preds += preds
            
            # Log to tensorboard
            if accelerator.is_main_process:
                writer.add_scalar("Loss/train_iter",train_loss[-1] / train_examples[-1], global_step)
                writer.add_scalar("WER/train_iter",errors/words, global_step)
            
            # Evaluation condition
            if (global_step + 1) % config.trainer.eval_every == 0:
                
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

                accelerator.print(f"Evaluation at step {global_step}")
                model.eval()

                for test_step, (model_inputs, prompt_inputs, sentences, true_phonemes, pred_phonemes) in enumerate(tqdm(test_dataloader)):
                    
                    with torch.no_grad() as A, accelerator.no_sync(model) as B:
                        outputs = model(**model_inputs)
                        loss = outputs.loss
                        examples = torch.tensor(outputs.n_examples).to(loss)

                    # Loss
                    test_loss.append(accelerator.gather(loss).sum().detach().item())
                    test_examples.append(accelerator.gather(examples).sum().detach().item())

                    labels = outputs.labels
                    labels_lens = [len(lab[lab != -100]) for lab in labels]
                    aid_preds = torch.argmax(outputs.logits, -1)
                    aid_preds = [p[(-labels_lens[i]-1):-1].detach().cpu() for i, p in enumerate(aid_preds)]
                    aid_preds = [tokenizer.decode(p.cpu().squeeze(), skip_special_tokens=True).lower().strip() for p in aid_preds]
                    aid_errors, aid_words = word_error_count(aid_preds, sentences)
                    
                    test_aid_errors.append(aid_errors)
                    test_aid_words.append(aid_words)
                    test_aid_preds += aid_preds

                    # Inference  
                    preds = model.predict(
                            **prompt_inputs,
                            **config.generation,
                            synced_gpus=is_ds_zero_3
                        )

                    # Gather test metrics
                    # sentences = accelerate.utils.gather_object(sentences)
                    # phonemes = accelerate.utils.gather_object(phonemes)
                    # pred_phonemes = accelerate.utils.gather_object(pred_phonemes)


                    # Word error rate (WER)
                    # prompts = accelerator.pad_across_processes(prompt_batch["input_ids"], dim=1, pad_index=tokenizer.eos_token_id, pad_first=True)
                    # prompts = accelerator.gather_for_metrics(prompts)
                    # preds = accelerator.pad_across_processes(preds, dim=1, pad_index=tokenizer.eos_token_id, pad_first=True)
                    # preds = accelerator.gather_for_metrics(preds)
                    preds = [tokenizer.decode(p.detach().cpu().squeeze(), skip_special_tokens=True).lower().strip() for i, p in enumerate(preds)]
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
                train_epoch_loss = sum(train_loss) / sum(train_examples)
                train_epoch_WER = sum(train_errors) / sum(train_words)
                test_epoch_loss = sum(test_loss) / sum(test_examples)
                test_epoch_WER = sum(test_errors) / sum(test_words)   
                test_epoch_aid_WER = sum(test_aid_errors) / sum(test_aid_words)  

                accelerator.print(f"{savestring=} {global_step=}:" + "\n" + \
                                  f"{train_epoch_loss=} {train_epoch_WER=}" + "\n" + \
                                  f"{test_epoch_loss=} {test_epoch_WER=} {test_epoch_aid_WER=}")  
                
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
                    writer.add_scalar("Loss/train",train_epoch_loss,global_step)
                    writer.add_scalar("WER/train",train_epoch_WER,global_step)
                    writer.add_scalar("Loss/test",test_epoch_loss,global_step)
                    writer.add_scalar("WER/test",test_epoch_WER,global_step)
                    writer.add_scalar("AidWER/test",test_epoch_aid_WER,global_step)

                    # Log to wandb
                    if config.log_to_wandb:
                        wandb.log({
                            "step": global_step,
                            "train_epoch_loss": train_epoch_loss,
                            "test_epoch_loss": test_epoch_loss ,
                            "train_epoch_WER": train_epoch_WER,
                            "test_epoch_WER": test_epoch_WER,
                            "test_epoch_aid_WER": test_epoch_aid_WER,
                        })
                
                # Reset train metrics
                train_loss = []
                train_examples = []
                train_errors = []
                train_words = []
                        
                # Reset train to save
                train_sentences = []
                train_true_phonemes = []
                train_pred_phonemes = []
                train_preds = []

                # End evaluation
                model.train()     


            # Save checkpoints
            if (global_step + 1) % config.trainer.save_every == 0:
                save_to_path = os.path.join(checkpoint_dir,f"STEP{global_step}")
                if not os.path.exists(save_to_path) and accelerator.is_main_process:
                    os.makedirs(save_to_path)

                accelerator.print(f"Saving checkpoint at step {global_step} to {save_to_path}")
                model.save_checkpoint(save_to_path)
                if accelerator.is_main_process:
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
                    
            global_step += 1    

        if config.optimizer.scheduler in ["step"]:
            lr_scheduler.step()


    writer.flush()
    writer.close()

    accelerator.print("Training done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)
