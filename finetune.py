import os
import json
from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from datasets import load_from_disk
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from peft import LoraConfig
from peft.utils.other import fsdp_auto_wrap_policy


from models.bci import BCI
from models.neural_encoder import NeuralConfig
from utils.data_utils import BCIDataset, pad_collate_fn

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False

def main():
    accelerator = Accelerator()
    savestring = "test2"
    model_name_or_path = "/n/home07/djimenezbeneto/lab/models/BCI"
    checkpoint_path = f"/n/home07/djimenezbeneto/lab/BCI/checkpoints/{savestring}"
    final_path = f"/n/home07/djimenezbeneto/lab/BCI/ft_models/{savestring}"
    
    proc_data_path = "/n/home07/djimenezbeneto/lab/datasets/BCI/processed.data"
    split = "train"
    seed = 1
    batch_size = 64
    lr = 3e-4
    num_epochs = 50
    save_epochs = [1,5,10,15,25,35]

    peft_config = LoraConfig(
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj","v_proj"]
    )
    
    neural_config = NeuralConfig()

    reset_seeds(seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='right')

    # Load model with peft adapter for decoder
    model = BCI.peft_from_pretrained(model_name_or_path, peft_config, neural_config=neural_config)   
    accelerator.print(model)
    accelerator.print(f"Encoder params: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad):,}")
    accelerator.print("Decoder: ")
    model.decoder.print_trainable_parameters()

    # Load preprocessed dataset
    data = torch.load(proc_data_path)
    train_data = data["train"]
    test_data = data["test"]

    train_dataset = BCIDataset(train_data, split=split)
    test_dataset = BCIDataset(test_data, split=split)

    # Arguments for padding function
    pad_id = tokenizer.eos_token_id
    L = max(neural_config.n_latents)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=partial(pad_collate_fn,pad_id,L,split), batch_size=batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=partial(pad_collate_fn,pad_id,L,split), batch_size=batch_size, pin_memory=True
    )


    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # for pn, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(pn)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )


    # Prepare model for distributed training
    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        print("\n\n\n\nUsing FSDP\n\n\n\n")
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    model, train_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, test_dataloader, optimizer, lr_scheduler
    )


    # Train
    all_test_epoch_loss = []
    all_train_epoch_loss = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        test_loss = 0
        test_preds = []
        for step, batch in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            test_loss += loss.detach().float()
        test_epoch_loss = test_loss / len(test_dataloader)
        test_ppl = torch.exp(test_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {test_ppl=} {test_epoch_loss=}")

        all_train_epoch_loss.append(train_epoch_loss.item())
        all_test_epoch_loss.append(test_epoch_loss.item())
        
        if epoch in save_epochs and accelerator.is_main_process:
            model.save_adapter(os.path.join(checkpoint_path,f"EP{epoch}"))

        accelerator.wait_for_everyone()
        

    if accelerator.is_main_process:
        model.merge_decoder()
        model.save_pretrained(final_path)
        json.dump(all_train_epoch_loss, open(os.path.join(final_path,"train_loss.json"),"w"))
        json.dump(all_test_epoch_loss, open(os.path.join(final_path,"test_loss.json"),"w"))

if __name__ == "__main__":
    main()