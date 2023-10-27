from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from datasets import load_from_disk
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from peft import LoraConfig
from peft.utils.other import fsdp_auto_wrap_policy


from bci import BCI
from data_utils import BCIDataset, pad_collate_fn

def main():
    accelerator = Accelerator()
    model_name_or_path = "/n/home07/djimenezbeneto/lab/models/BCI"
    adapter_path = "/n/home07/djimenezbeneto/lab/BCI/peft/"
    merged_path = "/n/home07/djimenezbeneto/lab/BCI/merged/"
    
    proc_data_path = "/n/home07/djimenezbeneto/lab/datasets/BCI/processed.data"
    
    batch_size = 1
    lr = 3e-4
    num_epochs = 1

    peft_config = LoraConfig(
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj","v_proj","gate_proj"]
    )
    

    # Load model with peft adapter for decoder
    model = BCI.peft_from_pretrained(model_name_or_path, peft_config)    
    accelerator.print(f"Encoder params: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad):,}")
    accelerator.print("Decoder: ")
    model.decoder.print_trainable_parameters()


    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='right')
    # Llama was pretrained without a pad token, we have to manually add it for fine-tuning
    tokenizer.add_special_tokens(
        {'pad_token': '[PAD]'}
    )
    pad_id = tokenizer.pad_token_id
    model.decoder.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    # accelerator.print(model)



    # Load preprocessed dataset
    data = torch.load(proc_data_path)
    train_dataset = BCIDataset(data["train"]["model_inputs"])
    test_dataset = BCIDataset(data["test"]["model_inputs"])

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=partial(pad_collate_fn,pad_id), batch_size=batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=partial(pad_collate_fn,pad_id), batch_size=batch_size, pin_memory=True
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
            preds = accelerator.gather_for_metrics(torch.argmax(outputs.logits, -1)).detach().cpu().numpy()
            test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
        test_epoch_loss = test_loss / len(test_dataloader)
        test_ppl = torch.exp(test_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {test_ppl=} {test_epoch_loss=}")


        accelerator.print(f"{test_preds[:10]=}")
        accelerator.print(f"{test_dataset['input_ids'][:10]=}")
        accelerator.wait_for_everyone()
        

    if accelerator.is_main_process:
        model.merge_decoder()
        model.save_pretrained(merged_path)

if __name__ == "__main__":
    main()