from copy import deepcopy
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup

from peft import LoraConfig, TaskType
from peft.utils.other import fsdp_auto_wrap_policy

from bci import BCI


def main():
    accelerator = Accelerator()
    model_name_or_path = "/n/home07/djimenezbeneto/lab/models/BCI"
    adapter_path = "/n/home07/djimenezbeneto/lab/BCI/peft/"
    merged_path = "/n/home07/djimenezbeneto/lab/BCI/merged/"
    batch_size = 8
    max_length = 64
    lr = 1e-3
    num_epochs = 1
    data_path = ""

    peft_config = LoraConfig(
        task_type=TaskType.ONLY_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )

    # Load model with peft adapter for decoder
    model = BCI.peft_from_pretrained(model_name_or_path, peft_config)
    accelerator.print(f"Encoder params: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad):,}")
    accelerator.print("Decoder: ")
    model.decoder.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Llama was pretrained without a pad token, we have to manually add it for fine-tuning
    tokenizer.add_special_tokens(
        {'pad_token': '[PAD]'}
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)



    # Dummy dataset for testing
    dataset = DatasetDict({"train": Dataset.from_dict({'inputs': ["i am an example"]*16}) ,
                           "validation": Dataset.from_dict({'inputs': ["i am an example"]*16}) 
                        })
    
    def preprocess_function(examples):
        inputs = examples['inputs']
        model_inputs = tokenizer(
            inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        labels = deepcopy(model_inputs["input_ids"])
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            desc="Running tokenizer on dataset",
        )


    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )


    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )


    # Prepare model for distributed training
    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )


    # accelerator.print(model)


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
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            preds = accelerator.gather_for_metrics(torch.argmax(outputs.logits, -1)).detach().cpu().numpy()
            eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        correct = 0
        total = 0
        for pred, true in zip(eval_preds, dataset["validation"]['inputs']):
            if pred.strip() == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100
        accelerator.print(f"{accuracy=}")
        accelerator.print(f"{eval_preds[:10]=}")
        accelerator.print(f"{dataset['validation']['inputs'][:10]=}")
        accelerator.wait_for_everyone()
        

    if accelerator.is_main_process:
        model.save_adapter(adapter_path)
        model.merge_decoder()
        model.save_pretrained(merged_path)

if __name__ == "__main__":
    main()