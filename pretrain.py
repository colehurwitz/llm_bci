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
from models.neural_encoder_pretrainer import NeuralEncoderPretrainer

from utils.config_utils import update_config, config_from_kwargs, ParseKwargs
from utils.data_utils import BCIDataset, pad_collate_fn

DEFAULT_CONFIG_FILE = "configs/default_pretrain_config.yaml"

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False

def main(args):
    
    config = update_config(DEFAULT_CONFIG_FILE, args.config_file if args.config_file != "none" else None) 
    config = update_config(config, config_from_kwargs(args.kwargs))

    accelerator = Accelerator()
    reset_seeds(config.seed)
    

    # Create saving paths
    checkpoint_dir = os.path.join(config.checkpoint_dir,config.savestring)
    ft_dir = os.path.join(config.ft_dir,config.savestring)
    log_dir = os.path.join(config.log_dir,config.savestring)

    # Init tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # create encoder model for pretraining
    encoder = NeuralEncoder(config.neural_config)

    # create pretraining model
    model = NeuralEncoderPretrainer(encoder, config.neural_pretrainer)
    
    accelerator.print(model)
    accelerator.print(f"Encoder params: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad):,}")
    accelerator.print("Decoder: ")
    model.decoder.print_trainable_parameters()

    # Load preprocessed dataset
    data = torch.load(config.path_to_data)
    train_data = data["train"]
    test_data = data["test"]

    train_dataset = BCIDataset(train_data, split="train", len=config.trainer.train_len)
    test_dataset = BCIDataset(test_data, split="test", len=config.trainer.test_len)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=partial(pad_collate_fn,pad_id), batch_size=config.trainer.train_batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=partial(pad_collate_fn,pad_id), batch_size=config.trainer.test_batch_size, pin_memory=True
    )

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
    for epoch in range(1,config.trainer.num_epochs+1):
        model.train()

        # Train metrics
        train_loss = 0.
        train_examples = 0.

        for step, (batch, sentences) in enumerate(tqdm(train_dataloader)):
            _, _, _, features, features_mask, features_timestamp, block_idx, date_idx = batch
            log_rates, loss = model(features, features_mask, features_timestamp, block_idx, date_idx)

            # Gather train metrics
            train_loss += loss.detach().float()
            train_examples += len(log_rates)

            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Log to tensorboard
            writer.add_scalar("Loss/train_iter",loss.detach().float()/len(log_rates), 1+step+(epoch-1)*len(train_dataloader))

        # Log to tensorboard
        train_epoch_loss = train_loss / train_examples
        writer.add_scalar("Loss/train",train_epoch_loss,epoch)


        model.eval()

        # Test metrics
        test_loss = 0.
        test_examples = 0

        for step, (batch, sentences) in enumerate(tqdm(test_dataloader)):

            # Forward pass
            with torch.no_grad():
                _, _, _, features, features_mask, features_timestamp, block_idx, date_idx = batch
                log_rates, loss = model(features, features_mask, features_timestamp, block_idx, date_idx)
            
            # Gather test metrics
            test_loss += loss.detach().float()
            test_examples += len(log_rates)

            # Log to tensorboard
            writer.add_scalar("Loss/test_iter",loss.detach().float()/len(log_rates),1+step+(epoch-1)*len(test_dataloader))

        # Log to tensorboard
        test_epoch_loss = test_loss / test_examples
        writer.add_scalar("Loss/test",test_epoch_loss,epoch)

        accelerator.print(f"{epoch=}: {train_epoch_ppl=} {train_epoch_loss=} {train_epoch_wer=} {test_epoch_ppl=} {test_epoch_loss=} {test_epoch_wer=}")

        # Save checkpoints (not merged because we don't want to delete the adapter yet)
        must_save = (config.trainer.save_every is not None and epoch%config.trainer.save_every == 0) or epoch in config.trainer.save_epochs
        if must_save:
            accelerator.print(f"Saving checkpoint at epoch {epoch}")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir,f"EP{epoch}.pth"))

        accelerator.wait_for_everyone()
        
    writer.flush()
    writer.close()

    # Save finetuned model
    torch.save(model.state_dict(), os.path.join(ft_dir, "final.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)