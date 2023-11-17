import os
import argparse
from tqdm import tqdm
from functools import partial

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from models.bci import BCI
from utils.config_utils import update_config, config_from_kwargs, ParseKwargs
from utils.data_utils import BCIDataset, bci_pad_collate_fn
from utils.eval_utils import word_error_count

DEFAULT_CONFIG_FILE = "configs/default_evaluate_config.yaml"

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):

    config = update_config(DEFAULT_CONFIG_FILE, args.config_file if args.config_file != "none" else None)
    config = update_config(config, config_from_kwargs(args.kwargs))

    reset_seeds(config.seed)
    


    if config.checkpoint_epoch == None:
        # Load merged model
        ft_dir = os.path.join(config.ft_dir,config.savestring)
        model = BCI.from_pretrained(ft_dir)
    else:
        # Load from checkpoint
        checkpoint_dir = os.path.join(config.checkpoint_dir,config.savestring,"EP"+str(config.checkpoint_epoch))
        model = BCI.from_pretrained(config.model_dir, device_map="auto")
        model.load_adapter(checkpoint_dir)
        model.load_encoder(checkpoint_dir)

    model.to("cuda")
    model.eval()
    print(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, padding_side='right')
    pad_id = tokenizer.eos_token_id

    # Load data
    data = torch.load(os.path.join(config.data_dir, config.data_file))[config.split]
    eval_dataset = BCIDataset(data, split="eval", len=config.eval_len)
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=partial(bci_pad_collate_fn,pad_id), batch_size=config.eval_batch_size, pin_memory=True
    )

    # Compute word error rate
    all_preds = []
    all_sentences = []
    for step, (batch, sentences) in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
    
            batch = {key: batch[key].to("cuda") for key in batch}
            preds = model.generate(
                **batch, 
                max_new_tokens=config.generation.max_new_tokens,
                do_sample=config.generation.do_sample,
                pad_token_id=tokenizer.eos_token_id
            )

            preds = [tokenizer.decode(p.cpu().squeeze(), skip_special_tokens=True) for p in preds]
            all_preds += preds
            all_sentences += sentences
    
    for p, t in zip(all_preds, all_sentences):
        print("Prediction: {}\nTarget: {}".format(p, t))

    errors, words = word_error_count(all_preds,all_sentences)
    print("Average error rate: {:.4f}".format(errors/words))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for evaluation", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)