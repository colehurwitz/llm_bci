import os
import re
import argparse
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from functools import partial
import string

import scipy
import torch
import numpy as np

from g2p_en import G2p

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from utils.config_utils import update_config, config_from_kwargs, ParseKwargs

DEFAULT_CONFIG_FILE = "configs/default_preprocess_config.yaml"

PHONEMES_VOCAB = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW', 
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K', 
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH', 
    'T', 'TH', 'UH', 'UW', 'V', 
    'W', 'Y', 'Z', 'ZH', 'SIL',
]



""" Extract neural data from files into dict. Returns the spikes data, the date and the
    block of the experiment, and the target sentence.
"""
def get_split_dict(split_dir, config):
    
    # Load data
    all_files = glob(os.path.join(split_dir,"*"))
    x = []
    y = []
    b = []
    d = []

    for file in tqdm(all_files):
        data = scipy.io.loadmat(file)
        x_i = np.array([np.concatenate([data[config.feature][0,i][:,:128],data["spikePow"][0,i][:,:128]],axis=1) for i in range(len(data[config.feature][0])) ],dtype=np.ndarray)  # 128 neurons correspond to area 6v
        y_i = data["sentenceText"]
        b_i = data["blockIdx"]
        d_i = [tuple(file.split("/")[-1].split(".")[1:4])] * len(b_i)

        if config.zscore:
            # Compute mean and std for each neuron across all experiments in the same block
            blocks = set([block for [block] in b_i.tolist()])
            for block in blocks:
                idx = np.where(b_i == block)[0]
                mu = np.mean(np.concatenate(x_i[idx], axis=0), axis=0)
                sd = np.std(np.concatenate(x_i[idx], axis=0), axis=0)
                for i in idx:
                    x_i[i] = (x_i[i] - mu) / sd 

        x.append(x_i)
        y.append(y_i)
        b.append(b_i)
        d += d_i

    x = np.concatenate(x).tolist()
    y = np.concatenate(y)
    b = (np.concatenate(b).squeeze() - 1).tolist()    # translate to start at 0

    return {
        "features":  x,   
        "sentence": y,
        "block": b,
        "date": d,
    }
    

""" Load the dataset dict for training/evaluation
"""
def load_dataset_dict(config):

    dataset_dict = {}

    # Get dict for each split
    for split in config.splits:
        dir = os.path.join(config.data_dir, split)
        print(f"Loading {split} data form {dir}")
        dict = get_split_dict(dir, config)
        dataset_dict[split] = dict


    # Index the dates and the blocks
    all_blocks = set([b  for split in config.splits for b in dataset_dict[split]["block"]])
    all_dates = set([d  for split in config.splits for d in dataset_dict[split]["date"]])
    
    d_to_i = {d: i for i, d in enumerate(all_dates)} # date (tuple) to index (int)
    b_to_i = {b: i for i, b in enumerate(all_blocks)} # block (int) to index (int)
    for split in config.splits:
        dataset_dict[split]["block_idx"] = [b_to_i[b] for b in dataset_dict[split]["block"]]
        dataset_dict[split]["date_idx"]  = [d_to_i[d] for d in dataset_dict[split]["date"]]
        

    # Useful to set n_blocks and n_dates in the BCI model config
    print("Dates: ", len(all_dates))
    print("Blocks: ", len(all_blocks))

    return dataset_dict



""" Convert sentences to phonograms and index
"""
def s_to_p(s, g2p):
    # keep only phonemes and add SIL at the end so that every word ends in SIL
    return [re.sub(r'[0-9]','',pp) if pp != " " else "SIL" for pp in g2p(s) if re.match(r'[A-Z]+', pp) or pp == " "] + ["SIL"] 

def p_to_i(p):
    return [PHONEMES_VOCAB.index(pp) for pp in p]


""" Preprocess training data. Returns 
        Dict { 
            "model_inputs" {
                "input_ids":        List[torch.LongTensor]  -   token ids for each sentence
                "attention_mask":   List[torch.LongTensor]  -   0 for masked tokens, 1 for visible tokens
                "labels":           List[torch.LongTensor]  -   same as input_ids for the sentence, mask (-100) for pad and prompt
                "phonemes_idx":     List[torch.LongTensor]  -   indexing of phonemes/subwords for each sentence
                "features":         List[torch.FloatTensor] -   neural signal features
                "block_idx":        List[torch.LongTensor]  -   index of block of experiment
                "date_idx":         List[torch.LongTensor]  -   index of day of experiment
            }
            "info" {
                "sentence":         List[str]               -   target sentences
                "phonogram":        List[str]               -   sentences decomposed into phonemes/subwords
                "date":             List[Tuple]             -   day of the experiment
                "block":            List[int]               -   block of the experiment
                "prompt_inputs" {
                    "input_ids":      torch.LongTensor      -   token ids for the prompt
                    "attention_mask": torch.LongTensor      -   0 for masked tokens, 1 for visible tokens
                }
                "vocab":           List[str]                -   vocabulary used to decompose sentence into phonemes/subwords
            }
        }
"""
def preprocess_function(examples, tokenizer, g2p,  prompt = ""):

        # Tokenize prompt and bos token: to mask these tokens in input_ids and to use for evaluation
        prompt_inputs = tokenizer(
            prompt + tokenizer.bos_token, truncation=False, return_tensors="pt"
        )
        prompt_inputs = {key: prompt_inputs[key][0] for key in prompt_inputs}
        prompt_ids_len = len(prompt_inputs["input_ids"]) - 1  # -1 to not count bos_token

        # Remove all punctuation except apostrophes, set to lowercase, remove extra blank spaces, 
        punctuation = string.punctuation.replace("'","")
        sentence = [
            s.translate(str.maketrans("","",punctuation)).lower().strip()
            for s in examples['sentence']
        ]

        # Get phonemes
        phonogram = [s_to_p(s, g2p) for s in sentence]
        phonemes_idx = [p_to_i(p) for p in phonogram]

        # Append prompt and bos/eos tokens
        sentence_prompt = [
            prompt  + tokenizer.bos_token + s + tokenizer.eos_token 
            for s in sentence
        ]

        # Tokenize prompt+sentence
        model_inputs = tokenizer(
            sentence_prompt, truncation=False
        )
        model_inputs = {key: [torch.tensor(row) for row in model_inputs[key]] for key in model_inputs}

        # Text to decode, mask prompt and predict only the sentence (with bos_token)
        labels = deepcopy(model_inputs["input_ids"])
        for row in labels:
            row[:prompt_ids_len] = -100 # default nn.CrossEntropyLoss ignore_index
        model_inputs["labels"] = labels
        
        # Phonemes indexing
        model_inputs["phonemes_idx"] = [torch.tensor(row,dtype=torch.int64) for row in phonemes_idx]

        # Neural signal. Convert to torch tensors
        model_inputs["features"] = [torch.tensor(row,dtype=torch.float) for  row in examples['features'] ]

        # Block index.
        model_inputs["block_idx"] = [torch.tensor(row,dtype=torch.int64) for  row in examples['block_idx'] ]

        # Date index
        model_inputs["date_idx"] = [torch.tensor(row,dtype=torch.int64) for  row in examples['date_idx'] ]

        # Keep to evaluate word error rate
        info = {}
        info["sentence"] = sentence
        info["phonogram"] = phonogram
        info["date"] = examples["date"]
        info["block"] = examples["block"]
        info["prompt_inputs"] = prompt_inputs
        info["vocab"] = PHONEMES_VOCAB

        # To check that dtypes are adequate
        # print(model_inputs["features"][0])
        # print(model_inputs["input_ids"][0])
        # print(model_inputs["labels"][0])
        # print(model_inputs["attention_mask"][0])
        # for key in model_inputs:
        #     print(key, model_inputs[key][0].dtype)
        # for key in prompt_inputs:
        #     print(key, prompt_inputs[key].dtype)

        return {
            "model_inputs": model_inputs,
            "info": info
        }


def main(args):

    # Get config
    config = update_config(DEFAULT_CONFIG_FILE, args.config_file if args.config_file != "none" else None) 
    config = update_config(config, config_from_kwargs(args.kwargs))

    print(config)

    # Create tokenizer and grapheme to phoneme conversor
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_dir, add_bos_token=False, add_eos_token=False, padding_side="right")
    g2p = G2p() 

    # Load (and maybe normalize and/or smooth) data
    ds = load_dataset_dict(config)

    # Process data (add prompt, tokenization, phonemes)
    proc_ds = {
            split: preprocess_function(ds[split], tokenizer, g2p, prompt=config.prompt) 
        for split in ds
    }

    torch.save(proc_ds, os.path.join(config.data_dir, config.data_file))
    print(f"Saving data to {os.path.join(config.data_dir, config.data_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for preprocessing", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)