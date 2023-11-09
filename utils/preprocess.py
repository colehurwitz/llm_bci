import os
from glob import glob
from tqdm import tqdm
from copy import deepcopy
import string

import scipy
import torch
import numpy as np

from datasets import Dataset, DatasetDict

# Extract data from files into dict
def get_split_dict(split_dir, feature="tx1"):
    
    # Load data
    all_files = glob(os.path.join(split_dir,"*"))
    x = []
    y = []
    b = []
    d = []
    for file in tqdm(all_files):
        data = scipy.io.loadmat(file)
        x_i = data[feature][0]
        y_i = data["sentenceText"]
        b_i = data["blockIdx"]
        d_i = [tuple(file.split("/")[-1].split(".")[1:4])] * len(b_i)
        x.append(x_i)
        y.append(y_i)
        b.append(b_i)
        d += d_i
    x = np.concatenate(x).tolist()
    y = np.concatenate(y)
    b = (np.concatenate(b).squeeze() - 1).tolist()    # translate to start at 0

    

    return {
        "features":  x,   
        "sentences": y,
        "block_idx": b,
        "date_idx": d,
    }
    

# Load the dataset dict for training/evaluation
def load_dataset_dict(data_dir, feature="tx1", split="train"):

    print("Loading train data...")
    train_dir = os.path.join(data_dir, "train")
    train_dict = get_split_dict(train_dir, feature)

    print("Loading test data...")
    test_dir = os.path.join(data_dir, "test")
    test_dict = get_split_dict(test_dir, feature)

    print("Loading heldout data...")
    heldout_dir = os.path.join(data_dir, "competitionHoldOut")
    heldout_dict = get_split_dict(heldout_dir, feature)


    all_dates = set(train_dict["date_idx"]+ test_dict["date_idx"] + heldout_dict["date_idx"])
    d_to_i = {d: i for i, d in enumerate(all_dates)}
    for split_dict in [train_dict, test_dict, heldout_dict]:
        split_dict["date_idx"] = [d_to_i[d] for d in split_dict["date_idx"]]

    all_blocks = set(train_dict["block_idx"]+ test_dict["block_idx"] + heldout_dict["block_idx"])
    print("Dates: ", len(d_to_i))
    print("Blocks: ", all_blocks)

    if split == "train":
        return {
                "train": train_dict,
                "test":  test_dict,
                }

    elif split == "test": 
        return {
                "test": test_dict,
                }

    elif split == "heldout": 
        return {
                "heldout": heldout_dict,
                }
    elif split == "all": 
        return {
                "train": train_dict,
                "test":  test_dict,
                "heldout": heldout_dict,
                }


""" Preprocess training data. Returns 
        Dict { 
            "model_inputs" {
                "input_ids":        List[torch.LongTensor]  -   token ids for each sentence
                "attention_mask":   List[torch.LongTensor]  -   0 for masked tokens, 1 for visible tokens
                "labels":           List[torch.LongTensor]  -   same as input_ids for the sentence, mask (-100) for pad and prompt
                "features":         List[torch.LongTensor]  -   neural signal features
                "block_idx":        List[torch.LongTensor]  -   index of block of trials
                "date_idx":         List[torch.LongTensor]  -   index of day of experiment
            }
            "eval" {
                "sentences":        List[string]            -   target sentences
                "prompt_inputs" {
                    "input_ids":    torch.LongTensor        -   token ids for the prompt
                    "attention_mask": torch.LongTensor      -   0 for masked tokens, 1 for visible tokens
                }
            }
        }
"""
def preprocess_function(examples, tokenizer, prompt = ""):

        prompt_inputs = tokenizer(
            prompt + tokenizer.bos_token, truncation=False, return_tensors="pt"
        )
        prompt_inputs = {key: prompt_inputs[key][0] for key in prompt_inputs}
        prompt_ids_len = len(prompt_inputs["input_ids"]) - 1

        # Remove all punctuation except apostrophes, set to lowercase, remove extra blank spaces and append prompt
        punctuation = string.punctuation.replace("'","")
        sentences = [
            prompt  + tokenizer.bos_token +
            s.translate(str.maketrans("","",punctuation)).lower().strip() + 
            tokenizer.eos_token 
            for s in examples['sentences']
        ]

        # "input_ids" and "attention_mask" for prompt+sentence
        model_inputs = tokenizer(
            sentences, truncation=False
        )
        model_inputs = {key: [torch.tensor(row) for row in model_inputs[key]] for key in model_inputs}

        # Text to decode, predict only the sentence
        labels = deepcopy(model_inputs["input_ids"])
        for row in labels:
            row[:prompt_ids_len] = -100 # default nn.CrossEntropyLoss ignore_index
        model_inputs["labels"] = labels
        
        # Neural signal. Convert to torch tensors
        model_inputs["features"] = [torch.tensor(row,dtype=torch.int64) for  row in examples['features'] ]

        # Block index.
        model_inputs["block_idx"] = [torch.tensor(row,dtype=torch.int64) for  row in examples['block_idx'] ]

        # Date index
        model_inputs["date_idx"] = [torch.tensor(row,dtype=torch.int64) for  row in examples['date_idx'] ]

        # Keep to evaluate infenrence
        eval = {}
        eval["sentences"] = [s.translate(str.maketrans("","",punctuation)).lower().strip() for s in examples["sentences"] ]
        eval["prompt_inputs"] = prompt_inputs

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
            "eval": eval
        }



from transformers import AutoTokenizer
from functools import partial

# data_dir = "/home/llm4bci/competitionData"
# path_to_model = "/home/llm4bci/LLM"
# path_to_data = "/home/llm4bci/competitionData/processed.data"

data_dir = "/n/home07/djimenezbeneto/lab/datasets/BCI/competitionData"
path_to_model = "/n/home07/djimenezbeneto/lab/models/BCI"
path_to_data = "/n/home07/djimenezbeneto/lab/datasets/BCI/processed.data"


tokenizer = AutoTokenizer.from_pretrained(path_to_model, add_bos_token=False, add_eos_token=False, padding_side="right")

prompt = ""
feature="tx1"
split="all"



ds = load_dataset_dict(data_dir, feature=feature, split=split)
proc_ds = {
        split: preprocess_function(ds[split], tokenizer, prompt=prompt) 
    for split in ds}

torch.save(proc_ds, path_to_data)