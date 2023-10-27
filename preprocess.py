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
    for file in tqdm(all_files):
        data = scipy.io.loadmat(file)
        x_i = data[feature][0]
        y_i = data["sentenceText"] 
        x.append(x_i)
        y.append(y_i)
    x = np.concatenate(x).tolist()
    y = np.concatenate(y)

    return {
        "features":  x,   
        "sentences": y,
    }
    

# Load the dataset dict for training/evaluation
def load_dataset_dict(data_dir, feature="tx1", split="train"):

    if split == "train":

        print("Loading train data...")
        train_dir = os.path.join(data_dir, "competitionData/train/")
        train_dict = get_split_dict(train_dir, feature)

        print("Loading test data...")
        test_dir = os.path.join(data_dir, "competitionData/test/")
        test_dict = get_split_dict(test_dir, feature)

        return {
                "train": train_dict,
                "test":  test_dict,
                }

    elif split == "test": 
        print("Loading test data...")
        test_dir = os.path.join(data_dir, "competitionData/test/")
        test_dict = get_split_dict(test_dir, feature)
        return {
                "test": test_dict,
                }

    elif split == "heldout": 
        print("Loading heldout data...")
        heldout_dir = os.path.join(data_dir, "competitionData/competitionHoldOut/")
        heldout_dict = get_split_dict(heldout_dir, feature)
        return {
                "heldout": heldout_dict,
                }



""" Preprocess training data. Result is 
        dict {
            "input_ids": List[torch.tensor]      -  token ids for each sentence
            "attention_mask": List[torch.tensor] -  0 for masked tokens, 1 for visible tokens
            "labels": List[torch.tensor]         -  same as input_ids for the sentence, -100 for pad prompt
            "features": List[torch.tensor]       -  neural signal features
        }
"""
def preprocess_function(examples, tokenizer, prompt = ""):

        prompt_inputs = tokenizer(
            prompt.strip(), truncation=False, return_tensors="pt"
        )
        prompt_ids_len = len(prompt_inputs["input_ids"][0])

        # Remove all punctuation except apostrophes, set to lowercase, remove extra blank spaces and append prompt
        punctuation = string.punctuation.replace("'","")
        sentences = [prompt + s.translate(str.maketrans("","",punctuation)).lower().strip() for s in examples['sentences']]

        # "input_ids" and "attention_mask" for prompt+sentence
        model_inputs = tokenizer(
            sentences, truncation=False,
        )
        model_inputs = {key: [torch.tensor(row) for row in model_inputs[key]] for key in model_inputs}

        # Text to decode, predict only the sentence
        labels = deepcopy(model_inputs["input_ids"])
        for row in labels:
            row[:prompt_ids_len] = -100 # default nn.CrossEntropyLoss ignore_index
        model_inputs["labels"] = labels
        
        # Neural signal. Conver to torch tensors
        model_inputs["features"] = [torch.tensor(row, dtype=torch.int64) for  row in examples['features'] ]

        # print(model_inputs["features"][0])
        # print(model_inputs["input_ids"][0])
        # print(model_inputs["labels"][0])
        # print(model_inputs["attention_mask"][0])

        # Keep to evaluate infenrence
        eval = {}
        eval["sentences"] = sentences
        eval["prompt_inputs"] = prompt_inputs
        return {
            "model_inputs": model_inputs,
            "eval": eval
        }



from transformers import AutoTokenizer
from functools import partial

data_dir = "/n/home07/djimenezbeneto/lab/datasets/BCI"
model_path = "/n/home07/djimenezbeneto/lab/models/BCI"
proc_path = "/n/home07/djimenezbeneto/lab/datasets/BCI/processed.data"

prompt = ""
feature="tx1"
split="train"


tokenizer = AutoTokenizer.from_pretrained(model_path)

ds = load_dataset_dict(data_dir, feature=feature, split=split)
proc_ds = {
        split: preprocess_function(ds[split], tokenizer, prompt=prompt) 
    for split in ds}

torch.save(proc_ds, proc_path)