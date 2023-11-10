import os
from glob import glob
from tqdm import tqdm
from copy import deepcopy
import string

import scipy
import torch
import numpy as np

from datasets import Dataset, DatasetDict

""" Extract neural data from files into dict. Returns the spikes data, the date and the
    block of the experiment, and the target sentence.
"""
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
        "blocks": b,
        "dates": d,
    }
    

""" Load the dataset dict for training/evaluation
"""
def load_dataset_dict(data_dir, feature="tx1", splits=["train","test","competitionHoldOut"]):

    dataset_dict = {}

    # Get dict for each split
    for split in splits:
        print(f"Loading {split} data...")
        dir = os.path.join(data_dir, split)
        dict = get_split_dict(dir, feature)
        dataset_dict[split] = dict


    # Index the dates and the blocks
    all_blocks = set([b  for split in splits for b in dict[split]["blocks"]])
    all_dates = set([d  for split in splits for d in dataset_dict[split]["dates"]])
    d_to_i = {d: i for i, d in enumerate(all_dates)} # date (tuple) to index (int)
    for split in splits:
        dataset_dict[split]["date_idx"] = [d_to_i[d] for d in dataset_dict[split]["dates"]]
        dataset_dict[split]["block_idx"] = [b for b in dataset_dict[split]["blocks"]]

    # Useful to set n_blocks and n_dates in the BCI model config
    print("Dates: ", len(all_dates))
    print("Blocks: ", len(all_blocks))

    return dataset_dict


""" Preprocess training data. Returns 
        Dict { 
            "model_inputs" {
                "input_ids":        List[torch.LongTensor]  -   token ids for each sentence
                "attention_mask":   List[torch.LongTensor]  -   0 for masked tokens, 1 for visible tokens
                "labels":           List[torch.LongTensor]  -   same as input_ids for the sentence, mask (-100) for pad and prompt
                "features":         List[torch.LongTensor]  -   neural signal features
                "block_idx":        List[torch.LongTensor]  -   index of block of experiment
                "date_idx":         List[torch.LongTensor]  -   index of day of experiment
            }
            "eval" {
                "sentences":        List[string]            -   target sentences
                "dates":            List[Tuple]             -   day of the experiment
                "blocks":           List[int]               -   block of the experiment
                "prompt_inputs" {
                    "input_ids":      torch.LongTensor      -   token ids for the prompt
                    "attention_mask": torch.LongTensor      -   0 for masked tokens, 1 for visible tokens
                }
            }
        }
"""
def preprocess_function(examples, tokenizer, prompt = ""):

        # Tokenize prompt and bos token: to mask these tokens in input_ids and to use for evaluation
        prompt_inputs = tokenizer(
            prompt + tokenizer.bos_token, truncation=False, return_tensors="pt"
        )
        prompt_inputs = {key: prompt_inputs[key][0] for key in prompt_inputs}
        prompt_ids_len = len(prompt_inputs["input_ids"]) - 1  # -1 to not count bos_token

        # Remove all punctuation except apostrophes, set to lowercase, remove extra blank spaces, append prompt
        punctuation = string.punctuation.replace("'","")
        sentences = [
            prompt  + tokenizer.bos_token +
            s.translate(str.maketrans("","",punctuation)).lower().strip() + 
            tokenizer.eos_token 
            for s in examples['sentences']
        ]

        # Tokenize prompt+sentence
        model_inputs = tokenizer(
            sentences, truncation=False
        )
        model_inputs = {key: [torch.tensor(row) for row in model_inputs[key]] for key in model_inputs}

        # Text to decode, mask prompt and predict only the sentence (with bos_token)
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

        # Keep to evaluate word error rate
        eval = {}
        eval["sentences"] = [s.translate(str.maketrans("","",punctuation)).lower().strip() for s in examples["sentences"] ]
        eval["dates"] = examples["dates"]
        eval["blocks"] = examples["blocks"]
        eval["prompt_inputs"] = prompt_inputs

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