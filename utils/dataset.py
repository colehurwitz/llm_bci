import os
import re
import json
import string
from glob import glob
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from g2p_en import G2p

import torch
import scipy
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


""" Load competition data from ".mat" format
"""
def load_competition_data(data_dir, zscore=False, splits=["train","test","competitionHoldOut"], features=["tx1","spikePow"], area_start=0, area_end=128):

    """ Extract neural data from files into dict. Returns the spikes data, the date and the
    block of the experiment, and the target sentence.
    """
    def get_split_dict(split_dir, zscore, features, area_start, area_end):
        all_files = glob(os.path.join(split_dir,"*"))
        x = []
        y = []
        b = []
        d = []
        for file in tqdm(all_files):
            data = scipy.io.loadmat(file)
            x_i = np.array([np.concatenate([data[feature][0,i][:,area_start:area_end] for feature in features],axis=1) for i in range(len(data["sentenceText"])) ],dtype=np.ndarray)  # 128 neurons correspond to area 6v
            y_i = data["sentenceText"]
            b_i = data["blockIdx"]
            d_i = [tuple(file.split("/")[-1].split(".")[1:4])] * len(b_i)
            if zscore:
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
            "spikes":  x,   
            "sentence": y,
            "block": b,
            "date": d,
        }
    
    # Get dict for each split
    dataset_dict = {}
    for split in splits:
        dir = os.path.join(data_dir, split)
        print(f"Loading {split} data form {dir}")
        dict = get_split_dict(dir, zscore, features, area_start, area_end)
        dataset_dict[split] = dict

    # Index the dates and the blocks
    all_blocks = set([b  for split in splits for b in dataset_dict[split]["block"]])
    all_dates = set([d  for split in splits for d in dataset_dict[split]["date"]])
        
    d_to_i = {d: i for i, d in enumerate(all_dates)} # date (tuple) to index (int)
    b_to_i = {b: i for i, b in enumerate(all_blocks)} # block (int) to index (int)
    for split in splits:
        dataset_dict[split]["block_idx"] = [b_to_i[b] for b in dataset_dict[split]["block"]]
        dataset_dict[split]["date_idx"]  = [d_to_i[d] for d in dataset_dict[split]["date"]]

    return dataset_dict





""" Dataset for spiking data.
"""
class SpikingDataset(Dataset):

    def __init__(
        self, 
        dataset: Dict[List[Any]], 
        method: Optional[str] = "ssl", 
        length: Optional[int] = None,
        **kwargs,
    ):  
        self.method = method
        self.dataset = {k: v[:length] if length is not None else v for k,v dataset.items()}
        self.dataset["spikes"] = [torch.Tensor(row, dtype=torch.float) for row in self.dataset["spikes"]]
        
        if method == "sft":
            self.g2p = G2p()
            self.vocab = json.load(open(kwargs["vocab_file"],"r"))
            self.create_ctc_labels()

    def __len__(self):
        return len(self.dataset["spikes"])

    def __getitem__(self, idx):
        if self.method == "ssl":
            return {
                "spikes": torch.Tensor(self.dataset["spikes"][idx]),    # (seq_len, num_channels)
            }
        elif self.method == "sft":
            targets = self.dataset["phonemes_idx"][idx].clone()
            return {
                "spikes":   self.dataset["spikes"][idx].clone(),    # (seq_len, num_channels)
                "targets":  targets,                                # (seq_len)
                "targets_lengths": torch.Tensor(len(targets), dtype=torch.long) # (1)
            }
        else raise Exception(f"Method {self.method} not implemented yet")

    """Sentence to phonemes
    """
    def s_to_p(
            self,
            s: str,
    ) -> List[str]:

        # keep only phonemes and add SIL at the end so that every word ends in SIL
        return [re.sub(r'[0-9]','',pp) if pp != " " else "SIL" for pp in g2p(s) if re.match(r'[A-Z]+', pp) or pp == " "] + ["SIL"] 

    """ Phonemes to vocab index
    """
    def p_to_i(
        self,
        p: List[str],
    ) -> List[int]:
        return [self.vocab.index(pp) for pp in p]

    def create_ctc_labels(self):
        punctuation = string.punctuation.replace("'","")
        self.dataset["sentence"] = [
            s.translate(str.maketrans("","",punctuation)).lower().strip()
            for s in self.dataset['sentence']
        ]
        self.dataset["phonemes"] = [self.s_to_p(s) for s in self.dataset["sentence"]]
        self.dataset["phonemes_idx"] = [torch.Tensor(self.p_to_i(p), dtype=torch.long) for p in self.dataset["phonemes"]]
