import os
import re
import json
import string
from glob import glob
from tqdm import tqdm
from typing import List, Optional, Any, Dict

import scipy
import numpy as np
from g2p_en import G2p


""" Load competition data from ".mat" format.
INPUTS
    data_dir: directory to load the data from
    zscore: wether to zscore the data by blocks
    splits: name of the split subfolders
    features: name of the features to extract
    area_start: start index to extract features from a given area
    area_end: end index to extract features from a given area
OUTPUTS
    A dictionary with splits as key. Each split key points to a list of examples. Each examples
    is a subdict of the form
    Dict {
        spikes: np.ndarray of shape (seq_len, num_channels) containing neural data
        sentence: str with the corresponding sentence
        block: number of the block of experiment that corresponds to the example
        date: date in which the data from the example was taken
        block_idx: normalized index for block
        date_idx: normalized index for date
    }
"""
def load_competition_data(
    data_dir:   str, 
    zscore:     Optional[bool]      = False, 
    splits:     Optional[List[str]] = ["train","test","competitionHoldOut"], 
    features:   Optional[List[str]] = ["tx1","spikePow"], 
    area_start: Optional[int]       = 0, 
    area_end:   Optional[int]       = 128,
) -> Dict[str,List[Dict[str,Any]]]:

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
        return [{
            "spikes":  x_i,   
            "sentence": y_i,
            "block": b_i,
            "date": d_i,
        }  for x_i, y_i, b_i, d_i in zip(x,y,b,d)]
    
    # Get dict for each split
    dataset_dict = {}
    for split in splits:
        dir = os.path.join(data_dir, split)
        print(f"Loading {split} data form {dir}")
        dict = get_split_dict(dir, zscore, features, area_start, area_end)
        dataset_dict[split] = dict

    # Index the dates and the blocks
    all_blocks = set([row["block"]  for split in splits for row in dataset_dict[split]])
    all_dates = set([row["date"]  for split in splits for row in dataset_dict[split]])
        
    d_to_i = {d: i for i, d in enumerate(all_dates)} # date (tuple) to index (int)
    b_to_i = {b: i for i, b in enumerate(all_blocks)} # block (int) to index (int)
    for split in splits:
        for i, row in enumerate(dataset_dict[split]):
            dataset_dict[split][i]["block_idx"] = np.asarray(b_to_i[row["block"]])
            dataset_dict[split][i]["date_idx"]  = np.asarray(d_to_i[row["date"]])

    return dataset_dict



""" Create fields phonemes and phonemes_idx for CTC training. This method does not 
create a copy and modifies the passed dataset.
INPUTS
    dataset: dict with split keys. Each split is a list of examples. Each example is a dict.
    vocab_file: json file that contains the vocabulary for CTC
OUTPUTS
    dataset: input dataset with added an added keys {
        phonemes: phonemes corresponding to the sentence
        phonemes_idx: phonemes indexed according to vocabulary
    }
"""
def create_phonemes_ctc_labels(
    dataset:    Dict[str,List[Dict[str,Any]]], 
    vocab_file: str,
) -> Dict[str,List[Dict[str,Any]]]:

    g2p = G2p() # graphme to phoneme processor
    vocab = json.load(open(vocab_file,"r")) # vocab to create labels

    """Sentence to phonemes
    """
    def s_to_p(s: str) -> List[str]:
        # keep only phonemes and add SIL at the end so that every word ends in SIL
        return [re.sub(r'[0-9]','',pp) if pp != " " else "SIL" for pp in g2p(s) if re.match(r'[A-Z]+', pp) or pp == " "] + ["SIL"] 

    """ Phonemes to vocab index
    """
    def p_to_i(p: List[str]) -> List[int]:
        return [vocab.index(pp) for pp in p]

    # Normalization of sentences
    punctuation = string.punctuation.replace("'","")
    for split in dataset:
        for i, row in enumerate(dataset[split]):
            phonemes = np.asarray(s_to_p(row["sentence"].translate(str.maketrans("","",punctuation)).lower().strip()))
            dataset[split][i]["phonemes"] = phonemes
            dataset[split][i]["phonemes_idx"] = np.asarray(p_to_i(phonemes))

    return dataset


