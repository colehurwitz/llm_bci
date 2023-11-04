import torch
from torch.utils.data import Dataset


class BCIDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.data["input_ids"][idx],
            "labels": self.data["labels"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "features": self.data["features"][idx],
            "block_idx": self.data["block_idx"][idx],
            "date_idx": self.data["date_idx"][idx],
        }


""" Batch data. Returns
        dict {
            "input_ids": torch.LongTensor        -  token ids for each sentence
            "attention_mask": torch.FloatTensor  -  0. for masked tokens, 1. for visible tokens
            "labels": torch.LongTensor           -  same as input_ids for the sentence, -100 for pad prompt
            "features": torch.FloatTensor        -  neural signal features
            "features_mask": torch.FloatTensor    -  0. for added time bins, 1. for real time bins
            "features_timestamp": torch.LongTensor        -  position encoding for neural data
            "block_idx":torch.LongTensor         -  index of block of trials
            "date_idx": torch.LongTensor         -  index of day of experiment
        }
"""
def pad_collate_fn(pad_id, T, batch):
    padded_batch = {}
    padded_batch["input_ids"] = []
    padded_batch["labels"] = []
    padded_batch["attention_mask"] = []
    padded_batch["features"] = []
    padded_batch["features_mask"] = []
    padded_batch["features_timestamp"] = []
    padded_batch["block_idx"] = [batch[i]["block_idx"] for i in range(len(batch))] # no need to pad
    padded_batch["date_idx"]  = [batch[i]["date_idx"]  for i in range(len(batch))] # no need to pad

    # Batch nodes and edges
    max_seq_len = max([len(batch[i]["input_ids"]) for i in range(len(batch))])
    max_fea_len = max([len(batch[i]["features"]) for i in range(len(batch))])
    

    for i in range(len(batch)):
        seq_len = len(batch[i]["input_ids"])
        pad_seq_len = max_seq_len - seq_len
        pad_seq = torch.ones(pad_seq_len, dtype=batch[i]["input_ids"].dtype)*pad_id
        mask_seq = torch.zeros(pad_seq_len)
        mask_lab = torch.ones(pad_seq_len, dtype=batch[i]["labels"].dtype) * (-100)
        fea_len = len(batch[i]["features"])
        pad_fea_len = max_fea_len - fea_len
        pad_fea = torch.zeros(pad_fea_len, len(batch[i]["features"][0]))
        mask_fea = torch.ones(max_fea_len)
        mask_fea[:pad_fea_len] = 0

        padded_batch["input_ids"].append(torch.cat((batch[i]["input_ids"], pad_seq),-1))
        padded_batch["labels"].append(torch.cat((batch[i]["labels"], mask_lab),-1))
        padded_batch["attention_mask"].append(torch.cat((batch[i]["attention_mask"], mask_seq),-1).float())
        padded_batch["features"].append(torch.cat((pad_fea, batch[i]["features"]), -2).float())
        padded_batch["features_mask"].append(mask_fea.float())
        padded_batch["features_timestamp"].append(
            torch.cat( (  torch.round(torch.arange(fea_len) * (T-1) / (fea_len-1)), pad_fea[:,0]) ).to(batch[i]["input_ids"].dtype)
        )    

    padded_batch = {key: torch.stack(padded_batch[key]) for key in padded_batch}
    # for key in padded_batch:
    #     print(key, padded_batch[key].dtype)

    return padded_batch
        
