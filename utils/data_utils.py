import torch
from torch.utils.data import Dataset


class BCIDataset(Dataset):

    def __init__(self, data, split="train", len = None):
        self.data = data
        self.split = split
        if len is not None:
            self.data["model_inputs"] = {key: self.data["model_inputs"][key][:len] for key in self.data["model_inputs"]}
            self.data["eval"]["sentences"] = self.data["eval"]["sentences"][:len]

    def __len__(self):
        return len(self.data["model_inputs"]["input_ids"])

    def __getitem__(self, idx):
        if self.split == "train":
            return {
                "input_ids": self.data["model_inputs"]["input_ids"][idx],
                "labels": self.data["model_inputs"]["labels"][idx],
                "attention_mask": self.data["model_inputs"]["attention_mask"][idx],
                "features": self.data["model_inputs"]["features"][idx],
                "block_idx": self.data["model_inputs"]["block_idx"][idx],
                "date_idx": self.data["model_inputs"]["date_idx"][idx],
            }
        elif self.split == "eval":
            return {
                "input_ids": self.data["eval"]["prompt_inputs"]["input_ids"],
                "labels": self.data["eval"]["prompt_inputs"]["input_ids"],
                "attention_mask": self.data["eval"]["prompt_inputs"]["attention_mask"],
                "features": self.data["model_inputs"]["features"][idx],
                "block_idx": self.data["model_inputs"]["block_idx"][idx],
                "date_idx": self.data["model_inputs"]["date_idx"][idx],
                "sentence": self.data["eval"]["sentences"][idx],
            }
        else:
            raise Exception(f"Split {self.split} is not known")


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
            "sentence": Optional[List[str]]      -  target sentence
        }
"""
def pad_collate_fn(pad_id, L, split, batch):
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
            torch.cat( (  torch.round(torch.arange(fea_len) * (L-1) / (fea_len-1)), pad_fea[:,0]) ).to(batch[i]["input_ids"].dtype)
        )    

    padded_batch = {key: torch.stack(padded_batch[key]) for key in padded_batch}
    # for key in padded_batch:
    #     print(key, padded_batch[key].dtype)

    if split == "train":
        return padded_batch
    elif split == "eval":
        del padded_batch["labels"]
        return padded_batch, [batch[i]["sentence"] for i in range(len(batch))]
    else:
        raise Exception(f"Split {split} is not known")

    return padded_batch
        
