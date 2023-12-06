import torch
from torch.utils.data import Dataset
from copy import deepcopy

""" Dataset for finetuning the BCI. If len = None then all the data is used. In "train" and "test" splits, the input_ids, labels, and
    attention_mask are for the prompt+sentence. In "info" split, these are for prompt only. 
"""
class BCIDataset(Dataset):

     
    def __init__(self, data, split="train", len = None):
        self.data = data
        self.split = split
        
        if len is not None:
            self.data["model_inputs"] = {key: self.data["model_inputs"][key][:len] for key in self.data["model_inputs"]}
            self.data["info"]["sentence"] = self.data["info"]["sentence"][:len]
            self.data["info"]["phonemes"] = self.data["info"]["phonemes"][:len]

    def __len__(self):
        return len(self.data["model_inputs"]["input_ids"])

    def __getitem__(self, idx):

        if self.split == "train" or self.split == "test":
            
            return {
                "input_ids": self.data["model_inputs"]["input_ids"][idx].clone(),
                "labels": self.data["model_inputs"]["labels"][idx].clone(),
                "attention_mask": self.data["model_inputs"]["attention_mask"][idx].clone(),
                "features": self.data["model_inputs"]["features"][idx].clone(),
                "block_idx": self.data["model_inputs"]["block_idx"][idx].clone(),
                "date_idx": self.data["model_inputs"]["date_idx"][idx].clone(),
                "sentence": deepcopy(self.data["info"]["sentence"][idx]),
            }
        elif self.split == "info":
            return {
                "input_ids": self.data["info"]["prompt_inputs"]["input_ids"].clone(),
                "labels": self.data["info"]["prompt_inputs"]["input_ids"].clone(),
                "attention_mask": self.data["info"]["prompt_inputs"]["attention_mask"].clone(),
                "features": self.data["model_inputs"]["features"][idx].clone(),
                "block_idx": self.data["model_inputs"]["block_idx"][idx].clone(),
                "date_idx": self.data["model_inputs"]["date_idx"][idx].clone(),
                "sentence": deepcopy(self.data["info"]["sentence"][idx]),
            }
        else:
            raise Exception(f"Split {self.split} not implemented")


""" Batch data. Returns
        Dict {
            "input_ids":            torch.LongTensor    -   token ids for each sentence
            "attention_mask":       torch.LongTensor   -   0. for masked tokens, 1. for visible tokens
            "labels":               torch.LongTensor    -   same as input_ids for the sentence, -100 for pad and prompt
            "features":             torch.FloatTensor   -   neural signal features
            "features_mask":        torch.LongTensor   -   0. for added time bins, 1. for real time bins
            "features_timestamp":   torch.LongTensor    -   position encoding for neural data
            "block_idx":            torch.LongTensor    -   index of block of trials
            "date_idx":             torch.LongTensor    -   index of day of experiment
        }
        List[str]                                       -   target sentences

    The first Dict can be dierctly fed to BCI. It can also be fed to NeuralEncoder after removing input_ids,
    labels, and attention_mask. The List of sentences is used for evaluation
"""  
def bci_pad_collate_fn(pad_id, batch):
    padded_batch = {}
    padded_batch["input_ids"] = []
    padded_batch["labels"] = []
    padded_batch["attention_mask"] = []
    padded_batch["features"] = []
    padded_batch["features_mask"] = []
    padded_batch["features_timestamp"] = []
    padded_batch["block_idx"] = [batch[i]["block_idx"] for i in range(len(batch))] # no need to pad
    padded_batch["date_idx"]  = [batch[i]["date_idx"]  for i in range(len(batch))] # no need to pad

    # Batch nodes and features
    max_seq_len = max([len(batch[i]["input_ids"]) for i in range(len(batch))])
    max_fea_len = max([len(batch[i]["features"]) for i in range(len(batch))])
    

    for i in range(len(batch)):
        seq_len = len(batch[i]["input_ids"])
        pad_seq_len = max_seq_len - seq_len   
        pad_seq = torch.ones(pad_seq_len, dtype=torch.int64)*pad_id
        mask_seq = torch.zeros(pad_seq_len, dtype=torch.int64)
        pad_lab = torch.ones(pad_seq_len, dtype=torch.int64) * (-100)
        fea_len = len(batch[i]["features"])
        pad_fea_len = max_fea_len - fea_len
        pad_fea = torch.zeros(pad_fea_len, len(batch[i]["features"][0]), dtype=torch.float)
        mask_fea = torch.ones(max_fea_len, dtype=torch.int64)
        mask_fea[fea_len:] = 0

        padded_batch["input_ids"].append(torch.cat((batch[i]["input_ids"], pad_seq),-1))
        padded_batch["labels"].append(torch.cat((batch[i]["labels"], pad_lab),-1))
        padded_batch["attention_mask"].append(torch.cat((batch[i]["attention_mask"], mask_seq),-1))
        padded_batch["features"].append(torch.cat((batch[i]["features"], pad_fea), -2))
        padded_batch["features_mask"].append(mask_fea)
        padded_batch["features_timestamp"].append(
            torch.cat( (torch.arange(fea_len), pad_fea[:,0]) ).to(torch.int64)
        )

    padded_batch = {key: torch.stack(padded_batch[key]) for key in padded_batch}
    
    # To check that the dtypes are ok
    # for key in padded_batch:
    #     print(key, padded_batch[key].dtype)


    return padded_batch, [batch[i]["sentence"] for i in range(len(batch))]
        


""" Dataset for pretraining the Neural Encoder. If len = None then all the data is used. For "ctc" loss,
targets are the phonemes/subwords to align. For "poisson" loss, targets are the spiking rates.
"""
class NeuralPretrainerDataset(Dataset):

     
    def __init__(self, data, date_idx=None, loss_fn="ctc", len = None):
        self.data = data
        self.loss_fn = loss_fn
        
        self.has_sentence = "info" in self.data and "sentence" in self.data["info"]
        self.has_phonogram = "info" in self.data and "phonogram" in self.data["info"]
        if len is not None:
            self.data["model_inputs"] = {key: self.data["model_inputs"][key][:len] for key in self.data["model_inputs"]}
            if self.has_sentence:
                self.data["info"]["sentence"] = self.data["info"]["sentence"][:len] 
            if self.has_phonogram:
                self.data["info"]["phonogram"] = self.data["info"]["phonogram"][:len]
        
        if date_idx is not None:
            keep_idx = (self.data["date_idx"] == )[0].tolist()
            self.data["model_inputs"] = {key: self.data["model_inputs"][key][keep_idx] for key in self.data["model_inputs"]}
            if self.has_sentence:
                self.data["info"]["sentence"] = self.data["info"]["sentence"][keep_idx] 
            if self.has_phonogram:
                self.data["info"]["phonogram"] = self.data["info"]["phonogram"][keep_idx]

    def __len__(self):
        return len(self.data["model_inputs"]["features"])

    def __getitem__(self, idx):

        if self.loss_fn == "ctc":
            
            return {
                "features": self.data["model_inputs"]["features"][idx].clone(),
                "targets": self.data["model_inputs"]["phonemes_idx"][idx].clone(),
                "block_idx": self.data["model_inputs"]["block_idx"][idx].clone() if "block_idx" in self.data["model_inputs"] else torch.tensor(0, dtype=torch.int64),
                "date_idx": self.data["model_inputs"]["date_idx"][idx].clone() if "date_idx" in self.data["model_inputs"] else torch.tensor(0, dtype=torch.int64),
                "sentence": deepcopy(self.data["info"]["sentence"][idx]) if self.has_sentence else [],
                "phonogram": deepcopy(self.data["info"]["phonogram"][idx]) if self.has_phonogram else [],
            }
        elif self.loss_fn == "poisson":
            return {
                "features": self.data["model_inputs"]["features"][idx].clone(),
                "targets": self.data["model_inputs"]["features"][idx].clone(),
                "block_idx": self.data["model_inputs"]["block_idx"][idx].clone() if "block_idx" in self.data["model_inputs"] else torch.tensor(0, dtype=torch.int64),
                "date_idx": self.data["model_inputs"]["date_idx"][idx].clone() if "date_idx" in self.data["model_inputs"] else torch.tensor(0, dtype=torch.int64),
                "sentence": deepcopy(self.data["info"]["sentence"][idx]) if self.has_sentence else [],
                "phonogram": deepcopy(self.data["info"]["phonogram"][idx]) if self.has_phonogram else [],
            }
        else:
            raise Exception(f"Loss function {self.loss_fn} not implemented")


""" Batch data. Returns
    Tuple(
        Dict {
            "features":             torch.FloatTensor   -   neural signal features
            "features_mask":        torch.LongTensor    -   0. for added time bins, 1. for real time bins
            "features_timestamp":   torch.LongTensor    -   position encoding for neural data
            "features_len":         torch.LongTensor    -   len of unpadded feature tensor
            "targets":              torch.Long/FloatTensor    -   phoneme/subword index or spiking rates
            "targets_len":          torch.LongTensor    -   len of unpadded target tensor
            "block_idx":            torch.LongTensor    -   index of block of trials
            "date_idx":             torch.LongTensor    -   index of day of experiment
        }
        List[str]                                       -   target sentences
        List[str]                                       -   target phonograms
    )

    The first Dict can be dierctly fed to NDT Pretrainer. The Lists of sentences and phonograms is used for evaluation
"""  
def pt_pad_collate_fn(blank_id, batch):
    padded_batch = {}
    padded_batch["features"] = []
    padded_batch["features_mask"] = []
    padded_batch["features_timestamp"] = []
    padded_batch["targets"] = []
    padded_batch["features_len"] = []
    padded_batch["targets_len"] = []
    padded_batch["block_idx"] = [batch[i]["block_idx"] for i in range(len(batch))] # no need to pad
    padded_batch["date_idx"]  = [batch[i]["date_idx"]  for i in range(len(batch))] # no need to pad

    # Batch nodes and features
    max_fea_len = max([len(batch[i]["features"]) for i in range(len(batch))])
    max_tar_len = max([len(batch[i]["targets"]) for i in range(len(batch))])
    

    for i in range(len(batch)):

        fea_len = len(batch[i]["features"])
        pad_fea_len = max_fea_len - fea_len
        pad_fea = torch.zeros(pad_fea_len, len(batch[i]["features"][0]), dtype=torch.float)
        mask_fea = torch.ones(max_fea_len, dtype=torch.int64)
        mask_fea[fea_len:] = 0
        tar_len = len(batch[i]["targets"])
        pad_tar_len = max_tar_len - tar_len
        if batch[i]["targets"].dim() == 2:
            pad_tar = torch.zeros(pad_tar_len, len(batch[i]["targets"][0]), dtype=torch.float)
            cat_idx = -2
        else:
            pad_tar = torch.ones(pad_tar_len, dtype=torch.int64)*blank_id
            cat_idx = -1

        padded_batch["features"].append(torch.cat((batch[i]["features"], pad_fea), -2))
        padded_batch["features_mask"].append(mask_fea)
        padded_batch["features_timestamp"].append(
            torch.cat( (torch.arange(fea_len), pad_fea[:,0]) ).to(torch.int64)
        )    
        padded_batch["features_len"].append(torch.tensor(fea_len, dtype=torch.int64))

        padded_batch["targets"].append(torch.cat((batch[i]["targets"], pad_tar), cat_idx))
        padded_batch["targets_len"].append(torch.tensor(tar_len,dtype=torch.int64))

    padded_batch = {key: torch.stack(padded_batch[key]) for key in padded_batch}
    
    # print("features", padded_batch["features"][:4], padded_batch["features"].shape, padded_batch["features"].dtype)
    # print("features_mask", padded_batch["features_mask"][:4], padded_batch["features_mask"].shape, padded_batch["features_mask"].dtype)
    # print("features_timestamp", padded_batch["features_timestamp"][:4], padded_batch["features_timestamp"].shape, padded_batch["features_timestamp"].dtype)
    # print("targets", padded_batch["targets"][:4], padded_batch["targets"].shape, padded_batch["targets"].dtype)
    # print("features_len", padded_batch["features_len"][:4], padded_batch["features_len"].shape, padded_batch["features_len"].dtype)
    # print("targets_len", padded_batch["targets_len"][:4], padded_batch["targets_len"].shape, padded_batch["targets_len"].dtype)
    # print("block_idx", padded_batch["block_idx"][:4], padded_batch["block_idx"].shape, padded_batch["block_idx"].dtype)
    # print("date_idx", padded_batch["date_idx"][:4], padded_batch["date_idx"].shape, padded_batch["date_idx"].dtype)

    return padded_batch, [batch[i]["phonogram"] for i in range(len(batch))], [batch[i]["sentence"] for i in range(len(batch))]
        

""" Dataset for finetuning the LLM to decode words from phonemes. If len = None then all the data is used. 
"""
class PhonemesFinetuneDataset(Dataset):

     
    def __init__(self, data, len = None):
        self.data = data
        
        if len is not None:
            self.data["model_inputs"] = {key: self.data["model_inputs"][key][:len] for key in self.data["model_inputs"]}
            self.data["info"]["sentence"] = self.data["info"]["sentence"][:len]
            self.data["info"]["phonogram"] = self.data["info"]["phonogram"][:len]

    def __len__(self):
        return len(self.data["model_inputs"]["input_ids"])

    def __getitem__(self, idx):

        return {
            "input_ids": self.data["model_inputs"]["input_ids"][idx].clone(),
            "labels": self.data["model_inputs"]["labels"][idx].clone(),
            "attention_mask": self.data["model_inputs"]["attention_mask"][idx].clone(),
            "phonogram": deepcopy(self.data["info"]["phonogram"][idx]),
            "sentence": deepcopy(self.data["info"]["sentence"][idx]),
        }


""" Batch data. Returns
        Dict {
            "input_ids":            torch.LongTensor    -   token ids for phonemes + sentence
            "attention_mask":       torch.LongTensor    -   0. for masked tokens, 1. for visible tokens
            "labels":               torch.LongTensor    -   same as input_ids for the sentence, -100 for phonemes
        }
        List[str]                                       -   target phonograms
        List[str]                                       -   target sentences

    The first Dict can be dierctly fed to LLM. The Lists of phonograms and sentences are used for evaluation
"""  
def ft_pad_collate_fn(pad_id, batch):
    padded_batch = {}
    padded_batch["input_ids"] = []
    padded_batch["labels"] = []
    padded_batch["attention_mask"] = []

    # Batch nodes and features
    max_seq_len = max([len(batch[i]["input_ids"]) for i in range(len(batch))])

    for i in range(len(batch)):
        seq_len = len(batch[i]["input_ids"])
        pad_seq_len = max_seq_len - seq_len   
        pad_seq = torch.ones(pad_seq_len, dtype=torch.int64)*pad_id
        mask_seq = torch.zeros(pad_seq_len, dtype=torch.int64)
        pad_lab = torch.ones(pad_seq_len, dtype=torch.int64) * (-100)

        padded_batch["input_ids"].append(torch.cat((pad_seq,batch[i]["input_ids"]),-1))
        padded_batch["labels"].append(torch.cat((pad_lab, batch[i]["labels"]),-1))
        padded_batch["attention_mask"].append(torch.cat((mask_seq, batch[i]["attention_mask"]),-1))
       

    padded_batch = {key: torch.stack(padded_batch[key]) for key in padded_batch}
    
    # To check that the dtypes are ok
    for key in padded_batch:
        print(key, padded_batch[key].dtype)


    return padded_batch, [batch[i]["phonogram"] for i in range(len(batch))], [batch[i]["sentence"] for i in range(len(batch))]