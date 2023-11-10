import torch
from torch.utils.data import Dataset

""" If len = None then all the data is used. In "train" and "test" splits, the input_ids, labels, and
    attention_mask are for the prompt+sentence. In "eval" split, these are for prompt only. 
"""
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

        if self.split == "train" or self.split == "test":
            
            return {
                "input_ids": self.data["model_inputs"]["input_ids"][idx],
                "labels": self.data["model_inputs"]["labels"][idx],
                "attention_mask": self.data["model_inputs"]["attention_mask"][idx],
                "features": self.data["model_inputs"]["features"][idx],
                "block_idx": self.data["model_inputs"]["block_idx"][idx],
                "date_idx": self.data["model_inputs"]["date_idx"][idx],
                "sentence": self.data["eval"]["sentences"][idx],
                "block": self.data["eval"]["block"][idx], 
                "date": self.data["eval"]["date"][idx],
            }
        elif :
            return {
                "input_ids": self.data["eval"]["prompt_inputs"]["input_ids"],
                "labels": self.data["eval"]["prompt_inputs"]["input_ids"],
                "attention_mask": self.data["eval"]["prompt_inputs"]["attention_mask"],
                "features": self.data["model_inputs"]["features"][idx],
                "block_idx": self.data["model_inputs"]["block_idx"][idx],
                "date_idx": self.data["model_inputs"]["date_idx"][idx],
                "sentence": self.data["eval"]["sentences"][idx],
                "block": self.data["eval"]["block"][idx],
                "date": self.data["eval"]["date"][idx],
            }
        else:
            raise Exception(f"Split {self.split} is not known")


""" Batch data. Returns
        Dict {
            "input_ids":            torch.LongTensor    -   token ids for each sentence
            "attention_mask":       torch.FloatTensor   -   0. for masked tokens, 1. for visible tokens
            "labels":               torch.LongTensor    -   same as input_ids for the sentence, -100 for pad and prompt
            "features":             torch.FloatTensor   -   neural signal features
            "features_mask":        torch.FloatTensor   -   0. for added time bins, 1. for real time bins
            "features_timestamp":   torch.LongTensor    -   position encoding for neural data
            "block_idx":            torch.LongTensor    -   index of block of trials
            "date_idx":             torch.LongTensor    -   index of day of experiment
        }
        List[str]                                       -   target sentences

        # Dict {
        #     "sentences":            List[str]           -   target sentences
        #     "block":                List[int]           -   block of experiment
        #     "date":                 List[Tuple]         -   date of experiment
        # }
        

    The first Dict can be dierctly fed to BCI. It can also be fed to NeuralEncoder after removing input_ids,
    labels, and attention_mask. The List of sentences is used for evaluation
"""
def pad_collate_fn(pad_id, batch):
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
        pad_seq = torch.ones(pad_seq_len, dtype=batch[i]["input_ids"].dtype)*pad_id
        mask_seq = torch.zeros(pad_seq_len)
        pad_lab = torch.ones(pad_seq_len, dtype=batch[i]["labels"].dtype) * (-100)
        fea_len = len(batch[i]["features"])
        pad_fea_len = max_fea_len - fea_len
        pad_fea = torch.zeros(pad_fea_len, len(batch[i]["features"][0]), dtype=batch[i]["features"].dtype)
        mask_fea = torch.ones(max_fea_len)
        mask_fea[:pad_fea_len] = 0

        padded_batch["input_ids"].append(torch.cat((pad_seq, batch[i]["input_ids"]),-1))
        padded_batch["labels"].append(torch.cat((pad_lab, batch[i]["labels"]),-1))
        padded_batch["attention_mask"].append(torch.cat((mask_seq, batch[i]["attention_mask"]),-1).float())
        padded_batch["features"].append(torch.cat((pad_fea, batch[i]["features"]), -2))
        padded_batch["features_mask"].append(mask_fea.to(batch[i]["features"].dtype))
        padded_batch["features_timestamp"].append(
            torch.cat( (pad_fea[:,0], torch.arange(fea_len)) ).to(batch[i]["features"].dtype)
        )    

    padded_batch = {key: torch.stack(padded_batch[key]) for key in padded_batch}
    
    # To check that the dtypes are ok
    # for key in padded_batch:
    #     print(key, padded_batch[key].dtype)


    # We can add this if we need it in the future
    # eval_dict = {
    #     "sentences": [batch[i]["sentence"] for i in range(len(batch))],
    #     "block": [batch[i]["block"] for i in range(len(batch))],
    #     "date": [batch[i]["date"] for i in range(len(batch))],
    # }
    return padded_batch, [batch[i]["sentence"] for i in range(len(batch))]
        