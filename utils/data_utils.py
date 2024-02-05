import torch
import torch.nn as nn
import torch.nn.functional as F
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
            keep_idx = (self.data["date_idx"] == date_idx)[0].tolist()
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
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["sentence"])

    def __getitem__(self, idx):
        return {
            "prompt_ids": self.data["prompt_ids"].clone(),
            "text_ids": self.data["text_ids"][idx].clone(),
            "text_labels": self.data["text_labels"][idx].clone(),
            "phoneme_logits": self.data["phoneme_logits"][idx].clone(),
            "phonemes_start": self.data["phonemes_start"].clone(),
            "sentence": deepcopy(self.data["sentence"][idx]),
            "true_phonemes": deepcopy(self.data["true_phonemes"][idx]),
            "pred_phonemes": deepcopy(self.data["pred_phonemes"][idx]),
        }



""" Mask some of the phonemes
"""
def add_mask(config, split, mask, start, end):
    if config is None or split != "train":
        return mask

    for i in range(len(mask)):
        print("before",int(torch.round(config.ratio / (config.max_span//2+1) * (end[i] - start[i] + 1)).item()), mask[i,start[i].item():end[i].item()])
        indices = torch.randint(
            start[i].item(), end[i].item()+1, 
            (int(torch.round(config.ratio / (config.max_span//2+1) * (end[i] - start[i] + 1)).item()),)
        )
        
        spans = torch.randint(
            1, config.max_span+1,
            (len(indices),)
        )

        for idx, sp in zip(indices, spans):
            mask[i,idx:(min(idx + sp, end[i].item()))] = 0
        print("after", mask[i,start[i].item():end[i].item()])

    return mask



""" Add fake spikes and gaussian noise to the logits
"""
def add_noise(config, split, logits):
    if config is None or split != "train":
        return logits

    probabilities = torch.zeros(config.max_spikes+1)
    probabilities[0] = (1-config.p_spike)
    probabilities[1:] = config.p_spike / config.max_spikes
    n_fake = torch.multinomial(probabilities, logits.size(0), replacement=True)
    for i, n in enumerate(n_fake):
        indices = torch.randint(0, logits.size(1), (n,))
        values = torch.rand(n) * config.spike_scale * (logits[i].max() - logits[i].min()).item()
        logits[i, indices] += values.to(logits)

    return logits + (config.gauss_mean + torch.randn_like(logits)*config.gauss_sd)



""" Batch data. Returns
        
        "model_inputs": Dict {
            "input_ids":        torch.LongTensor        -   (batch, seq_len_text)
            "attention_mask":   torch.LongTensor        -   (batch, seq_len_text)
            "phoneme_logits":   List[torch.FloatTensor] -   (batch, seq_len_text, vocab)
            "phonemes_start":   torch.LongTensor        -   (batch)
            "phonemes_end":     torch.LongTensor        -   (batch)
            "labels":      torch.LongTensor        -   (batch, seq_len_text)
        }
        "prompt_inputs": Dict {
            "input_ids":        torch.LongTensor        -   (batch, seq_len_text)
            "attention_mask":   torch.LongTensor        -   (batch, seq_len_text)
            "phoneme_logits":   List[torch.FloatTensor] -   (batch, seq_len_text, vocab)
            "phonemes_start":   torch.LongTensor        -   (batch)
            "phonemes_end":     torch.LongTensor        -   (batch)
        }
        "sentences":            List[str]               -   (batch)
        "true_phonemes":        List[str]               -   (batch)
        "pred_phonemes":        List[str]               -   (batch)
"""  
def ft_pad_collate_fn(noise_config, mask_config, pad_id, split, batch):

    text_lens = [len(row["text_ids"]) + len(row["phoneme_logits"]) for row in batch]
    max_text_len = max(text_lens)
    prompt_lens = [len(row["prompt_ids"]) + len(row["phoneme_logits"]) for row in batch]
    max_prompt_len = max(prompt_lens)

    model_inputs = {k: [] for k in ["input_ids", "labels", "attention_mask", "phonemes_start", "phonemes_end"]}
    prompt_inputs = {k: [] for k in ["input_ids", "attention_mask", "phonemes_start", "phonemes_end"]}
    
    for row in batch:
        text_pad = max_text_len - len(row["text_ids"]) - len(row["phoneme_logits"])
        prompt_pad = max_prompt_len - len(row["prompt_ids"]) - len(row["phoneme_logits"])
        logits_pad = len(row["phoneme_logits"])

        a = row["phonemes_start"]
        model_inputs["input_ids"].append(torch.cat(
            (
                torch.ones(text_pad).long()*pad_id, 
                row["text_ids"][:a], 
                torch.ones(logits_pad).long()*pad_id, 
                row["text_ids"][a:]
            ), dim=0
        ))
        model_inputs["labels"].append(torch.cat(
            (
                torch.ones(text_pad).long()*(-100), 
                row["text_labels"][:a],
                torch.ones(logits_pad).long()*(-100), 
                row["text_labels"][a:],
            ), dim=0
        ))
        model_inputs["attention_mask"].append(torch.cat(
            (
                torch.zeros(text_pad).long(), 
                torch.ones(a).long(),
                torch.ones(logits_pad).long(),
                torch.ones(len(row["text_labels"][a:])).long(),
            ), dim=0
        ))
        model_inputs["phonemes_start"].append(
            row["phonemes_start"] + text_pad
        )
        model_inputs["phonemes_end"].append(
            row["phonemes_start"] + text_pad + logits_pad
        )

        prompt_inputs["input_ids"].append(torch.cat(
            (
                torch.ones(prompt_pad).long()*pad_id, 
                row["prompt_ids"][:a], 
                torch.ones(logits_pad).long()*pad_id, 
                row["prompt_ids"][a:]
            ), dim=0
        ))
        prompt_inputs["attention_mask"].append(torch.cat(
            (
                torch.zeros(prompt_pad).long(), 
                torch.ones(len(row["prompt_ids"][:a])).long(),
                torch.ones(logits_pad).long(),
                torch.ones(len(row["prompt_ids"][a:])).long()
            ), dim=0
        ))
        prompt_inputs["phonemes_start"].append(
            row["phonemes_start"] + prompt_pad
        )
        prompt_inputs["phonemes_end"].append(
            row["phonemes_start"] + prompt_pad + logits_pad
        )

    model_inputs = {k: torch.stack(v, dim=0) for k,v in model_inputs.items()}
    model_inputs["phoneme_logits"] = [add_noise(noise_config, split, row["phoneme_logits"]) for row in batch]
    prompt_inputs = {k: torch.stack(v, dim=0) for k,v in prompt_inputs.items()}
    prompt_inputs["phoneme_logits"] = [add_noise(noise_config, split, row["phoneme_logits"]) for row in batch]

    model_inputs["attention_mask"] = add_mask(mask_config, split, model_inputs["attention_mask"], model_inputs["phonemes_start"], model_inputs["phonemes_end"])
    prompt_inputs["attention_mask"] = add_mask(mask_config, split, prompt_inputs["attention_mask"], prompt_inputs["phonemes_start"], prompt_inputs["phonemes_end"])
    

    if split == "train":
        return model_inputs, [row["sentence"] for row in batch], [row["true_phonemes"] for row in batch], [row["pred_phonemes"] for row in batch]
    elif split == "test":
        return model_inputs, prompt_inputs, [row["sentence"] for row in batch], [row["true_phonemes"] for row in batch], [row["pred_phonemes"] for row in batch]



""" Stack consecutive logits to reduce length
"""
def stack_logits(logits, config):
    size = config.size if config is not None else 1
    stride = config.stride if config is not None else 1
    stacking = nn.Unfold(kernel_size=(size, logits.size(1)),stride=(stride,1))
    return stacking(logits.unsqueeze(0).unsqueeze(1)).transpose(1,2).squeeze(0)

""" Convert sentences to phonemes and index
"""
PHONEMES_VOCAB = [
    'AA', 'AE', 'AH', 'AO', 'AW', 
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K', 
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH', 
    'T', 'TH', 'UH', 'UW', 'V', 
    'W', 'Y', 'Z', 'ZH', 'SIL', 'BLANK'
]

def s_to_p(s, g2p):
    # keep only phonemes and add SIL at the end so that every word ends in SIL
    return [re.sub(r'[0-9]','',pp) if pp != " " else "SIL" for pp in g2p(s) if re.match(r'[A-Z]+', pp) or pp == " "] + ["SIL"] 

def p_to_i(p):
    return [PHONEMES_VOCAB.index(pp) for pp in p]

""" Data should be a list of sentences and possibly a list of phoneme_logits. The parameters for adding gaussian
    noise to the logits are set in config
"""
def prepare_phonemes_data(data, tokenizer, g2p, prompt, stack_config=None):

    # If we don't have logits, we create one-hot logits
    if not "phoneme_logits" in data:
        data["pred_phonemes"] = [s_to_p(s, g2p) for s in data["sentence"]]
        data["true_phonemes"] = data["pred_phonemes"]
        data["phoneme_logits"] = [F.one_hot(p, num_classes=len(PHONEMES_VOCAB )) for p in data["pred_phonemes"]]
    # Assume that if we have logits we also have true_phonemes and pred_phonemes
    else:
        data["phoneme_logits"] = [torch.tensor(l) for l in data["phoneme_logits"]]
    
    new_logits = []
    for l in data["phoneme_logits"]:
        new_l = stack_logits(l, stack_config)
        new_l = new_l.view(len(new_l), l.size(1), -1).mean(-1)
        new_logits.append(new_l)
    data["phoneme_logits"] = new_logits
        
    # Tokenize the prompt by parts
    text_a = prompt.split("%%")[0]
    text_b = prompt.split("%%")[1]
    text_b_full = []
    for s in data["sentence"]:
        text_b_full.append(prompt.split("%%")[1] + " " + s + tokenizer.eos_token)


    text_a = tokenizer(
            tokenizer.bos_token + text_a, truncation=False, return_tensors="pt"
    )["input_ids"][0]
    text_b = tokenizer(
            text_b, truncation=False, return_tensors="pt"
    )["input_ids"][0]
    text_b_full = tokenizer(
            text_b_full, truncation=False, 
    )

    # Create prompt model inputs
    data["prompt_ids"] = torch.cat((text_a,text_b),dim=0)
    data["text_ids"] = [torch.cat((text_a,torch.tensor(b_full)),dim=0) for b_full in text_b_full["input_ids"]]
    text_labels = deepcopy(data["text_ids"])
    prompt_len = len(text_a) + len(text_b)
    for i in range(len(text_labels)):
        text_labels[i][:prompt_len] = -100
    data["text_labels"] = text_labels
    data["phonemes_start"] = torch.tensor(len(text_a))
    
    # Format sentences and phonemes
    data["pred_phonemes"] = [" ".join(pred).replace(" ","").replace("SIL"," ").strip().lower() for pred in data["pred_phonemes"]]
    data["true_phonemes"] = [" ".join(phon).replace(" ","").replace("SIL"," ").strip().lower() for phon in data["true_phonemes"]]

    return data
