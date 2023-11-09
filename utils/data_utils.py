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
        if self.split == "train" or self.split == "test":
            return {
                "input_ids": self.data["model_inputs"]["input_ids"][idx],
                "labels": self.data["model_inputs"]["labels"][idx],
                "attention_mask": self.data["model_inputs"]["attention_mask"][idx],
                "features": self.data["model_inputs"]["features"][idx],
                "block_idx": self.data["model_inputs"]["block_idx"][idx],
                "date_idx": self.data["model_inputs"]["date_idx"][idx],
                "sentence": self.data["eval"]["sentences"][idx],
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
        List                        str                 -   target sentences
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
        pad_fea = torch.zeros(pad_fea_len, len(batch[i]["features"][0]), dtype=batch[i]["features"].dtype)
        mask_fea = torch.ones(max_fea_len)
        mask_fea[:pad_fea_len] = 0

        padded_batch["input_ids"].append(torch.cat((pad_seq, batch[i]["input_ids"]),-1))
        padded_batch["labels"].append(torch.cat((mask_lab, batch[i]["labels"]),-1))
        padded_batch["attention_mask"].append(torch.cat((mask_seq, batch[i]["attention_mask"]),-1).float())
        padded_batch["features"].append(torch.cat((pad_fea, batch[i]["features"]), -2))
        padded_batch["features_mask"].append(mask_fea.to(batch[i]["features"].dtype))
        padded_batch["features_timestamp"].append(
            torch.cat( (pad_fea[:,0], torch.arange(fea_len)) ).to(batch[i]["features"].dtype)
        )    

    padded_batch = {key: torch.stack(padded_batch[key]) for key in padded_batch}
    # for key in padded_batch:
    #     print(key, padded_batch[key].dtype)


    return padded_batch, [batch[i]["sentence"] for i in range(len(batch))]
        


# class PretrainDataset(Dataset):

#     def __init__(self, data, split="train", len = None, mask_p):
#         self.data = data
#         self.split = split
#         if len is not None:
#             self.data["model_inputs"] = {key: self.data["model_inputs"][key][:len] for key in self.data["model_inputs"]}
#             self.data["eval"]["sentences"] = self.data["eval"]["sentences"][:len]

#     def __len__(self):
#         return len(self.data["model_inputs"]["features"])

#     def __getitem__(self, idx):
#         if self.split == "train":
            

#             return {
#                 "features": self.data["model_inputs"]["features"][idx],
#                 "block_idx": self.data["model_inputs"]["block_idx"][idx],
#                 "date_idx": self.data["model_inputs"]["date_idx"][idx],
#             }
#         else:
#             raise Exception(f"Split {self.split} is not known")


# """ Batch data. Returns
#         dict {
#             "input_ids": torch.LongTensor        -  token ids for each sentence
#             "attention_mask": torch.FloatTensor  -  0. for masked tokens, 1. for visible tokens
#             "labels": torch.LongTensor           -  same as input_ids for the sentence, -100 for pad prompt
#             "features": torch.FloatTensor        -  neural signal features
#             "features_mask": torch.FloatTensor    -  0. for added time bins, 1. for real time bins
#             "features_timestamp": torch.LongTensor        -  position encoding for neural data
#             "block_idx":torch.LongTensor         -  index of block of trials
#             "date_idx": torch.LongTensor         -  index of day of experiment
#             "sentence": Optional[List[str]]      -  target sentence
#         }
# """
# def pad_mask_collate_fn(pad_id, split, mask_config, batch):
#     padded_batch = {}

#     padded_batch["features"] = []
#     padded_batch["features_mask"] = []
#     padded_batch["features_timestamp"] = []
#     padded_batch["block_idx"] = [batch[i]["block_idx"] for i in range(len(batch))] # no need to pad
#     padded_batch["date_idx"]  = [batch[i]["date_idx"]  for i in range(len(batch))] # no need to pad

#     # Batch nodes and edges
#     max_seq_len = max([len(batch[i]["input_ids"]) for i in range(len(batch))])
#     max_fea_len = max([len(batch[i]["features"]) for i in range(len(batch))])
    

#     for i in range(len(batch)):
#         seq_len = len(batch[i]["input_ids"])
#         pad_seq_len = max_seq_len - seq_len
#         pad_seq = torch.ones(pad_seq_len, dtype=batch[i]["input_ids"].dtype)*pad_id
#         mask_seq = torch.zeros(pad_seq_len)
#         mask_lab = torch.ones(pad_seq_len, dtype=batch[i]["labels"].dtype) * (-100)
#         fea_len = len(batch[i]["features"])
#         pad_fea_len = max_fea_len - fea_len
#         pad_fea = torch.zeros(pad_fea_len, len(batch[i]["features"][0]))
#         mask_fea = torch.ones(max_fea_len)
#         mask_fea[:pad_fea_len] = 0

#         padded_batch["input_ids"].append(torch.cat((batch[i]["input_ids"], pad_seq),-1))
#         padded_batch["labels"].append(torch.cat((batch[i]["labels"], mask_lab),-1))
#         padded_batch["attention_mask"].append(torch.cat((batch[i]["attention_mask"], mask_seq),-1).float())
#         padded_batch["features"].append(torch.cat((pad_fea, batch[i]["features"]), -2).float())
#         padded_batch["features_mask"].append(mask_fea.float())
#         padded_batch["features_timestamp"].append(
#             torch.cat( (pad_fea[:,0], torch.arange(fea_len)) ).to(batch[i]["input_ids"].dtype)
#         )    

#     padded_batch = {key: torch.stack(padded_batch[key]) for key in padded_batch}
#     # for key in padded_batch:
#     #     print(key, padded_batch[key].dtype)

#     if split == "train":
#         return padded_batch
#     elif split == "eval":
#         del padded_batch["labels"]
#         return padded_batch, [batch[i]["sentence"] for i in range(len(batch))]
#     else:
#         raise Exception(f"Split {split} is not known")

#     return padded_batch
        
# def mask_batch(
#         self,
#         batch,
#         mask=None,
#         max_spikes=DEFAULT_MASK_VAL - 1,
#         should_mask=True,
#         expand_prob=0.0,
#         heldout_spikes=None,
#         forward_spikes=None,
#     ):
#         r""" Given complete batch, mask random elements and return true labels separately.
#         Modifies batch OUT OF place!
#         Modeled after HuggingFace's `mask_tokens` in `run_language_modeling.py`
#         args:
#             batch: batch NxTxH
#             mask_ratio: ratio to randomly mask
#             mode: "full" or "timestep" - if "full", will randomly drop on full matrix, whereas on "timestep", will mask out random timesteps
#             mask: Optional mask to use
#             max_spikes: in case not zero masking, "mask token"
#             expand_prob: with this prob, uniformly expand. else, keep single tokens. UniLM does, with 40% expand to fixed, else keep single.
#             heldout_spikes: None
#         returns:
#             batch: list of data batches NxTxH, with some elements along H set to -1s (we allow peeking between rates)
#             labels: true data (also NxTxH)
#         """
#         batch = batch.clone() # make sure we don't corrupt the input data (which is stored in memory)

#         mode = self.cfg.MASK_MODE
#         should_expand = self.cfg.MASK_MAX_SPAN > 1 and expand_prob > 0.0 and torch.rand(1).item() < expand_prob
#         width =  torch.randint(1, self.cfg.MASK_MAX_SPAN + 1, (1, )).item() if should_expand else 1
#         mask_ratio = self.cfg.MASK_RATIO if width == 1 else self.cfg.MASK_RATIO / width

#         labels = batch.clone()
#         if mask is None:
#             if self.prob_mask is None or self.prob_mask.size() != labels.size():
#                 if mode == "full":
#                     mask_probs = torch.full(labels.shape, mask_ratio)
#                 elif mode == "timestep":
#                     single_timestep = labels[:, :, 0] # N x T
#                     mask_probs = torch.full(single_timestep.shape, mask_ratio)
#                 elif mode == "neuron":
#                     single_neuron = labels[:, 0] # N x H
#                     mask_probs = torch.full(single_neuron.shape, mask_ratio)
#                 elif mode == "timestep_only":
#                     single_timestep = labels[0, :, 0] # T
#                     mask_probs = torch.full(single_timestep.shape, mask_ratio)
#                 self.prob_mask = mask_probs.to(self.device)
#             # If we want any tokens to not get masked, do it here (but we don't currently have any)
#             mask = torch.bernoulli(self.prob_mask)

#             # N x T
#             if width > 1:
#                 mask = self.expand_mask(mask, width)

#             mask = mask.bool()
#             if mode == "timestep":
#                 mask = mask.unsqueeze(2).expand_as(labels)
#             elif mode == "neuron":
#                 mask = mask.unsqueeze(0).expand_as(labels)
#             elif mode == "timestep_only":
#                 mask = mask.unsqueeze(0).unsqueeze(2).expand_as(labels)
#                 # we want the shape of the mask to be T
#         elif mask.size() != labels.size():
#             raise Exception(f"Input mask of size {mask.size()} does not match input size {labels.size()}")

#         labels[~mask] = UNMASKED_LABEL  # No ground truth for unmasked - use this to mask loss
#         if not should_mask:
#             # Only do the generation
#             return batch, labels

#         # We use random assignment so the model learns embeddings for non-mask tokens, and must rely on context
#         # Most times, we replace tokens with MASK token
#         indices_replaced = torch.bernoulli(torch.full(labels.shape, self.cfg.MASK_TOKEN_RATIO, device=mask.device)).bool() & mask
#         if self.cfg.USE_ZERO_MASK:
#             batch[indices_replaced] = 0
#         else:
#             batch[indices_replaced] = max_spikes + 1

#         # Random % of the time, we replace masked input tokens with random value (the rest are left intact)
#         indices_random = torch.bernoulli(torch.full(labels.shape, self.cfg.MASK_RANDOM_RATIO, device=mask.device)).bool() & mask & ~indices_replaced
#         random_spikes = torch.randint(batch.max(), labels.shape, dtype=torch.long, device=batch.device)
#         batch[indices_random] = random_spikes[indices_random]

#         if heldout_spikes is not None:
#             # heldout spikes are all masked
#             batch = torch.cat([batch, torch.zeros_like(heldout_spikes, device=batch.device)], -1)
#             labels = torch.cat([labels, heldout_spikes.to(batch.device)], -1)
#         if forward_spikes is not None:
#             batch = torch.cat([batch, torch.zeros_like(forward_spikes, device=batch.device)], 1)
#             labels = torch.cat([labels, forward_spikes.to(batch.device)], 1)
#         # Leave the other 10% alone
#         return batch, labels