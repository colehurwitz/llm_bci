
from copy import deepcopy
from typing import Dict, List, Any, Optional

import torch
import scipy
import numpy as np

from torch.utils.data import Dataset




""" Dataset for spiking data.
"""
class SpikingDataset(Dataset):

    def __init__(
        self, 
        dataset: Dict[List[Any]], 
        method: Optional[str] = "ssl", 
        length: Optional[int] = None
    ):  
        self.method = method
        self.dataset = self.dataset[:length]

    def __len__(self):
        return len(self.dataset)

    """ This method should be overriden by subclasses of this Dataset
    """
    def __getitem__(self, idx):
        return {
            "spikes": self.dataset[idx]["spikes"],    # (seq_len, num_channels)
        }




class SpikingDatasetForCTC(SpikingDataset):

    def __init__(
        self, 
        dataset: Dict[List[Any]], 
        method: Optional[str] = "ssl", 
        length: Optional[int] = None,
    ):  
        super(self, SpikingDataset).__init__(dataset, method, length)

        self.target_name = kwargs["target_name"]

    def __getitem__(self, idx):
        targets = self.dataset[idx][f"{self.target_name}_idx"]
        return {
            "spikes":   self.dataset[idx]["spikes"],    # (seq_len, num_channels)
            "targets":  targets,                        # (seq_len)
            "targets_lengths": np.asarray(len(targets)) # (1)
        }


            

""" Batches a list of arrays that only differ in sizes in dimension "dim", padding with
pad value.
"""
def padded_array(
    arrays: List[np.ndarray],
    dim: int,
    pad_value: Optional[int] = 0,
) -> np.ndarray:

    max_size = max(arr.shape[dim] for arr in arrays)
    return np.vstack([np.pad(arr, (0, max_size - arr.shape[dim]), mode='constant', constant_values=pad_value) for arr in arrays])




""" Collate function that converts lists of ndarrays into padded torch tensors. Returns a tuple
of dicts, padded_bacth containing model_inputs and unused_inputs containing other keys.
"""
def pad_collate_fn(
    batch: List[Dict[Union[np.ndarray,Any]]], 
    model_inputs: List[str],
    pad_dict: Dict[Tuple(int,Any)],
):
    
    keys = batch[0].keys()
    pad_keys = pad_dict.keys()
    array_keys = [k for k in keys if isinstance(batch[0][k],np.ndarray)]

    padded_batch = {}
    unused_inputs = {}
    for key in keys:
        value = [row for row in batch[k]]
        if key in pad_keys:
            value = padded_array(value, pad_dict[key]["dim"], pad_dict[key]["value"])
        if key in array_keys:
            value = torch.from_numpy(value).clone()
        else:
            value = deepcopy(value)
        if key in model_inputs:
            padded_batch[key] = value
        else:
            unused_inputs[key] = value

    return padded_batch, unused_inputs