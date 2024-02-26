
from copy import deepcopy
from typing import Dict, List, Any, Optional, Union, Tuple

import torch
import scipy
import numpy as np

from torch.utils.data import Dataset


""" Base dataset for neural data. Can be used for self-supervised pretraining.
INPUTS:
    dataset: list of examples. Each example is a dict.
    length: the list is trimmed up to this value
OUTPUTS
    Dict{
        spikes: neural data of size (seq_len, num_channels)
    }
"""
class SpikingDataset(Dataset):

    def __init__(
        self, 
        dataset: Dict[str,List[Any]], 
        length: Optional[int] = None
    ):  
        self.dataset = dataset[:length]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            "spikes": self.dataset[idx]["spikes"],    # (seq_len, num_channels)
        }



""" Dataset to use for CTC training. 
INPUTS
    dataset: list of examples. Each example is a dict.
    target_name: name of the CTC labels in the dataset
    length: the list is trimmed up to this value
OUTPUTS
    Dict{
        spikes: neural data of size (seq_len, num_channels)
        targets: ctc labels of size (seq_len, vocab_size)
        targets_lengths: length of each sequence of ctc labels 
    }
"""
class SpikingDatasetForCTC(SpikingDataset):

    def __init__(
        self, 
        dataset: List[Dict[str,Union[np.ndarray,Any]]], 
        length: Optional[int] = None,
        target_name: Optional[str] = "targets",
    ):  
        super().__init__(dataset, length)

        self.target_name = target_name

    def __getitem__(self, idx):
        targets = self.dataset[idx][f"{self.target_name}_idx"]
        return {
            "spikes":   self.dataset[idx]["spikes"],    # (seq_len, num_channels)
            "spikes_lengths": np.asarray(self.dataset[idx]["spikes"].shape[0]) # (1)
            "targets":  targets,                        # (seq_len)
            "targets_lengths": np.asarray(len(targets)) # (1)
        }


            

""" Batches a ``list`` of ``np.ndarray`` that only differ in sizes in dimension ``dim``, padding with
``pad value``.
INPUTS
    arrays: ``list`` of ``np.ndarray`` to concatenate
    dim: axis along which to pad
    side: side of the axis to pad. Can be left or right
    value: value to fill the padded array
    truncate: maximum length of the axis to pad
    min_length: minimum length of the padded axis
OUTPUTS
    padded numpy array with a preprended batch dimension
"""
def padded_array(
    arrays: List[np.ndarray],
    dim: Optional[int] = 0,
    side: Optional[str] = "right",
    value: Optional[int] = 0,
    truncate: Optional[int] = None,
    min_length: Optional[int] = None,
) -> np.ndarray:

    max_size = max(arr.shape[dim] for arr in arrays)
    if truncate is None:
        truncate = max_size
    if min_length is None:
        min_length = 0
    assert min_length <= truncate, "Can't truncate below the minimum length"
    pad_size = min(truncate,max(max_size, min_length))

    # Padding widths
    pad_width = np.zeros((arrays[0].ndim,2), dtype=np.int64)
    if side == "left":
        pad_width[dim,0] = 1
    elif side == "right":
        pad_width[dim,1] = 1
    else:
        raise Exception(f' "side" can only take values "right" or "left", got {side}')

    # Slice used to truncate arbitrary axis of ``np.ndarray``
    slc = [slice(None)] * arrays[0].ndim
    slc[dim] = slice(0, truncate)

    return np.stack([np.pad(arr, pad_width*max(0, pad_size - arr.shape[dim]), mode='constant', constant_values=value)[tuple(slc)] for arr in arrays], axis=0)





""" Collate function. Numpy arrays with keys in ``pad_dict`` are padded. Padded arrays are concatenated into
a ``torch.Tensor`` with a preprended batch dimension, other arrays are concatenated into a list of ``torch.Tensor``
INPUTS
    batch: List of examples. Each example is a dict corresponding to a row in the dataset.
    model_inputs: Names of keys that are used by the model
    pad_dict: The keys in this dict are padded with value ``value`` along dimension ``dim``
OUTPUTS
    A tuple of padded batch, which contains model inputs, and unused inputs.
"""
def pad_collate_fn(
    batch: List[Dict[str,Union[np.ndarray,Any]]], 
    model_inputs: List[str],
    pad_dict: Dict[str,Dict[str,Any]],
) -> Tuple[Dict[str,Union[torch.Tensor, List[Union[torch.Tensor, Any]]]]]:
    
    keys = batch[0].keys()
    pad_keys = pad_dict.keys()
    array_keys = [k for k in keys if isinstance(batch[0][k],np.ndarray)]
    assert set(pad_keys).issubset(array_keys), f"Can't pad keys which are not arrays: {set(pad_keys)-set(array_keys)} "
    
    padded_batch = {}
    unused_inputs = {}
    for key in keys:
        if key in array_keys:
            if key in pad_keys:
                value = torch.from_numpy(padded_array([row[key] for row in batch],**pad_dict[key])).clone()
            else:
                value = [torch.from_numpy(row[key]) for row in batch]
        else:
            value = deepcopy(value)

        if key in model_inputs:
            padded_batch[key] = value
        else:
            unused_inputs[key] = value

    return padded_batch, unused_inputs