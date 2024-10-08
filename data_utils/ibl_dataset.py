import os

from typing import Dict, Optional, List, Any

import numpy as np
from scipy.sparse import csr_array
from datasets import load_from_disk

""" Load IBL dataset from a given eid.
INPUTS
    data_dir: directory to load the data from
    eid: EID to load
    test_size: float between 0 and 1 indicating the proportion of dataset to use for testing. ``None`` to not split
    static_behaviours: list of static behaviours to load
    dynamic_behaviours: list of dynamic behaviours to load
    norm_behaviours: bool indicating if the behaviours should be normalized by mean and std
    seed: int to randomize split partitions
OUTPUTS
    A dictionary with splits as key. Each split key points to a list of examples. Each examples
    is a subdict of the form
    Dict {
        spikes: np.ndarray[float] of shape (seq_len, n_channels) containing neural data
        regions: np.ndarray[str] of shape (n_channels) containing the name of the region of each channel
        depths: np.ndarray[float] of shape (n_channels) containing the name of the depth of each channel
        *static_behaviour_n*: np.ndarray of shape (1) containing the behaviour label
        *dynamic_behaviour_n*: np.ndarray of shape (seq_len) containing the sequence of behaviour values

    }
"""
def load_ibl_dataset(
    data_dir:           str, 
    eid:                str,
    test_size:          Optional[float] = None, 
    static_behaviours:  Optional[List[str]] = [], 
    dynamic_behaviours: Optional[List[str]] = [],
    norm_behaviours:    Optional[bool] = False, 
    seed:               Optional[int] = 1,
    **kwargs,
) -> Dict[str,List[Dict[str,Any]]]:

    "Convert sparse data from IBL to binned spikes"
    def get_binned_spikes_from_sparse(spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list):
        sparse_binned_spikes = [csr_array((spikes_sparse_data_list[i], spikes_sparse_indices_list[i], spikes_sparse_indptr_list[i]), shape=spikes_sparse_shape_list[i]) for i in range(len(spikes_sparse_data_list))]
        binned_spikes = np.array([csr_matrix.toarray() for csr_matrix in sparse_binned_spikes], dtype=np.float32)
        return binned_spikes
    
    # Get dict for each split
    raw_dataset = load_from_disk(os.path.join(data_dir, eid))
    if test_size is not None:
        raw_dataset = raw_dataset.train_test_split(test_size=test_size, seed=seed)
    dataset_dict = {}
    for split in raw_dataset.keys():
        dataset_dict[split] = {}
        dataset_dict[split]["spikes"] = get_binned_spikes_from_sparse(raw_dataset[split]["spikes_sparse_data"], raw_dataset[split]["spikes_sparse_indices"], raw_dataset[split]["spikes_sparse_indptr"], raw_dataset[split]["spikes_sparse_shape"])
        if "cluster_uuids" in raw_dataset[split].column_names:
            dataset_dict[split]["neuron_uuids"] = raw_dataset[split]["cluster_uuids"]
        if "cluster_regions" in raw_dataset[split].column_names:
            dataset_dict[split]["neuron_regions"] = raw_dataset[split]["cluster_regions"]
        if "cluster_depths" in raw_dataset[split].column_names:
            dataset_dict[split]["neuron_depths"] = np.asarray(raw_dataset[split]["cluster_depths"], dtype=np.float32)
        for beh in static_behaviours:
            dataset_dict[split][beh] = raw_dataset[split][beh]
        exclude_idx = []
        for beh in dynamic_behaviours:
            dataset_dict[split][beh] = np.asarray(raw_dataset[split][beh], dtype=np.float32)
            for i in range(len(dataset_dict[split][beh])):
                if dataset_dict[split][beh][i] is None:
                    exclude_idx.append(i)

        # Convert from Dict[List] to List[Dict]
        dataset_dict[split] = [{k: np.atleast_1d(dataset_dict[split][k][i]) for k in dataset_dict[split]} for i in range(len(dataset_dict[split]["spikes"])) if not i in set(exclude_idx) ]
    
    if norm_behaviours:
        for beh in dynamic_behaviours:
            all_trials = np.stack([row[beh] for rows in dataset_dict.values() for row in rows], axis=0)
            mean = all_trials.mean()
            std = all_trials.std()
            for split in dataset_dict:
                for i in range(len(dataset_dict[split])):
                    dataset_dict[split][i][beh] = (dataset_dict[split][i][beh] - mean) / std


    return dataset_dict
