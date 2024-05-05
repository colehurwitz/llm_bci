from tqdm import tqdm
from typing import Optional, List

import torch
import torch.nn as nn
import numpy as np
from scipy.special import gammaln

from models.masker import Masker

from utils.viz_neuron_fit import viz_single_cell, viz_single_cell_unaligned
from utils.config_utils import DictConfig


def co_smoothing_eval(
        trainer,
        save_path,
        method,          # how was the model trained
        is_aligned,
        subtract_psth:      Optional[str]  = "task",
        onset_alignment:    Optional[List] = [40],
        target_regions:     Optional[List] = ["all"],
        modes:              Optional[List] = ["neuron","intra-region","inter-region"],      # how to evaluate model
        make_r2_plots:      Optional[bool] = False,
        max_N:              Optional[int] = None,
):

    
    def probe(model, model_inputs, unused_inputs, outputs, **kwargs):
        a = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in model_inputs.items()}
        b = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in unused_inputs.items()}
        c = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in outputs.items() if v is not None}
        all_batches.append({"inputs": a, "unused": b, "outputs": c})
        return torch.tensor(0.0)

    trainer.config["training"]["shuffle_test_dataloader"] = False
    trainer.metric_fns = {"probe": probe}
    uuids_list = trainer.test_dataset[0]["neuron_uuids"]
    region_list = trainer.test_dataset[0]["neuron_regions"]
    all_regions = list(set(region_list))
    neurons_by_region = {region: [i for i, row in enumerate(region_list) if row == region] for region in region_list}

    if 'all' in target_regions:
        target_regions = all_regions
    else:
        target_regions = list(set(target_regions))

    T = next(iter(trainer.test_dataloader))[0]["spikes"].size(1)
    N = max_N or next(iter(trainer.test_dataloader))[0]["spikes"].size(2)

    if is_aligned:
        # prepare the condition matrix
        b_list = []
        # choice
        choice = np.stack([row["choice"] for row in trainer.test_dataset], axis=0)
        choice = np.tile(np.reshape(choice, (choice.shape[0], 1)), (1, T))
        b_list.append(choice)
        # reward
        reward = np.stack([row["reward"] for row in trainer.test_dataset], axis=0)
        reward = np.tile(np.reshape(reward, (reward.shape[0], 1)), (1, T))
        b_list.append(reward)
        # block
        block = np.stack([row["block"] for row in trainer.test_dataset], axis=0)
        block = np.tile(np.reshape(block, (block.shape[0], 1)), (1, T))
        b_list.append(block)
        behavior_set = np.stack(b_list, axis=-1)
        var_name2idx = {
            'choice': [0],
            'reward': [1],
            'block': [2],
            'wheel': [3],
        }
        var_value2label = {
            'block': {
                (0.2,): "p(left)=0.2",
                (0.5,): "p(left)=0.5",
                (0.8,): "p(left)=0.8", 
            },
            'choice': {
                (-1.0,): "right",
                (1.0,): "left"
            },
            'reward': {
                (0.,): "no reward",
                (1.,): "reward", 
                }
        }
        var_tasklist = ['block', 'choice', 'reward']
        var_behlist = []


    def get_bps_r2(trainer, all_batches, n_i, mode):
        preds = torch.cat([b["outputs"]["preds"] for b in all_batches], dim=0).numpy()
        if trainer.model.log_input:
            preds = np.exp(preds)
        spikes = np.stack([row["spikes"] for row in trainer.test_dataset], axis=0)
        # compute co-bps
        spikes_held_out = spikes[:, :, [n_i]]
        pred_held_out = preds[:, :, [n_i]]
        bps = bits_per_spike(pred_held_out, spikes_held_out)
        if make_r2_plots:
            if is_aligned:
                r2 = viz_single_cell(
                    behavior_set, spikes[:, :, n_i], preds[:, :, n_i],
                    var_name2idx, var_tasklist, var_value2label, var_behlist,
                    subtract_psth=subtract_psth,
                    aligned_tbins=onset_alignment,
                    neuron_idx=uuids_list[n_i][:4],
                    neuron_region=region_list[n_i],
                    save_path=save_path,
                    method=method, mode=mode,
                )
            else:
                r2 = viz_single_cell_unaligned(
                    spikes[:, :, n_i], preds[:, :, n_i], 
                    neuron_idx=uuids_list[n_i][:4],
                    neuron_region=region_list[n_i],
                    save_path=save_path,
                    method=method, mode=mode,
                )
        else:
            r2 = [0.0, 0.0]
        return bps, r2

    results_dict =  {}

    if 'neuron' in modes:
        masker_config = {
            "main": {
                "force_active": True,         
                "mode": "co-smooth",
                "ratio": 1.0,           
                "zero_ratio": 1.0,    
                "random_ratio": 1.0,   
                "expand_prob": 0.0,    
                "max_timespan": 1,     
                "channels": None,
                "timesteps": None,          
                "mask_regions": None,      
                "target_regions": None,   
                "n_mask_regions": 1,
            }
        }
        trainer.model.masker = nn.ModuleDict({k: Masker(DictConfig(m_config)) for k, m_config in masker_config.items()})
        
        bps_result_list, r2_result_list = [], []    
        for n_i in tqdm(range(N)):
            all_batches = []
            trainer.model.masker.main.channels = [n_i]
            _ = trainer.evaluate(eval_train_set=False)
            bps, r2 = get_bps_r2(trainer, all_batches, n_i, "neuron")
            bps_result_list.append(bps)
            r2_result_list.append(r2)
        
        results_dict["neuron"] = {
            "bps": bps_result_list,
            "r2": r2_result_list,
        }

    if 'intra-region' in modes:
        masker_config = {
            "region": {
                "force_active": True,         
                "mode": "intra-region",
                "ratio": 0.0,           
                "zero_ratio": 1.0,    
                "random_ratio": 1.0,   
                "expand_prob": 0.0,    
                "max_timespan": 1,     
                "channels": None,
                "timesteps": None,          
                "mask_regions": None,      
                "target_regions": None,   
                "n_mask_regions": 1,
            },
            "main": {
                "force_active": True,         
                "mode": "co-smooth",
                "ratio": 1.0,           
                "zero_ratio": 1.0,    
                "random_ratio": 1.0,   
                "expand_prob": 0.0,    
                "max_timespan": 1,     
                "channels": None,
                "timesteps": None,          
                "mask_regions": None,      
                "target_regions": None,   
                "n_mask_regions": 1,
            }
        }
        trainer.model.masker = nn.ModuleDict({k: Masker(DictConfig(m_config)) for k, m_config in masker_config.items()})
        bps_result_list, r2_result_list = [], []
        for n_i in tqdm(range(N)):
            all_batches = []
            trainer.model.masker.main.channels = [n_i]
            trainer.model.masker.region.target_regions = [region_list[n_i]]
            _ = trainer.evaluate(eval_train_set=False)
            bps, r2 = get_bps_r2(trainer, all_batches, n_i, "intra-region")
            bps_result_list.append(bps)
            r2_result_list.append(r2)
        
        results_dict["intra-region"] = {
            "bps": bps_result_list,
            "r2": r2_result_list,
        }

    if 'inter-region' in modes:
        masker_config = {
            "region": {
                "force_active": True,         
                "mode": "inter-region",
                "ratio": 1.0,           
                "zero_ratio": 1.0,    
                "random_ratio": 1.0,   
                "expand_prob": 0.0,    
                "max_timespan": 1,     
                "channels": None,
                "timesteps": None,          
                "mask_regions": None,      
                "target_regions": None,   
                "n_mask_regions": 1,
            },
        }
        trainer.model.masker = nn.ModuleDict({k: Masker(DictConfig(m_config)) for k, m_config in masker_config.items()})
        bps_result_list, r2_result_list = [], []
        n_target_neurons = len([n for region in target_regions for n in neurons_by_region[region]])
        bar = iter(tqdm(range(n_target_neurons)))
        for region in target_regions:
            all_batches = []
            trainer.model.masker.region.mask_regions = [region]
            _ = trainer.evaluate(eval_train_set=False)
            for n_i in neurons_by_region[region]:
                bps, r2 = get_bps_r2(trainer, all_batches, n_i, "inter-region")
                bps_result_list.append(bps)
                r2_result_list.append(r2)
                _ = next(bar)
       
        results_dict["inter-region"] = {
            "bps": bps_result_list,
            "r2": r2_result_list,
        }

    return results_dict

def neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
            spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            logger.warning(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates == 0] = 1e-9

    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    # print('nll_score', np.sum(result))
    # print('rate', rates.reshape(-1, rates.shape[1]*rates.shape[2]), '\nspikes', spikes.reshape(-1, spikes.shape[1]*spikes.shape[2]), '\nresult', result.reshape(-1, result.shape[1]*result.shape[2]))
    # print(rates.shape, spikes.shape, result.shape)
    return np.sum(result)


def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, zero_warning=False)
    # print(np.nansum(spikes))
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2) if np.nanmean(spikes) != 0 else np.nan
