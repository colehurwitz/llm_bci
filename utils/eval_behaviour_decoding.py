from typing import Optional, List

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

from models.masker import Masker
from utils.config_utils import DictConfig

def metrics_list(targets, preds, metrics=["r2", "mse", "mae", "acc"]):
    results = {}
    if "r2" in metrics:
        r2_list = []
        for i in range(targets.shape[0]):
            r2 = r2_score(y_true=targets[i], y_pred=preds[i])
            r2_list.append(r2)
        r2 = np.mean(r2_list)
        results["r2"] = r2
    if "mse" in metrics:
        mse = np.mean((targets - preds) ** 2)
        results["mse"] = mse
    if "mae" in metrics:
        mae = np.mean(np.abs(targets - preds))
        results["mae"] = mae
    if "acc" in metrics:
        acc = accuracy_score(targets, preds)
        results["acc"] = acc
    return results


def behaviour_decoding_eval(
        trainer,
        is_cls,
        regression_metrics: Optional[List] = ["r2"]
):

    def probe(model, model_inputs, unused_inputs, outputs, **kwargs):
        a = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in model_inputs.items()}
        b = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in unused_inputs.items()}
        c = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in outputs.items() if v is not None}
        all_batches.append({"inputs": a, "unused": b, "outputs": c})
        return torch.tensor(0.0)

    trainer.config["training"]["shuffle_test_dataloader"] = False
    trainer.metric_fns = {"probe": probe}

    masker_config = {
        "main": {
            "force_active": False,         
            "mode": "neuron",
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
        }
    }
    trainer.model.masker = nn.ModuleDict({k: Masker(DictConfig(m_config)) for k, m_config in masker_config.items()})
        

    all_batches = []
    trainer.evaluate(eval_train_set=False)
    preds = torch.cat([b["outputs"]["preds"] for b in all_batches], dim=0).numpy()
    targets = torch.cat([b["outputs"]["targets"] for b in all_batches], dim=0).numpy()

    if is_cls:
        preds = np.argmax(preds, axis=-1)
        results = metrics_list(
            targets = targets,
            preds = preds, 
            metrics=["acc"], 
        )
    else:
        results = metrics_list(
            targets = targets,
            preds   = preds,
            metrics = regression_metrics,
        )

    return results