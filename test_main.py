import yaml
import os
import sys
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from importlib import reload as rl
from utils.config_utils import config_from_kwargs, update_config, DictConfig

import utils
import utils.eval_bci
from utils.eval_bci import format_ctc, word_error_count

import viz_neuron_fit
from viz_neuron_fit import viz_single_cell

import transformers
import transformers.models
from transformers.models import *

import data_utils
import data_utils.datasets
from data_utils.speechbci_dataset import load_competition_data, create_phonemes_ctc_labels
from data_utils.ibl_dataset import load_ibl_dataset

import models
import models.trainer
import models.patchtst
import models.ndt1
import models.itransformer
from models.trainer import Trainer, default_trainer_config


rl(utils)
rl(utils.eval_bci)
from utils.eval_bci import *

rl(data_utils)
rl(data_utils.datasets)
rl(data_utils.speechbci_dataset)
rl(data_utils.ibl_dataset)
from data_utils.speechbci_dataset import *
from data_utils.ibl_dataset import *
from data_utils.datasets import *

rl(models)
rl(models.trainer)
rl(models.patchtst)
rl(models.itransformer)
rl(models.ndt1)
rl(models.masker)
from models.patchtst import *
from models.ndt1 import *
from models.itransformer import *
from models.masker import *
from models.trainer import Trainer



def wer(model, model_inputs, unused_inputs, outputs, **kwargs):
    preds = outputs["preds"].argmax(-1)
    preds = [" ".join(format_ctc(pred, vocab, blank_id)) for pred in preds]
    phonemes = [" ".join(p) for p in unused_inputs["phonemes"]]
    errors, n_phonemes = word_error_count(preds, phonemes)
    for i in range(kwargs["n_print"]):
        print(preds[i].replace(" ","").replace("SIL"," SIL "), "\n#####\n ", 
            phonemes[i].replace(" ","").replace("SIL"," SIL "),"\n#####\n ", 
            unused_inputs["sentence"][i], "\n#####\n\n ")
    return torch.tensor(errors/n_phonemes)

def probe(model, model_inputs, unused_inputs, outputs, **kwargs):
    a = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in model_inputs.items()}
    b = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in unused_inputs.items()}
    c = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in outputs.items() if v is not None}
    all_batches.append({"inputs": a, "unused": b, "outputs": c})
    return torch.tensor(0.0)

def accuracy(model, model_inputs, unused_inputs, outputs, **kwargs):
    preds = outputs["preds"].argmax(-1)
    targets = model_inputs["targets"].squeeze(1)
    acc = (preds == targets).sum() / preds.size(0)
    return acc

kwargs = {
    "savestring": "itransformer-671c7ea7-wheel-test",
    "training.num_epochs": "100", "training.train_batch_size": "16", "training.test_batch_size": "16",
    "optimizer.gradient_accumulation_steps": "1",
    "training.eval_every": "100", "training.save_every": "100", 
    "data.train_len": "null", "data.test_len": "null",
    "model": "include:configs/itransformer.yaml",
    "data.data_load": "ibl",
    "data.eid": "671c7ea7-6726-4fbe-adeb-f89c2c8e489b_aligned",
    "data.test_size": "null",
}
config_file = "configs/trainer_wheel_itransformer.yaml"
config = update_config(default_trainer_config(), config_file)
config = update_config(config, config_from_kwargs(kwargs))   

# Load dataset
if config.data.data_load == "file":
    dataset = torch.load(os.path.join(config.data.data_dir, config.data.data_file))
elif config.data.data_load == "ibl":
    dataset = load_ibl_dataset(**config.data)
elif config.data.data_load == "speechbci":
    dataset = load_competition_data(**config.data)
    if "vocab_file" in config["data"] and config.data.vocab_file is not None:
        blank_id = config.method.model_kwargs.blank_id
        vocab = json.load(open(config.data.vocab_file,"r"))
        dataset = create_phonemes_ctc_labels(dataset, config.data.vocab_file)


# Adjust lablels for static behaviour decoding
if config.method.model_kwargs.method_name == "stat_behaviour" and config.method.model_kwargs.loss == "xent":
    beh = config.method.dataset_kwargs.targets_name
    all_labels = set([int(row[beh][0]) for rows in dataset.values() for row in rows])
    l_to_i = {l: i for i, l in enumerate(all_labels)}
    i_to_l = {v: k for k, v in l_to_i.items()}
    for split in dataset.keys():
        for i in range(len(dataset[split])):
            dataset[split][i][beh] = np.atleast_1d([l_to_i[int(dataset[split][i][beh][0])]])
    config["method"]["model_kwargs"]["n_labels"] = len(all_labels)



# Get regions for region embeddings
if config.model.model_class == "iTransformer" and config.model.encoder.embed_region:
    all_regions = list(set(str(r)  for rows in dataset.values() for row in rows for r in row["neuron_regions"]))
    config["model"]["encoder"]["regions"] = all_regions
    for key in config["model"]["masker"].keys():
        config["model"]["masker"][key]["target_regions"] = all_regions
        config["model"]["masker"][key]["mask_regions"] = all_regions


# Adjust models based on dataset
spikes_name = "spikes" if "spikes" in dataset["train"][0] else config.method.dataset_kwargs.spikes_name
if config.model.model_class in ["iTransformer","PatchTST"]:      
    # We need uniform lenght of the padded batches for PatchTST and iTransformer
    if config.model.model_class == "PatchTST":
        config["model"]["encoder"]["num_input_channels"] = dataset["train"][0][spikes_name].shape[1]
        p = config.model.encoder.patch_length
        context = ((max(row[spikes_name].shape[0] for split in dataset.keys() for row in dataset[split]) + p-1) // p) * p
        config["model"]["encoder"]["context_length"] = context
    else:
        context = max(row[spikes_name].shape[0] for split in dataset.keys() for row in dataset[split])
        config["model"]["encoder"]["embedder"]["max_n_bins"] = context
    pad_update = DictConfig( {"method": {"dataloader_kwargs": {"pad_dict":
        {
            "spikes": 
                {
                    "dim": 0,
                    "side": "left",
                    "value": 0,
                    "truncate": context,
                    "min_length": context,
                },   
            "spikes_mask": {
                "dim": 0,
                "side": "left",
                "value": 0,
                "truncate": context,
                "min_length": context,
                },
            "spikes_timestamp": {
                "dim": 0,
                "side": "left",
                "value": 0,
                "truncate": context,
                "min_length": context,
            }
        }
    }}})
    config = update_config(config, pad_update)
elif config.model.model_class == "NDT1":
    config["model"]["encoder"]["embedder"]["n_channels"] = dataset["train"][0]["spikes"].shape[1]



from_pt = "/home/gridsan/dbeneto/TFG/BCI/pt_checkpoints/itransformer/ssl/itransformer-671c7ea7-ssl-mask_neuron_1_0.2-opt_1000_16_1.e-4_0.01-arch_5_768-d_0.4_0.2-cls_true-embed_mlp_true_false_1500/STEP24000"
config = DictConfig(torch.load(os.path.join(from_pt, "trainer_config.pth")))
config["savestring"] = "test_eval"
config["model"]["encoder"]["from_pt"] = from_pt
config["model"]["decoder"]["from_pt"] = from_pt
config["data"]["static_behaviours"] = ["choice","reward","block"]
config["data"]["dynamic_behaviours"] = ["wheel-speed"]
config["data"]["test_name"] = "test"


trainer = Trainer(config, dataset=dataset, metric_fns={"A": probe}) #, "accuracy": accuracy})#, "WER": wer})
all_batches = []
trainer.train()
all_batches = []
trainer.evaluate(eval_train_set=False)


###########  BEHAVIOUR DECODING EVAL ################
import os
import numpy as np
import torch
from models.trainer import Trainer
from data_utils.ibl_dataset import load_ibl_dataset
from utils.config_utils import  DictConfig
from utils.eval_behaviour_decoding import behaviour_decoding_eval

to_eval = [
    # {
    #     "from_pt": "/home/gridsan/dbeneto/TFG/BCI/pt_checkpoints/itransformer/choice/itransformer-671c7ea7-choice-mask_inter-region_1_1.0-opt_1000_16_1.e-4_0.01-arch_5_768-d_0.4_0.2-cls_true-embed_mlp_true_false_1500/STEP5000",
    #     "method": "inter-region",
    # },
    # {
    #     "from_pt": "/home/gridsan/dbeneto/TFG/BCI/pt_checkpoints/itransformer/choice/itransformer-671c7ea7-choice-mask_intra-region_1_0.2-opt_1000_16_1.e-4_0.01-arch_5_768-d_0.4_0.2-cls_true-embed_mlp_true_false_1500/STEP24000",
    #     "method": "intra-region",
    # },
    # {
    #     "from_pt": "/home/gridsan/dbeneto/TFG/BCI/pt_checkpoints/itransformer/choice/itransformer-671c7ea7-choice-mask_neuron_1_0.2-opt_1000_16_1.e-4_0.01-arch_5_768-d_0.4_0.2-cls_true-embed_mlp_true_false_1500/STEP5000",
    #     "method": "neuron",
    # },
    # {
    #     "from_pt": "/home/gridsan/dbeneto/TFG/BCI/pt_checkpoints/itransformer/wheel/itransformer-671c7ea7-wheel-mask_inter-region_1_1.0-opt_1000_16_1.e-4_0.01-arch_5_768-d_0.4_0.2-cls_true-embed_mlp_true_false_1500/STEP24000",
    #     "method": "inter-region",
    # },
    {
        "from_pt": "/home/gridsan/dbeneto/TFG/BCI/pt_checkpoints/itransformer/wheel/itransformer-671c7ea7-wheel-mask_intra-region_1_0.2-opt_1000_16_1.e-4_0.01-arch_5_768-d_0.4_0.2-cls_true-embed_mlp_true_false_1500/STEP24000",
        "method": "intra-region",
    },
    {   
        "from_pt": "/home/gridsan/dbeneto/TFG/BCI/pt_checkpoints/itransformer/wheel/itransformer-671c7ea7-wheel-mask_neuron_1_0.2-opt_1000_16_1.e-4_0.01-arch_5_768-d_0.4_0.2-cls_true-embed_mlp_true_false_1500/STEP24000",
        "method": "neuron",
    },
]

for row in to_eval:
    from_pt = row["from_pt"]
    config = DictConfig(torch.load(os.path.join(from_pt, "trainer_config.pth")))
    config["savestring"] = "test_eval"
    config["model"]["encoder"]["from_pt"] = from_pt
    config["model"]["decoder"]["from_pt"] = from_pt
    config["data"]["static_behaviours"] = ["choice","reward","block"]
    config["data"]["dynamic_behaviours"] = ["wheel-speed"]
    config["data"]["test_name"] = "test"
    is_cls = config.method.model_kwargs.method_name == "stat_behaviour" and config.method.model_kwargs.loss == "xent"
    dataset = load_ibl_dataset(**config.data)
    if is_cls:
        beh = config.method.dataset_kwargs.targets_name
        all_labels = set([int(row[beh][0]) for rows in dataset.values() for row in rows])
        l_to_i = {l: i for i, l in enumerate(all_labels)}
        i_to_l = {v: k for k, v in l_to_i.items()}
        for split in dataset.keys():
            for i in range(len(dataset[split])):
                dataset[split][i][beh] = np.atleast_1d([l_to_i[int(dataset[split][i][beh][0])]])
    config["method"]["model_kwargs"]["n_labels"] = len(all_labels)
    trainer = Trainer(config, dataset=dataset)
    method = row["method"]
    regression_metrics = ["r2","mse"]
    results = behaviour_decoding_eval(trainer, is_cls, regression_metrics)
    if not os.path.exists(f"plots/itransformer/aligned/{config.method.dataset_kwargs.targets_name}"):
        os.makedirs(f"plots/itransformer/aligned/{config.method.dataset_kwargs.targets_name}")
    torch.save(results, os.path.join(f"plots/itransformer/aligned/{config.method.dataset_kwargs.targets_name}", f"{row['method']}_metrics.pth"))





####################################################


############ SSL CO-SMOOTHING EVAL ###################
import os
import torch
from models.trainer import Trainer
from data_utils.ibl_dataset import load_ibl_dataset
from utils.config_utils import  DictConfig
from utils.eval_co_smoothing import co_smoothing_eval

to_eval = [
    {
        "from_pt": "/home/gridsan/dbeneto/TFG/BCI/pt_checkpoints/itransformer/ssl/itransformer-671c7ea7-ssl-mask_inter-region_1_1.0-opt_1000_16_1.e-4_0.01-arch_5_768-d_0.4_0.2-cls_true-embed_mlp_true_false_1500/STEP11500",
        "method": "inter-region",
    },
    {
        "from_pt": "/home/gridsan/dbeneto/TFG/BCI/pt_checkpoints/itransformer/ssl/itransformer-671c7ea7-ssl-mask_neuron_1_0.2-opt_1000_16_1.e-4_0.01-arch_5_768-d_0.4_0.2-cls_true-embed_mlp_true_false_1500/STEP24000",
        "method": "neuron",
    },
    {
        "from_pt": "/home/gridsan/dbeneto/TFG/BCI/pt_checkpoints/itransformer/ssl/itransformer-671c7ea7-ssl-mask_intra-region_1_0.2-opt_1000_16_1.e-4_0.01-arch_5_768-d_0.4_0.2-cls_true-embed_mlp_true_false_1500/STEP24000",
        "method": "intra-region",
    },
]

for row in to_eval:
    from_pt = row["from_pt"]
    config = DictConfig(torch.load(os.path.join(from_pt, "trainer_config.pth")))
    config["savestring"] = "test_eval"
    config["model"]["encoder"]["from_pt"] = from_pt
    config["model"]["decoder"]["from_pt"] = from_pt
    config["data"]["static_behaviours"] = ["choice","reward","block"]
    config["data"]["dynamic_behaviours"] = ["wheel-speed"]
    config["data"]["test_name"] = "test"
    dataset = load_ibl_dataset(**config.data)
    trainer = Trainer(config, dataset=dataset)
    eval_config = dict(
        trainer = trainer,
        save_path = "plots/itransformer/aligned/ssl/",
        method = row["method"],
        is_aligned=True,
        subtract_psth = "task",
        onset_alignment = [40],
        target_regions = ["all"],
        modes = ["neuron","intra-region","inter-region"],
        make_r2_plots = False,
    )
    results_dict = co_smoothing_eval(**eval_config)
    if not os.path.exists("plots/itransformer/aligned/ssl/"):
        os.makedirs("plots/itransformer/aligned/ssl")
    torch.save(results_dict, os.path.join("plots/itransformer/aligned/ssl/", f"{row['method']}_bps_r2.pth"))


#######################################################3



############### TABLE ###############################33
import torch
import numpy as np

ssl_neuron = torch.load("/home/gridsan/dbeneto/TFG/BCI/plots/itransformer/aligned/ssl/neuron_bps_r2.pth")
ssl_inter = torch.load("/home/gridsan/dbeneto/TFG/BCI/plots/itransformer/aligned/ssl/inter-region_bps_r2.pth")
ssl_intra = torch.load("/home/gridsan/dbeneto/TFG/BCI/plots/itransformer/aligned/ssl/intra-region_bps_r2.pth")

choice_neuron = torch.load("/home/gridsan/dbeneto/TFG/BCI/plots/itransformer/aligned/choice/neuron_metrics.pth")
choice_inter = torch.load("/home/gridsan/dbeneto/TFG/BCI/plots/itransformer/aligned/choice/inter-region_metrics.pth")
choice_intra = torch.load("/home/gridsan/dbeneto/TFG/BCI/plots/itransformer/aligned/choice/intra-region_metrics.pth")

wheel_neuron = torch.load("/home/gridsan/dbeneto/TFG/BCI/plots/itransformer/aligned/wheel-speed/neuron_metrics.pth")
wheel_inter = torch.load("/home/gridsan/dbeneto/TFG/BCI/plots/itransformer/aligned/wheel-speed/inter-region_metrics.pth")
wheel_intra = torch.load("/home/gridsan/dbeneto/TFG/BCI/plots/itransformer/aligned/wheel-speed/intra-region_metrics.pth")

neuron = [np.nanmean(np.array(ssl_neuron["neuron"]["bps"])), np.nanmean(np.array(ssl_neuron["inter-region"]["bps"])), np.nanmean(np.array(ssl_neuron["intra-region"]["bps"])), choice_neuron["acc"], wheel_neuron["mse"]]
inter = [np.nanmean(np.array(ssl_inter["neuron"]["bps"])), np.nanmean(np.array(ssl_inter["inter-region"]["bps"])), np.nanmean(np.array(ssl_inter["intra-region"]["bps"])), choice_inter["acc"], wheel_inter["mse"]]
intra = [np.nanmean(np.array(ssl_intra["neuron"]["bps"])), np.nanmean(np.array(ssl_intra["inter-region"]["bps"])), np.nanmean(np.array(ssl_intra["intra-region"]["bps"])), choice_intra["acc"], wheel_intra["mse"]]



