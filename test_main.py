import yaml
import os
import sys
import json
import torch
from importlib import reload as rl
from utils.config_utils import config_from_kwargs, update_config, DictConfig

import utils
import utils.eval_utils
from utils.eval_utils import format_ctc, word_error_count

import transformers
import transformers.models
from transformers.models import *

import data_utils
import data_utils.datasets
from data_utils.speechbci_dataset import load_competition_data, create_phonemes_ctc_labels

import models
import models.trainer
import models.patchtst
import models.ndt1
from models.trainer import Trainer, default_trainer_config

kwargs = {
    "training.num_epochs": "200", "training.train_batch_size": "32", "training.test_batch_size": "32",
    "optimizer.gradient_accumulation_steps": "1",
    "training.eval_every": "500", "training.save_every": "500", 
    "data.train_len": "-1", "data.test_len": "-1",
    "model": "include:configs/ndt1s.yaml",
    "method.model_kwargs.loss": "mse",
    "data.data_name": "maze",
}
from_pt = "pt_checkpoints/test/STEP5499"

config_file = "configs/trainer_autoregressive.yaml"
config = update_config(default_trainer_config(), config_file)
config = update_config(config, config_from_kwargs(kwargs))
config = DictConfig(torch.load(os.path.join(from_pt, "trainer_config.pth")))

# # Load
if config.data.data_name == "maze":
        dataset = torch.load(config.data.data_file)
elif config.data.data_name == "speechbci":
    dataset = load_competition_data(config.data.dataset_dir, **config.data)
    if "vocab_file" in config["data"] and config.data.vocab_file is not None:
        blank_id = config.method.model_kwargs.blank_id
        vocab = json.load(open(config.data.vocab_file,"r"))
        dataset = create_phonemes_ctc_labels(dataset, config.data.vocab_file)

# Adjust models based on dataset
if config.model.model_class == "PatchTST":
    # We need uniform lenght of the padded batches for PatchTST
    p = config.model.encoder.patch_length
    context = ((max(row["spikes"].shape[0] for split in dataset.values() for row in split) + p-1) // p) * p
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
    config["model"]["encoder"]["context_length"] = context
    config["model"]["encoder"]["num_input_channels"] = dataset["train"][0]["spikes"].shape[1]
elif config.model.model_class == "NDT1":
    config["model"]["encoder"]["embedder"]["n_channels"] = dataset["train"][0]["spikes"].shape[1]



# print(yaml.dump(dict(config), allow_unicode=True, default_flow_style=False))

rl(utils)
rl(utils.eval_utils)
from utils.eval_utils import *
rl(data_utils)
rl(data_utils.datasets)
rl(data_utils.speechbci_dataset)
from data_utils.speechbci_dataset import *
from data_utils.datasets import *
rl(models)
rl(models.trainer)
rl(models.patchtst)
rl(models.ndt1)
from models.patchtst import *
from models.ndt1 import *
from models.trainer import Trainer



# config["model"]["encoder"]["from_pt"] = from_pt
# config["model"]["decoder"]["from_pt"] = from_pt
config["verbosity"] = 0
trainer = Trainer(config, dataset=dataset, metric_fns={"A": metric_1})#, "WER": wer})
trainer.model.load_checkpoint(from_pt)
di = iter(trainer.train_dataloader)
ex = next(di)

with torch.no_grad():
    preds = trainer.model.generate(**ex[0], max_new_bins=128)

trainer.train()


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

def metric_1(model, model_inputs, unused_inputs, outputs, **kwargs):
    a = model_inputs
    b = unused_inputs
    c = outputs
    torch.save({"inputs": a, "unused": b, "outputs": c}, "b.pth")
    return torch.tensor(0.0)


b = torch.load("b.pth")
inputs = b["inputs"]
unused = b["unused"]
outputs = b["outputs"]
preds = outputs["preds"]
targets = outputs["targets"]
mask = outputs["mask"]