import yaml
import os
import sys
import json
import torch
import matplotlib.pyplot as plt
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

all = []
def metric_1(model, model_inputs, unused_inputs, outputs, **kwargs):
    a = model_inputs
    b = unused_inputs
    c = outputs
    torch.save({"inputs": a, "unused": b, "outputs": c}, "b.pth")
    all.append({"inputs": a, "unused": b, "outputs": c})
    return torch.tensor(0.0)


kwargs = {
    "savestring": "test",
    "training.num_epochs": "200", "training.train_batch_size": "64", "training.test_batch_size": "64",
    "optimizer.gradient_accumulation_steps": "1",
    "training.eval_every": "100", "training.save_every": "500", 
    "data.train_len": "null", "data.test_len": "null",
    "model": "include:configs/ndt1s.yaml",
    "data.data_name": "maze",
}
config_file = "configs/trainer_ssl2.yaml"
config = update_config(default_trainer_config(), config_file)
config = update_config(config, config_from_kwargs(kwargs))

# # # Load
# if config.data.data_name == "maze":
#         dataset = torch.load(config.data.data_file)
# elif config.data.data_name == "speechbci":
#     dataset = load_competition_data(config.data.dataset_dir, **config.data)
#     if "vocab_file" in config["data"] and config.data.vocab_file is not None:
#         blank_id = config.method.model_kwargs.blank_id
#         vocab = json.load(open(config.data.vocab_file,"r"))
#         dataset = create_phonemes_ctc_labels(dataset, config.data.vocab_file)

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
config["savestring"] = "mlm_ndt_poisson"
config["model"]["encoder"]["masker"]["active"] = True
config["model"]["encoder"]["context"]["forward"] = -2
config["method"]["model_kwargs"]["loss"] = "poisson_nll"
config["method"]["model_kwargs"]["log_input"] = True
config["method"]["model_kwargs"]["method_name"] = "mlm"
config["training"]["eval_every"] = 100
config["verbosity"] = 0

from_pt = "pt_checkpoints/mlm_ndt_mse/STEP6000"
config = DictConfig(torch.load(os.path.join(from_pt, "trainer_config.pth")))

all = []
trainer = Trainer(config, dataset=dataset, metric_fns={"A": metric_1})#, "WER": wer})
trainer.model.load_checkpoint(from_pt)
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




b = torch.load("b.pth")
all_preds = []
all_targets = []
all_mask = []
for b in all:
    inputs = b["inputs"]
    unused = b["unused"]
    outputs = b["outputs"]
    preds = outputs["preds"]
    targets = outputs["targets"]
    mask = outputs["mask"]
    all_preds.append(preds)
    all_targets.append(targets)
    all_mask.append(mask)

preds = torch.cat(all_preds,0)
targets = torch.cat(all_targets,0)
mask = torch.cat(all_mask,0)


preds_m = preds[:,:-1,:].detach().cpu()  # [mask[:,:-1].bool()]
if trainer.model.loss_name == "poisson_nll" and trainer.model.log_input:
    preds_m = preds_m.exp()

targets_m = targets[:,1:,:].detach().cpu()


plt.clf()
i = 997
j = 135
targets_m[i,:,j]
plt.plot((preds_m[i,:,j] - preds_m[i,:,j].mean())/preds_m[i,:,j].std(), label="pred")
plt.plot((targets_m[i,:,j] - targets_m[i,:,j].mean())/targets_m[i,:,j].std(), label="targets")
plt.legend()
plt.savefig(f"train_mlm_mse_t_0.png")


fig, ax = rate_vs_spikes(preds_m,targets_m) 
fig, ax = rate_vs_spikes(preds[mask.bool()].detach().cpu(),targets[mask.bool()].detach().cpu()) 
ax.set_ylim(0.0,0.6)
fig.savefig("plot_autor_mse_train.png")

def rate_vs_spikes(preds, targets, max_count=3, yaxis="rates"):
    nspikes = list(range(max_count+1))
    mean_y = [preds[targets == i].mean() for i in nspikes]
    std_y = [preds[targets == i].std() for i in nspikes]
    fig, ax = plt.subplots()
    ax.errorbar(nspikes, mean_y, yerr=std_y, fmt='o', color='red', label='Error bars')
    ax.set_xlabel('Observed spikes')
    ax.set_ylabel(f'Mean predicted {yaxis}')
    return fig, ax



di = iter(trainer.test_dataloader)
ex = next(di)

# for b in [1,5,20,40]:
b = 5
max_new_bins = b

all_inputs = []
all_targets = []
for ex in di:
    inputs = {
        "spikes": ex[0]["spikes"][:,:-max_new_bins,:],
        "spikes_mask": ex[0]["spikes_mask"][:,:-max_new_bins],
        "spikes_timestamp": ex[0]["spikes_timestamp"][:,:-max_new_bins],
        "spikes_lengths": ex[0]["spikes_lengths"] - torch.maximum(torch.tensor(0),(max_new_bins + (ex[0]["spikes_mask"]-1).sum(1)))
    }
    targets = ex[0]["spikes"][:,-max_new_bins:,:]
    all_targets.append(targets)
    all_inputs.append(inputs)


inputs = {k: torch.cat([row[k] for row in all_inputs],0) for k in all_inputs[0]}
targets = torch.cat(all_targets,0)

with torch.no_grad():
    preds = trainer.model.generate(**inputs, max_new_bins=max_new_bins)


nspikes = [0,1,2,3]
mean_rates = [preds[targets == i].mean().item() for i in range(4)]
std_rates = [preds[targets == i].std().item() for i in range(4)]
plt.clf()
plt.errorbar(nspikes, mean_rates, yerr=std_rates, fmt='o', color='red', label='Error bars')
plt.xlabel('Observed spikes')
plt.ylabel('Mean predicted poisson rate')
plt.ylim(0.0,0.6)
# plt.ylim(0,0.5)
plt.savefig(f"gen_autor_mse.png")


plt.clf()
preds = preds.detach().cpu()
targets = targets.detach().cpu()
plt.plot((preds[i,:,134] - preds[15,:,134].mean())/preds[15,:,134].std(), label="pred")
plt.plot((targets[15,:,134] - targets[15,:,134].mean())/targets[15,:,134].std(), label="targets")
plt.legend()
plt.savefig(f"gen_autor_mse_t.png")

mse = []
for i in range(max_new_bins):
    preds_i = preds[:,i,:]
    targets_i = targets[:,i,:]
    mse.append(torch.nn.functional.mse_loss(preds_i, targets_i).item())

mse

false_pos = []
true_pos = []
false_neg = []
true_neg = []
for i in range(max_new_bins):
    preds_i = preds[:,i,:]
    targets_i = targets[:,i,:]
    false_pos.append(((preds_i[targets_i == 0] > 0.1).sum() / preds_i[targets_i == 0].nelement()).item())
    true_pos.append(((preds_i[targets_i > 0] > 0).sum() / preds_i[targets_i > 0].nelement()).item())
    false_neg.append(((preds_i[targets_i > 0] == 0).sum() / preds_i[targets_i > 0].nelement()).item())
    true_neg.append(((preds_i[targets_i == 0 ] == 0).sum() / preds_i[targets_i == 0].nelement()).item())


false_pos
true_pos
false_neg
true_neg



