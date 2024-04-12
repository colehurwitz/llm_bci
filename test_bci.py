import yaml
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from importlib import reload as rl
from utils.config_utils import config_from_kwargs, update_config, DictConfig

import utils
import utils.eval_utils
from utils.eval_utils import format_ctc, word_error_count

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, LlamaConfig, AutoTokenizer

import data_utils
import data_utils.datasets
from data_utils.speechbci_dataset import load_competition_data, create_phonemes_ctc_labels, create_llm_labels
from data_utils.ibl_dataset import load_ibl_dataset

import models
import models.trainer
import models.patchtst
import models.ndt1
import models.itransformer
from models.trainer import Trainer, default_trainer_config


rl(utils)
rl(utils.eval_utils)
from utils.eval_utils import *

rl(data_utils)
rl(data_utils.datasets)
rl(data_utils.speechbci_dataset)
rl(data_utils.ibl_dataset)
from data_utils.speechbci_dataset import *
from data_utils.ibl_dataset import *
from data_utils.datasets import *

rl(models)
rl(models.trainer)
rl(models.bci)
rl(models.itransformer)
rl(models.ndt1)
from models.bci import *
from models.ndt1 import *
from models.itransformer import *
from models.trainer import Trainer


def probe(model, model_inputs, unused_inputs, outputs, **kwargs):
    a = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in model_inputs.items()}
    b = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in unused_inputs.items()}
    c = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in outputs.items() if v is not None}
    all_batch.append({"inputs": a, "unused": b, "outputs": c})
    return torch.tensor(0.0)


def wer(model, model_inputs, unused_inputs, outputs, **kwargs):
    preds = outputs["preds"].argmax(-1)[:,:-1]
    targets = outputs["targets"][:,1:]
    pred_sentences = [tokenizer.decode(p[t!=100]) for t, p  in zip(targets,preds)]
    target_sentences = unused_inputs["sentence"]
    errors, n_words = word_error_count(pred_sentences, target_sentences)
    for i in range(kwargs["n_print"]):
        print(pred_sentences[i], "\n#####\n ", 
            target_sentences[i], "\n#####\n\n ")
    return torch.tensor(errors/n_words)


llm = None
def get_llm(debug, freeze_llm, llm_path, lora):
    if debug:
        llm_config = LlamaConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=32,  num_attention_heads=4)
        llm = AutoModelForCausalLM.from_config(llm_config)
    else:
        llm = AutoModelForCausalLM.from_pretrained(llm_path) 
    if lora is not None:
        peft_config = LoraConfig(
            inference_mode=False, r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
            target_modules=lora.target_modules, modules_to_save=lora.modules_to_save,
        )
        llm = get_peft_model(llm, peft_config)  
    if freeze_llm:
        for param in llm.parameters():
            param.requires_grad = False
    return llm

# Preload LLM
debug = True
freeze_llm = False
llm_path = "/home/gridsan/dbeneto/MAML-Soljacic_shared/llama2/7b"
lora = DictConfig({"r": 1, "alpha": 32, "dropout": 0.2, 
      "target_modules": ["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
      "modules_to_save": ["embed_tokens"]})
llm = get_llm(debug, freeze_llm, llm_path, lora)


kwargs = {
    "savestring": "bci_test",
    "training.num_epochs": "500", "training.train_batch_size": "4", "training.test_batch_size": "4",
    "optimizer.gradient_accumulation_steps": "1",
    "training.eval_every": "100", "training.save_every": "100", 
    "data.train_len": "null", "data.test_len": "null",
    "model": "include:configs/bci.yaml",
}
config_file = "configs/trainer_bci.yaml"
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
    if "tokenizer_path" in config["data"] and config.data.tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path, add_bos_token=False, add_eos_token=False)
        dataset = create_llm_labels(dataset, tokenizer, config.data.prompt)


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


# Adjust models based on dataset
extra_model_kwargs = None
spikes_name = "spikes" if "spikes" in dataset["train"][0] else config.method.dataset_kwargs.spikes_name
if config.model.model_class in ["iTransformer","PatchTST"]:
    config["model"]["encoder"]["num_input_channels"] = dataset["train"][0][spikes_name].shape[1]
    # We need uniform lenght of the padded batches for PatchTST and iTransformer
    if config.model.model_class == "PatchTST":
        p = config.model.encoder.patch_length
        context = ((max(row[spikes_name].shape[0] for split in ["train","test"] for row in dataset[split]) + p-1) // p) * p
        config["model"]["encoder"]["context_length"] = context
    else:
        context = max(row[spikes_name].shape[0] for split in ["train","test"] for row in dataset[split])
        config["model"]["encoder"]["max_n_bins"] = context
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
elif config.model.model_class == "BCI" and llm is not None:
    if extra_model_kwargs is None:
        extra_model_kwargs = {}
    extra_model_kwargs["llm"] = llm



rl(models)
rl(models.trainer)
rl(models.bci)
rl(models.itransformer)
rl(models.ndt1)
from models.bci import *
from models.ndt1 import *
from models.itransformer import *
from models.trainer import Trainer

config["method"]["model_kwargs"]["load_ndt1_from_pt"] = "pt_checkpoints/ndt1-ctc-opt_64_1.e-3_5.e-5-nl_5-hs_1024-is_1024-d_0.4_0.2-noise_true-context_false_false_false/STEP12200"
trainer = Trainer(config, dataset=dataset, extra_model_kwargs=extra_model_kwargs, metric_fns={"A": probe, "WER": wer})#, "WER": wer})
all_batch = []
trainer.evaluate(eval_train_set=False)
trainer.train()

a = torch.load("a.pth")
b = torch.load("b.pth")
input_ids = a[0]
attention_mask = a[1]
input_split = a[2]
spikes = a[3]
spikes_mask = a[4]
targets = a[5]
input_embeds = b[0]
new_attention_mask = b[1]
new_spikes_mask = b[2]
new_targets = b[3]

# config["model"]["encoder"]["from_pt"] = from_pt
# config["model"]["decoder"]["from_pt"] = from_pt
# config["savestring"] = "mlm_ndt_poisson"
# config["model"]["encoder"]["context"]["forward"] = -2
# config["method"]["model_kwargs"]["loss"] = "poisson_nll"
# config["method"]["model_kwargs"]["log_input"] = True
# config["method"]["model_kwargs"]["method_name"] = "mlm"
# config["training"]["eval_every"] = 100
# config["verbosity"] = 0
# config["model"]["masker"]["mode"] = "full"
# config["model"]["masker"]["mode"] = "full"


from_pt = "pt_checkpoints/ndt1-ctc-opt_64_1.e-3_5.e-5-nl_5-hs_1024-is_1024-d_0.4_0.2-noise_true-context_false_false_false/STEP12200"
config = DictConfig(torch.load(os.path.join(from_pt, "trainer_config.pth")))
trainer.model.load_checkpoint(from_pt)





static_behaviours = ["block","reward","choice"]
if config.method.dataset_kwargs.targets_name in static_behaviours:
    static_behaviours.remove(config.method.dataset_kwargs.targets_name)
    target_as_stat = True
else:
    target_as_stat = False


all_preds = []
all_targets = []
all_stat_behv = {k: [] for k in static_behaviours}
for b in all_batch:
    inputs = b["inputs"]
    unused = b["unused"]
    outputs = b["outputs"]
    preds = outputs["preds"]
    targets = outputs["targets"]
    all_preds.append(preds)
    all_targets.append(targets)
    for k in static_behaviours:
        all_stat_behv[k].append(unused[k])

preds = torch.cat(all_preds,0).detach().cpu().exp()
targets = torch.cat(all_targets,0).detach().cpu()
stat_behv = {k: torch.cat(v,0).detach().cpu() for k,v in all_stat_behv.items()}
if target_as_stat:
    stat_behv[config.method.dataset_kwargs.targets_name] = targets

### SUHQI PLOTS ### 
import viz_neuron_fit
from viz_neuron_fit import viz_single_cell


ys = targets # [#trials, #timesteps, #neurons]
y_preds = preds # [#trials, #timesteps, #neurons]
var_name2idx = {}
X = []
for i, (k, v) in enumerate(stat_behv.items()):
    X.append(v.unsqueeze(1).expand(preds.shape[:2]))
    var_name2idx[k] = [i]


X = torch.stack(X,2).numpy()
var_value2label = {'block': {(0.2,): "p(left)=0.2",
                            (0.5,): "p(left)=0.5",
                            (0.8,): "p(left)=0.8",},
                   'choice': {(-1.0,): "right",
                            (1.0,): "left"},
                   'reward': {(0.,): "no reward",
                            (1.,): "reward", } }
var_tasklist = static_behaviours
var_behlist = []

import viz_neuron_fit

from viz_neuron_fit import viz_single_cell

tmp = ys.mean(axis=(0,1))
neurons = np.argsort(tmp)[-5:]
rl(viz_neuron_fit)
for ni in neurons:
    viz_single_cell(X,ys[:,:,ni],y_preds[:,:,ni], 
                    var_name2idx, var_tasklist, var_value2label, var_behlist,
                    subtract_psth="task", aligned_tbins=[26], neuron_idx=ni, save_name=f"single_neuron_{ni}_itransformer.png")

### PLOTTING CONDITION-AVERAGED AND SINGLE-TRIAL ###

def gaussian_kernel(kernel_size, sigma):
    kernel = np.exp(-(np.arange(-kernel_size//2, kernel_size//2 + 1)**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return torch.tensor(kernel, dtype=torch.float32)


def plot_single_trials(targets, preds, k=10, n=5, split="train", start_gen=None):
    v, neurons = torch.topk(targets.mean((0,1)), k=k)
    neurons = [neuron.item() for neuron in neurons]
    neurons.append(75)
    kernel_size = 4
    sigma = 1.5
    kernel = gaussian_kernel(kernel_size, sigma).view(1, 1, -1)
    for neuron in neurons:
        fig, ax = plt.subplots(n, sharey=True)
        for i in range(n):
            for c, color in zip([-1,1],["blue","red"]):
                p = preds[choice == c][i,kernel_size//2:-kernel_size//2,neuron]
                t = targets[choice == c][i,:,neuron]
                t_smooth = F.conv1d(t.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2).squeeze()[kernel_size//2:-kernel_size//2]
                r2 = r2_score(t_smooth, p)
                ax[i].plot(range(t_smooth.size(0)), t_smooth, color=color, linewidth=2, alpha=0.5, label=f"Targets choice {c}")
                ax[i].plot(range(p.size(0)), p, color=color, linewidth=0.5,  label=f"Predictions choice {c}")
                ax[i].text(0.57+ 0.1*c, 0.75, "R2={:.2f}".format(r2), color=color, transform=ax[i].transAxes)
                if start_gen is not None:
                    ax[i].axvline(start_gen, color="black", linewidth=0.5)
        ax[0].legend(loc='upper right', bbox_to_anchor=(1.65, 0.5))
        ax[-1].set_xlabel("Timesteps")
        fig.suptitle(f"smoothed activity vs predicted rates. \nSession f312aaec-3b6f-44b3-86b4-3a0c119c0438   Neuron #{neuron}")
        fig.subplots_adjust(right=0.65, hspace=0.5)
        fig.savefig(f"{split}_single_trial_{neuron}{'_'+str(start_gen) if start_gen else ''}.png")


plot_single_trials(targets, preds)



def plot_condition_averaged(targets, preds, k=10, split="test", start_gen=None):
    v, neurons = torch.topk(targets.mean((0,1)), k=k)
    neurons = [neuron.item() for neuron in neurons]
    neurons.append(75)
    kernel_size = 4
    sigma = 1.5
    kernel = gaussian_kernel(kernel_size, sigma).view(1, 1, -1)
    for neuron in neurons:
        fig, ax = plt.subplots(2, sharey=True)
        for c, color in zip([-1,1],["blue","red"]):
            p = preds[choice == c][:,kernel_size//2:-kernel_size//2,neuron]
            t = targets[choice == c][:,:,neuron]
            t_smooth = F.conv1d(t.unsqueeze(1), kernel, padding=kernel_size//2).squeeze(1)[:,kernel_size//2:-kernel_size//2]
            r2 = r2_score(t_smooth.mean(0), p.mean(0))
            ax[0].fill_between(range(t_smooth.size(1)), t_smooth.mean(0) - 0.1*t_smooth.std(0), t_smooth.mean(0) + 0.1*t_smooth.std(0), alpha=0.7, color=color, label=f"Choice {c}")
            ax[1].fill_between(range(p.size(1)), p.mean(0) - 0.1*p.std(0), p.mean(0) + 0.1*p.std(0), alpha=0.7, color=color)
            ax[1].text(0.65+ 0.1*c, 0.75, "R2={:.2f}".format(r2), color=color, transform=ax[1].transAxes)
        fig.suptitle(f"smoothed activity vs predicted rates. \nSession f312aaec-3b6f-44b3-86b4-3a0c119c0438   Neuron #{neuron}")
        if start_gen is not None:
            ax[1].axvline(start_gen, color="black", linewidth=0.5)
            ax[0].axvline(start_gen, color="black", linewidth=0.5)
        ax[0].set_xlabel("Timesteps")
        ax[1].set_xlabel("Timesteps")
        ax[0].set_ylabel("Smoothed activity")
        ax[1].set_ylabel("NDT1")
        ax[0].legend(loc='upper right', bbox_to_anchor=(1.45, 0.0))
        fig.subplots_adjust(right=0.65, hspace=0.5)
        fig.savefig(f"{split}_condition_averaged_{neuron}{'_'+str(start_gen) if start_gen else ''}.png")
        plt.close()

plot_condition_averaged(targets, preds)



####







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



di = iter(trainer.train_dataloader)
# ex = next(di)

# for b in [1,5,20,40]:
b = 60
max_new_bins = b

all_inputs = []
all_targets = []
all_choice = []
for ex in di:
    inputs = {
        "spikes": ex[0]["spikes"][:,:-max_new_bins,:],
        "spikes_mask": ex[0]["spikes_mask"][:,:-max_new_bins],
        "spikes_timestamp": ex[0]["spikes_timestamp"][:,:-max_new_bins],
        "spikes_lengths": ex[0]["spikes_lengths"] - torch.maximum(torch.tensor(0),(max_new_bins + (ex[0]["spikes_mask"]-1).sum(1)))
    }
    targets = ex[0]["spikes"]
    choice = ex[1]["choice"]
    all_targets.append(targets)
    all_inputs.append(inputs)
    all_choice.append(choice)


inputs = {k: torch.cat([row[k] for row in all_inputs],0) for k in all_inputs[0]}
targets = torch.cat(all_targets,0).detach().cpu()
choice = torch.cat(all_choice,0).detach().cpu()

with torch.no_grad():
    preds, bins = trainer.model.generate(**inputs, max_new_bins=max_new_bins)
    preds = preds.detach().cpu()
    bins = bins.detach().cpu()

preds = torch.cat((targets[:,:-max_new_bins,:], preds),1)

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

