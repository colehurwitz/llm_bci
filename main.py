import os
import json
import yaml
import argparse

import numpy as np

import torch
from utils.config_utils import config_from_kwargs, update_config, ParseKwargs, DictConfig
from utils.eval_utils import format_ctc, word_error_count
from data_utils.speechbci_dataset import load_competition_data, create_phonemes_ctc_labels
from data_utils.ibl_dataset import load_ibl_dataset
from models.trainer import Trainer, default_trainer_config

def main(args):
    config = update_config(default_trainer_config(), args.config_file if args.config_file != "none" else None) 
    config = update_config(config, config_from_kwargs(args.kwargs))
    
    metric_fns = {}
    eval_metric_fns = {}

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
        
        def accuracy(model, model_inputs, unused_inputs, outputs, **kwargs):
            preds = outputs["preds"].argmax(-1)
            targets = model_inputs["targets"].squeeze(1)
            acc = (preds == targets).sum() / preds.size(0)
            return acc
        metric_fns.update({"accuracy": accuracy})
    
    # Add CTC metric fns
    if config.method.model_kwargs.method_name == "ctc":
        def cer(model, model_inputs, unused_inputs, outputs, **kwargs):
            preds = outputs["preds"].argmax(-1)
            preds = [" ".join(format_ctc(pred, vocab, blank_id)) for pred in preds]
            phonemes = [" ".join(p) for p in unused_inputs["phonemes"]]
            errors, n_phonemes = word_error_count(preds, phonemes)
            return torch.tensor(errors/n_phonemes, device=model_inputs["spikes"].device)
        metric_fns.update({"CER": cer})

        def eval_cer(model, model_inputs, unused_inputs, outputs, **kwargs):
            preds = outputs["preds"].argmax(-1)
            preds = [" ".join(format_ctc(pred, vocab, blank_id)) for pred in preds]
            phonemes = [" ".join(p) for p in unused_inputs["phonemes"]]
            errors, n_phonemes = word_error_count(preds, phonemes)
            for i in range(kwargs["n_print"]):
                print(preds[i].replace(" ","").replace("SIL"," SIL "), "\n#####\n ", 
                    phonemes[i].replace(" ","").replace("SIL"," SIL "),"\n#####\n ", 
                    unused_inputs["sentence"][i], "\n#####\n\n ")
            return torch.tensor(errors/n_phonemes, device=model_inputs["spikes"].device)
        eval_metric_fns.update({"CER": eval_cer})


    # Get regions for region embeddings
    if config.model.model_class == "iTransformer" and config.model.encoder.embed_region:
        config["model"]["encoder"]["regions"] = list(set(str(b) for a in [row["regions"] for rows in dataset.values() for row in rows] for b in a))


    # Adjust models based on dataset
    spikes_name = "spikes" if "spikes" in dataset["train"][0] else config.method.dataset_kwargs.spikes_name
    if config.model.model_class in ["iTransformer","PatchTST"]:
            
        # We need uniform lenght of the padded batches for PatchTST and iTransformer
        if config.model.model_class == "PatchTST":
            config["model"]["encoder"]["num_input_channels"] = dataset["train"][0][spikes_name].shape[1]
            p = config.model.encoder.patch_length
            context = ((max(row[spikes_name].shape[0] for split in ["train","test"] for row in dataset[split]) + p-1) // p) * p
            config["model"]["encoder"]["context_length"] = context
        else:
            context = max(row[spikes_name].shape[0] for split in ["train","test"] for row in dataset[split])
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


    if metric_fns == {}:
        metric_fns = None
    if eval_metric_fns == {}:
        eval_metric_fns = None
    trainer = Trainer(config, dataset=dataset, metric_fns=metric_fns, eval_metric_fns=eval_metric_fns)
    trainer.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)