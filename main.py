import json
import yaml
import argparse

import torch
from utils.config_utils import config_from_kwargs, update_config, ParseKwargs, DictConfig
from utils.eval_utils import format_ctc, word_error_count
from data_utils.speechbci_dataset import load_competition_data, create_phonemes_ctc_labels
from models.trainer import Trainer, default_trainer_config

def main(args):
    config = update_config(default_trainer_config(), args.config_file if args.config_file != "none" else None) 
    config = update_config(config, config_from_kwargs(args.kwargs))

    # Load raw dataset and prepare specific metrics
    metric_fns = None
    if config.data.data_name == "maze":
        dataset = torch.load(config.data.data_file)
    elif config.data.data_name == "speechbci":
        dataset = load_competition_data(config.data.dataset_dir, **config.data)
        if "vocab_file" in config["data"] and config.data.vocab_file is not None:
            dataset = create_phonemes_ctc_labels(dataset, config.data.vocab_file)

            blank_id = config.method.model_kwargs.blank_id
            vocab = json.load(open(config.data.vocab_file,"r"))
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

            metric_fns = {"WER": wer}

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


    trainer = Trainer(config, dataset=dataset, metric_fns=metric_fns)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)