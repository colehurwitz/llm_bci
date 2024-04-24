import argparse
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
from utils.config_utils import config_from_kwargs, update_config, DictConfig, ParseKwargs

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


def main(args):
    kwargs = config_from_kwargs(args.kwargs)
    beams = kwargs.beams if "beams" in kwargs else 1
    from_pt = kwargs.from_pt if "from_pt" in kwargs else "/home/gridsan/dbeneto/TFG/BCI/bci_checkpoints/seed_3-bci_train_null-opt_8_1.e-4_5.e-5-arch_2048-stack_1-lora_8-freeze_false-pt_pt_checkpoints/ndt1-ctc-seed_2-opt_64_1.e-3_5.e-5-arch_5_1024_1024-d_0.4_0.2-noise_true-smooth_2-context_-2_false_false_false-ds-decoding_true_true-stack_32_4/STEP10300/STEP36000"
    savestring = kwargs.savestring if "savestring" in kwargs else "test_decoding"
    iaifi = kwargs.iaifi if "iaifi" in kwargs else False
    test_len = kwargs.test_len if "test_len" in kwargs else None
    
    
    def assisted_wer(model, model_inputs, unused_inputs, outputs, **kwargs):
        preds = outputs["preds"].argmax(-1)[:,:-1]
        targets = outputs["targets"][:,1:]
        pred_sentences = [tokenizer.decode(p[t!=-100], skip_special_tokens=True) for t, p  in zip(targets,preds)]
        target_sentences = unused_inputs["sentence"]
        errors, n_words = word_error_count(pred_sentences, target_sentences)
        for i in range(kwargs["n_print"]):
            print("-" + pred_sentences[i] + "-", "\n#####\n")
            print("-" + target_sentences[i] + "-", "\n#####\n\n ")
        return torch.tensor(errors/n_words, device=model_inputs["spikes"].device)


    config = DictConfig(torch.load(os.path.join(from_pt, "trainer_config.pth")))
    if iaifi:
        iaifi_dirs = update_config("configs/iaifi_dirs.yaml")
        config["dirs"]["checkpoint_dir"] = iaifi_dirs.checkpoint_dir
        config["dirs"]["log_dir"] = iaifi_dirs.log_dir
        config["data"]["data_dir"] = iaifi_dirs.data_dir
        config["data"]["tokenizer_path"] = iaifi_dirs.tokenizer_path
        config["method"]["model_kwargs"]["llm_path"] = iaifi_dirs.llm_path



    config["model"]["from_pt"] = from_pt
    config["training"]["test_batch_size"] = 1
    config["data"]["test_len"] = test_len
    config["method"]["metric_kwargs"]["n_beams"] = 1
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


    config["method"]["metric_kwargs"]["n_beams"] = beams
    trainer = Trainer(config, dataset=dataset, metric_fns={"A-WER": assisted_wer})
    all_preds = []
    trainer.evaluate(eval_train_set=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)