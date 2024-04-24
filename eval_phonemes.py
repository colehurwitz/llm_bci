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
    
    def wer(model, model_inputs, unused_inputs, outputs, **kwargs):
        prompt_ids = model_inputs["input_ids"][torch.logical_and(model_inputs["targets"] == -100, model_inputs["input_ids"] != tokenizer.unk_token_id)]
        if len(prompt_ids.size()) == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        attention_mask = torch.ones_like(prompt_ids)
        model_inputs.update({
            "input_ids": prompt_ids,
            "attention_mask": attention_mask,
        })
        model_inputs.pop("targets")
        beams = kwargs["n_beams"]
        if beams > 1:
            gen_config = {
                "max_new_tokens": 20, 
                "do_sample": False, #"temperature": 1.0,  "top_p": 0.6, "top_k": 40, 
                "num_beams": beams, 
                "num_beam_groups": beams, "diversity_penalty": 1.2,
                "repetition_penalty": 1.0, "length_penalty": 1.0, "no_repeat_ngram_size": 2, 
                "renormalize_logits": True, 
                "low_memory": True,
                "num_return_sequences": beams, "output_scores": True, "return_dict_in_generate": True,
                "pad_token_id": tokenizer.unk_token_id,
            }
        else:
            gen_config = {
                "max_new_tokens": 20, 
                "do_sample": False,
                "low_memory": True,
                "pad_token_id": tokenizer.unk_token_id,
            }
        preds = model.generate(**model_inputs, **gen_config)
        if beams > 1:
            pred = preds.sequences[0]
        else:
            pred = preds[0]
        pred_sentence = tokenizer.decode(pred, skip_special_tokens=True).strip()
        target_sentence = unused_inputs["sentence"][0]
        errors, n_words = word_error_count(pred_sentence, target_sentence)
        print("-" + pred_sentence + "-", "\n#####\n")
        print("-" + target_sentence + "-", "\n#####\n\n ")
        all_preds.append((preds, target_sentence))
        torch.save(all_preds, f"{savestring}_dec.pth")
        return torch.tensor(errors/n_words)


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
    trainer = Trainer(config, dataset=dataset, metric_fns={"WER": wer})
    all_preds = []
    trainer.evaluate(eval_train_set=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)
