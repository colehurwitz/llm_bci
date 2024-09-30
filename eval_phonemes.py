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
import utils.eval_bci
from utils.eval_bci import format_ctc, word_error_count

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
                "repetition_penalty": 1.0, "length_penalty": 1.0, 
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
            preds = preds.sequences.detach().cpu()
        else:
            preds = preds.detach().cpu()
        pred = preds[0]
        pred_sentence = tokenizer.decode(pred, skip_special_tokens=True).strip()
        target_sentence = unused_inputs["sentence"][0]
        errors, n_words = word_error_count(pred_sentence, target_sentence)
        print("-" + pred_sentence + "-", "\n#####\n")
        print("-" + target_sentence + "-", "\n#####\n\n ")
        all_preds.append((preds, target_sentence))
        torch.save(all_preds, f"{savestring}.pth")
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
    config["method"]["metric_kwargs"]["n_beams"] = beams

    # Load dataset
    dataset = load_competition_data(**config.data)
    blank_id = config.method.model_kwargs.blank_id
    # vocab = json.load(open(config.data.vocab_file,"r"))
    # dataset = create_phonemes_ctc_labels(dataset, config.data.vocab_file)
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path, add_bos_token=False, add_eos_token=False)
    dataset = create_llm_labels(dataset, tokenizer, config.data.prompt)


    trainer = Trainer(config, dataset=dataset, metric_fns={"WER": wer})
    all_preds = []
    trainer.evaluate(eval_train_set=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)




import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaConfig, AutoTokenizer
from utils.eval_bci import word_error_count
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("/home/gridsan/dbeneto/MAML-Soljacic_shared/llama2/tokenizer")



################### COMPUTE STATISTICS #############################################

all_data = {"lora": {}, "freeze": {}}
for beams in [1,3,5,10,25,50]:
    for name in ["lora","freeze"]:
        a = torch.load(f"/home/gridsan/dbeneto/TFG/BCI/plots/bci/{name}_{beams}.pth")
        words = []
        errors = []
        best_errors = []
        for row in a:
            sentence = row[1]
            preds = tokenizer.batch_decode(row[0],skip_special_tokens=True)
            all_new_errors = []
            for pred in preds:
                new_errors, new_words = word_error_count(pred.strip(), sentence)
                all_new_errors.append(new_errors)
            words.append(new_words)
            errors.append(all_new_errors[0])
            best_errors.append(min(all_new_errors))
        words = np.array(words)
        errors = np.array(errors)
        best_errors = np.array(best_errors)
        n_resamples = 10000
        resampled_wer = np.zeros([n_resamples,])
        resampled_best_wer = np.zeros([n_resamples,])
        for i in range(n_resamples):
            resample_idx = np.random.randint(0, words.shape[0], [words.shape[0]])
            resampled_wer[i] = np.sum(errors[resample_idx]) / np.sum(words[resample_idx])
            resampled_best_wer[i] = np.sum(best_errors[resample_idx]) / np.sum(words[resample_idx])
        all_data[name][beams] = (resampled_wer, resampled_best_wer)
        



###### WER #######################################################

labels = ["1", "3","5","10","25","50"]
plt.clf()
x = np.arange(len(labels))  # Create an array for the x-axis
bar_width = 0.35  # Width of each bar
plt.figure(figsize=(10, 6))
plt.bar(
    x - bar_width/2, 
    [100*row[0].mean() for row in all_data["lora"].values()], 
    yerr=[[100*(row[0].mean() - np.percentile(row[0], [2.5, 97.5])[0]) for row in all_data["lora"].values()], [100*(np.percentile(row[0], [2.5, 97.5])[1]-row[0].mean()) for row in all_data["lora"].values()]],
    width=bar_width, label="Finetune LLM", color='lightgreen'
) 

plt.bar(
    x + bar_width/2, 
    [100*row[0].mean() for row in all_data["freeze"].values()], 
    yerr=[[100*(row[0].mean()-np.percentile(row[0], [2.5, 97.5])[0]) for row in all_data["freeze"].values()], [100*(np.percentile(row[0], [2.5, 97.5])[1]-row[0].mean()) for row in all_data["freeze"].values()]],
    width=bar_width, label="Freeze LLM", color='lightblue'
) 
plt.xlabel(r"Beam size", fontsize=15)
plt.xticks(x, labels, fontsize=15) 
plt.ylabel("Word Error Rate (%)", fontsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.ylim(20,33.5)
plt.grid(True)
plt.minorticks_on() 
plt.grid(True, which='minor', linestyle='--', linewidth=0.5) 
plt.legend(loc='upper center',bbox_to_anchor=(0.35, 1), fancybox=True, ncol=1, fontsize=15)

# plt.legend(fontsize=14)
plt.savefig("plots/bci/wer.png")
####################################################33



###### BEST WER #######################################################

labels = ["1", "3","5","10","25","50"]
plt.clf()
x = np.arange(len(labels))  # Create an array for the x-axis
bar_width = 0.35  # Width of each bar
plt.figure(figsize=(10, 6))
plt.bar(
    x - bar_width/2, 
    [100*row[1].mean() for row in all_data["lora"].values()], 
    yerr=[[100*(row[1].mean() - np.percentile(row[1], [2.5, 97.5])[0]) for row in all_data["lora"].values()], [100*(np.percentile(row[1], [2.5, 97.5])[1]-row[1].mean()) for row in all_data["lora"].values()]],
    width=bar_width, label="Finetune LLM", color='lightgreen'
) 

plt.bar(
    x + bar_width/2, 
    [100*row[1].mean() for row in all_data["freeze"].values()], 
    yerr=[[100*(row[1].mean()-np.percentile(row[1], [2.5, 97.5])[0]) for row in all_data["freeze"].values()], [100*(np.percentile(row[1], [2.5, 97.5])[1]-row[1].mean()) for row in all_data["freeze"].values()]],
    width=bar_width, label="Freeze LLM", color='lightblue'
) 
plt.xlabel(r"Top-$k$", fontsize=15)
plt.xticks(x, labels, fontsize=15) 
plt.ylabel("Best Word Error Rate (%)", fontsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.yticks(np.arange(10, 35, 5))
plt.ylim(10,32)
plt.grid(True)
plt.minorticks_on() 
plt.grid(True, which='minor', linestyle='--', linewidth=0.5) 
plt.legend(loc='upper center',bbox_to_anchor=(0.65, 0.95), fancybox=True, ncol=1, fontsize=15)

# plt.legend(fontsize=14)
plt.savefig("plots/bci/best_wer.png")
####################################################33


######### EXAMPLE SENTENCES ############3

a = torch.load(f"/home/gridsan/dbeneto/TFG/BCI/plots/bci/lora_5.pth")
np.percentile(all_data["lora"][5][0], [2.5, 97.5])
np.mean(all_data["lora"][5][0])
all_examples = []
for row in a:
    sentence = row[1]
    preds = tokenizer.batch_decode(row[0],skip_special_tokens=True)
    errors, words = word_error_count(preds[0].strip(), sentence)
    all_examples.append([preds[0].strip(), sentence, errors, words])

json.dump(all_examples,open("best_wer_test.json","w"))

all_examples = json.load(open("best_wer_test.json","r"))
all_examples.sort(key = lambda ex:ex[2]/ex[3])
[i for i, row in enumerate(all_examples) if row[2]/row[3] <= 0]