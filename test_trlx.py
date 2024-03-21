import json
import os
import sys
from importlib import reload as rl

from typing import List
from functools import partial
from g2p_en import G2p

import torch
from peft import LoraConfig
from peft.utils.config import TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM

import trlx
import trlx.trainer
import trlx.models
import trlx.models.modeling_base
from trlx.pipeline.offline_pipeline import PhonemePromptPipeline
from trlx.utils import set_seed
from trlx.trainer.accelerate_ppo_trainer_embed import AcceleratePPOTrainerEmbed
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

from utils.data_utils import PhonemesFinetuneDataset, ft_pad_collate_fn, prepare_phonemes_data
from utils.config_utils import DictConfig, update_config
from utils.eval_utils import word_error_count
from models.phoneme_llm import PhonemeLLM

from accelerate import Accelerator
accelerator = Accelerator()

config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,            # max seq len (for ILQL, not PPO)
        epochs=10000,   
        total_steps=10000,
        batch_size=2,              
        checkpoint_interval=10000,
        eval_interval=100,
        pipeline="PhonemePromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir="checkpoints/ppo_hh",
    ),
    model=ModelConfig(model_path="/home/gridsan/dbeneto/MAML-Soljacic_shared/llama2/7B-hf", num_layers_unfrozen=-1), #num_layers_unfrozen is ignored when using peft
    tokenizer=TokenizerConfig(tokenizer_path="/home/gridsan/dbeneto/MAML-Soljacic_shared/llama2/tokenizer", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=8,
        chunk_size=2,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        cliprange_reward=10.0,
        vf_coef=1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        # cliprange_reward=10,      # clips reward i make_experience
        gen_kwargs=dict(
            max_new_tokens=20,
            temperature=1.0,        # rpobably want to ramp up the temperature to produce diverse samples
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)


# config.model.peft_config = LoraConfig(
#         r=8,
#         task_type=TaskType.CAUSAL_LM,
#         lora_alpha=32,
#         lora_dropout=0.1,
#     ) 

checkpoint_path = "/home/gridsan/dbeneto/TFG/BCI/checkpoints/old_ft/phonemes-rank_1-lr_1.e-4-gauss_0.0-spikes_0.7_2_0.9-norm_identity_1/EP12-STEP1849"
config.model.model_path = checkpoint_path
config.train.batch_size = 2
config.train.total_steps = 6000
config.method.chunk_size = 2

hparams = {}    # Use this to update config for sweeping
# Merge sweep config with default config if given
config = TRLConfig.update(config.to_dict(), hparams)

def reward_fn(samples: List[str], **kwargs):
    rewards = []
    preds = kwargs["outputs"]
    sentences = kwargs["sentences"]
    for pred, sentence in zip(preds, sentences):
        errors, n_words = word_error_count(pred, sentence)
        rewards.append(1 - errors/n_words)
    return rewards


ft_config = DictConfig(torch.load(os.path.join(checkpoint_path, "config.pth")))

tokenizer = AutoTokenizer.from_pretrained(ft_config.dirs.tokenizer_dir, padding_side='left', add_bos_token=False, add_eos_token=False)
pad_id = tokenizer.eos_token_id
g2p = G2p()

ft_config["trainer"]["test_len"] = 64
data = torch.load("/home/gridsan/dbeneto/MAML-Soljacic_shared/BCI/data/competitionData/phonemes_data.pth")
train_data = {k: v[:ft_config.trainer.train_len] if ft_config.trainer.train_len != -1 else v for k,v in data["train"].items()}
train_data = prepare_phonemes_data(train_data, tokenizer, g2p, "phonemes: %% sentence:")
test_data = {k: v[:ft_config.trainer.test_len] if ft_config.trainer.test_len != -1 else v for k,v in data["test"].items()}
test_data = prepare_phonemes_data(test_data, tokenizer, g2p, "phonemes: %% sentence:")

train_dataset = PhonemesFinetuneDataset(train_data)
test_dataset = PhonemesFinetuneDataset(test_data)

prompts = [ex for ex in train_dataset]
eval_prompts = [ex for ex in test_dataset]
collate_fn = partial(ft_pad_collate_fn,ft_config.noise,ft_config.mask,pad_id,"train")
collate_fn_eval = partial(ft_pad_collate_fn,ft_config.noise,ft_config.mask,pad_id,"test")


llm = AutoModelForCausalLM.from_pretrained(checkpoint_path)
model = PhonemeLLM(llm, checkpoint_path)
adapter_file = os.path.join(checkpoint_path, "adapter_config.json")
if os.path.isfile(adapter_file):
    model.load_lora_adapter(checkpoint_path, is_trainable=True)

model.to("cuda")
# model.merge_lora_adapter()
config.model.model_path = model.llm




rl(trlx)
rl(trlx.trainer)
rl(trlx.trainer.accelerate_ppo_trainer_embed)
from trlx.trainer.accelerate_ppo_trainer_embed import *

set_seed(config.train.seed)
_trainer = AcceleratePPOTrainerEmbed(
    config=config,
    reward_fn=reward_fn,
    metric_fn=None,
    extra_opt_params=model.coupler.parameters(),
    **config.train.trainer_kwargs,
)

pipeline = PhonemePromptPipeline(
    prompts, collate_fn, model.prepare_embeds,
)
_trainer.add_prompt_pipeline(pipeline)

eval_pipeline = PhonemePromptPipeline(
    eval_prompts, collate_fn_eval, model.prepare_embeds,
)
_trainer.add_eval_pipeline(eval_pipeline)

batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
max_prompt_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

if config.train.resume_from_checkpoint and os.path.exists(config.train.resume_from_checkpoint):
    _trainer.load(config.train.resume_from_checkpoint)

_trainer.learn()




batch = torch.load("batch.pth")


from copy import deepcopy
prev_model = deepcopy(_trainer.model)
for ((pn, p), (bn, b)) in zip(prev_model.named_parameters(), _trainer.model.named_parameters()):
    if not torch.allclose(p,b):
        print(pn, bn)

for pn, p in _trainer.model.named_parameters():
    if p.requires_grad:
        print(pn)



# from torch.utils.data import DataLoader
# train_dataloader = DataLoader(
#     train_dataset, shuffle=True, collate_fn=partial(ft_pad_collate_fn,ft_config.noise,ft_config.mask,pad_id,"test"), batch_size=1, pin_memory=True,
# )
# test_dataloader = DataLoader(
#     test_dataset, collate_fn=partial(ft_pad_collate_fn,ft_config.noise,ft_config.mask,pad_id,"test"), batch_size=1, pin_memory=True,
# )

# train_iter = iter(train_dataloader)
# test_iter = iter(test_dataloader)

# rl(trlx)
# rl(trlx.models)
# rl(trlx.models.modeling_base)
# import trlx
# from trlx.models import *
# from trlx.models.modeling_base import *