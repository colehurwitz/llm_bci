from importlib import reload as rl
from utils.config_utils import config_from_kwargs, update_config

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
    "training.num_epochs": "1", "training.train_batch_size": "1", "training.test_batch_size": "1",
    "training.eval_every": "32", "training.save_every": "32", 
    "data.train_len": "32", "data.test_len": "32",
    "model": "include:configs/patchtst.yaml"
}

config_file = "configs/trainer_ssl.yaml"
config = update_config(default_trainer_config(), config_file)
config = update_config(config, config_from_kwargs(kwargs))
import yaml
print(yaml.dump(dict(config), allow_unicode=True, default_flow_style=False))

dataset = load_competition_data(config.data.dataset_dir, **config.data)
if getattr(config["data"], "vocab_file", None) is not None:
    dataset = create_phonemes_ctc_labels(dataset, config.data.vocab_file)


# def metric_1(model, model_inputs, unused_inputs, outputs):
#     return torch.tensor(0.0)

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

kwargs = {
    "training.num_epochs": "1", "training.train_batch_size": "4", "training.test_batch_size": "1",
    "training.eval_every": "32", "training.save_every": "32", 
    "data.train_len": "320", "data.test_len": "320",
    "model": "include:configs/ndt1.yaml"
}

config_file = "configs/trainer_ssl.yaml"
config = update_config(default_trainer_config(), config_file)
config = update_config(config, config_from_kwargs(kwargs))

trainer = Trainer(config, dataset=dataset)
di = iter(trainer.train_dataloader)
ex = next(di)

trainer.train()

