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
from models.trainer import Trainer, default_trainer_config

kwargs = {"data.test_len": "32", "training.eval_every": "32", "training.train_batch_size": "1"}
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

rl(transformers)
rl(transformers.models)
from transformers.models import *
rl(data_utils)
rl(data_utils.datasets)
rl(data_utils.speechbci_dataset)
from data_utils.speechbci_dataset import *
from data_utils.datasets import *
rl(models)
rl(models.patchtst)
from models.patchtst import *
rl(models)
rl(models.trainer)
from models.trainer import Trainer

kwargs = {"data.test_len": "32", "training.eval_every": "32", "training.train_batch_size": "32"}
config_file = "configs/trainer_ssl.yaml"
config = update_config(default_trainer_config(), config_file)
config = update_config(config, config_from_kwargs(kwargs))
trainer = Trainer(config, dataset=dataset)
trainer.train()

