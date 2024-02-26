from importlib import reload as rl
from utils.config_utils import config_from_kwargs, update_config

import data_utils
import data_utils.datasets
from data_utils.speechbci_dataset import load_competition_data, create_phonemes_ctc_labels

import models
import models.trainer
import models.patchtst
from models.trainer import Trainer, default_trainer_config


config_file = "configs/trainer_ctc.yaml"
config = update_config(default_trainer_config(), config_file)
# config = update_config(config, config_from_kwargs(kwargs))
import yaml
print(yaml.dump(dict(config), allow_unicode=True, default_flow_style=False))

dataset = load_competition_data(config.data.dataset_dir, config.data.zscore)
if getattr(config.data, "vocab_file", None) is not None:
    dataset = create_phonemes_ctc_labels(dataset, config.data.vocab_file)


def metric_1(model, model_inputs, unused_inputs, outputs):
    return 0


rl(data_utils)
rl(data_utils.datasets)
from data_utils.datasets import *
rl(models)
rl(models.patchtst)
from models.patchtst import *
rl(models)
rl(models.trainer)
from models.trainer import Trainer
trainer = Trainer(config, dataset=dataset, metric_fns = {"metric-1": metric_1})
trainer.train()

