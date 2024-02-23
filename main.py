
from typing import Optional
from dataset.speechbci_dataset import load_competition_data, create_phonemes_ctc_labels
from utils.config_utils import config_from_kwargs, update_config
from models.trainer import Trainer

from importlib import reload as rl


DEFAULT_TRAINER_CONFIG = "configs/trainer.yaml"

config_file: Optional[str] = None
config = update_config(DEFAULT_TRAINER_CONFIG, config_file) 
# config = update_config(config, config_from_kwargs(kwargs))

dataset = load_competition_data(config.data.data_dir, config.data.zscore)
if config.method.model_kwargs.method_name == "ctc":
    dataset = create_phonemes_ctc_labels(dataset, config.data.vocab_file)

def metric_1(model, model_inputs, unused_inputs, outputs):
    return 0

trainer = Trainer(config, dataset=dataset, metric_fns=[metric_1])