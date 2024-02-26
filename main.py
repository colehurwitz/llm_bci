import yaml
from utils.config_utils import config_from_kwargs, update_config
import data_utils.datasets
from data_utils.speechbci_dataset import load_competition_data
from models.trainer import Trainer, default_trainer_config


config_file = "configs/trainer_ssl.yaml"
config = update_config(default_trainer_config(), config_file)
# config = update_config(config, config_from_kwargs(kwargs))
#print(yaml.dump(dict(config), allow_unicode=True, default_flow_style=False))

dataset = load_competition_data(config.data.dataset_dir)

# Test metric
def metric_1(model, model_inputs, unused_inputs, outputs):
    return torch.tensor(0.0)

trainer = Trainer(config, dataset=dataset, metric_fns = {"metric-1": metric_1})
trainer.train()

