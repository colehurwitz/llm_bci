
from typing import Optional
# from models.itransformer import iTransformer
from models.patchtst import PatchTSTForSpikingActivity
from utils.dataset import load_competition_data
from utils.config_utils import config_from_kwargs, update_config

# STR2MODEL = {"iTransformer": iTransformer, "PatchTST": PatchTSTForSpikingActivity}
STR2MODEL = {"PatchTST": PatchTSTForSpikingActivity}
DEFAULT_TRAINER_CONFIG = "configs/trainer.yaml"

config_file: Optional[str] = None
config = update_config(DEFAULT_TRAINER_CONFIG, config_file) 
# config = update_config(config, config_from_kwargs(kwargs))

model_class = STR2MODEL[config.model.name]
model = model_class(config.model)
dataset = load_competition_data(config.data.data_dir, config.data.dataset_kwargs)