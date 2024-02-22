import os
import wandb 
from tqdm import tqdm

from dataclasses import dataclass
from typing import Optional
from accelerate import Accelerator

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.config_utils import DictConfig, config_from_kwargs



@dataclass
class ModelOutput():
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None



class Trainer():

    def __init__(
        self,
        model,
        dataset,
        config: DictConfig,
        **kwargs,
    ):

        # Set config
        self.config = config
        self.savestring = config.savestring
        
        self.model = model
        self.dataset = dataset

        self.accelerator = Accelerator(
            step_scheduler_with_optimizer=config.optimizer.scheduler in ["linear","cosine"], 
            split_batches=True,
            gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps
        )
        
        self.prepare_logging()
        self.reset_seeds(config.seed)
        self.verbosity = config.verbosity

        self.build_dataloaders()


    """ Create checkpoint dirs, build tensorboard logger and prepare wandb run
    """
    def prepare_logging(self):
        log_dir = os.path.join(self.config.dirs.log_dir,self.config.savestring)
        if not os.path.exists(checkpoint_dir) and self.accelerator.is_main_process:
            os.makedirs(checkpoint_dir)
        if config.log_to_wandb:
            self.wandb_run = wandb.init(self.config.wandb_project)
            self.config = update_config(self.config, config_from_kwargs(wandb.config, convert=False))
        self.writer = SummaryWriter(log_dir=log_dir)

    
    def build_dataloaders(self):

        train_dataset = SpikingDataset(self.dataset[config.trainer.train_name], config.trainer.train_len, config.method.name, **config.method.dataset_kwargs)
        test_dataset = SpikingDataset(self.dataset[config.trainer.test_name], config.trainer.test_len, config.method.name, **config.method.dataset_kwargs)
    

    """ Set seeds for reproducibility
    """
    @staticmethod
    def reset_seeds(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic=True
        # torch.backends.cudnn.benchmark = False

    
    """ Print messages with the appropriate level of verbosity
        0: ALL
        1: TRAINING LOOP
        2: NOTHING
    """
    @staticmethod
    def print_v(*args, verbosity=3):
        if verbosity >= self.verbosity:
            accelerator.print(*args)
        
    
    
        

    def run(self):

        config = self.trainer_config

        self.print_v(f"Starting run {config.savestring}", 0)
        self.print_v(config, 0) 



