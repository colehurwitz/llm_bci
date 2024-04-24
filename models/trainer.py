import os
import wandb 
import yaml
import json
import inspect
from functools import partial
from tqdm import tqdm


from typing import Optional, Union, Dict, List, Any, Callable

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from transformers import get_linear_schedule_with_warmup


from accelerate import Accelerator
from datasets import load_dataset

from utils.config_utils import DictConfig, config_from_kwargs, update_config
from data_utils.datasets import SpikingDataset, SpikingDatasetForDecoding, DaySpecificSpikingDatasetForDecoding, pad_collate_fn
from models.patchtst import PatchTSTForSpikingActivity
from models.itransformer import iTransformer
from models.ndt1 import NDT1
from models.bci import BCI

""" Mapping from dataset class names to dataset class. New dataset classes should be registered here
"""
NAME2DATASET = {"base": SpikingDataset, "decoding": SpikingDatasetForDecoding, "day": DaySpecificSpikingDatasetForDecoding}

""" Mapping from model class names to model class. New model classes should be registered here
"""
NAME2MODEL = {"PatchTST": PatchTSTForSpikingActivity, "NDT1": NDT1, "iTransformer": iTransformer, "BCI": BCI}

""" Base configuration for the Trainer. 
"""
DEFAULT_TRAINER_CONFIG = "configs/trainer.yaml"
def default_trainer_config():
    return update_config(DEFAULT_TRAINER_CONFIG, None)



""" Trainer class. 
    INPUTS
        config: configuration object. Updates DEFAULT_TRAINER_CONFIG
        model: can be a ``nn.Module``. If it is not provided, it will be loaded form the configuration
        dataset: can be a file path, hf dataset name or a ``dict`` with split keys. Each split is a 
        ``list`` of examples, where each example is a ``dict``. If it is not provided, it will be 
        loaded from the configuration.
        metric_fns: ``dict`` of metric functions to be used during training and evaluation. Metric 
        functions receive as input the model, model inputs, unused inputs and model outputs, and should
        return a ``torch.Tensor`` which will be averaged across all devices.
        eval_metric_fns: ``dict`` of additional functions to be used during evaluation
        extra_model_kwargs: ``dict`` of additional kwargs passed to model
"""
class Trainer():

    def __init__(
        self,
        config:             DictConfig,
        model:              Optional[nn.Module]         = None,
        dataset:            Optional[Union[str,Dict[str,List[Dict[str,Any]]]]]   = None,
        metric_fns:         Optional[Dict[str,Callable]]    = None,
        eval_metric_fns:    Optional[Dict[str,Callable]]    = None,
        extra_model_kwargs: Optional[Dict[str,Any]]         = None,
    ):  
        self.config = update_config(default_trainer_config(), config) 
        
        self.verbosity = config.verbosity
        self.init_wandb()
        self.reset_seeds()

        # Used for distributed training
        self.accelerator = Accelerator(
            step_scheduler_with_optimizer=config.optimizer.scheduler in ["linear","cosine"], 
            split_batches=True,
        )
        
        self.print_v(yaml.dump(dict(self.config), allow_unicode=True, default_flow_style=False), verbosity=0) 

        self.prepare_logging()

        self.set_model(model, extra_model_kwargs)
        self.get_model_inputs()     # Used by the dataloader to provide only the keys used by the model

        self.set_dataset(dataset)
        self.build_dataloaders()        

        self.build_optimizer_and_scheduler()

        self.prepare_for_distributed_training()
        
        self.metric_kwargs = self.config.method.metric_kwargs
        self.metric_fns = metric_fns if metric_fns else {}
        self.eval_metric_fns = eval_metric_fns if eval_metric_fns else {}
        

    """ Print messages with the appropriate level of verbosity
        0: ALL
        1: TRAINING LOOP
        2: NOTHING
    """
    def print_v(self, *args, verbosity=3):
        if verbosity >= self.verbosity:
            self.accelerator.print(*args)


    """ Initialize wandb run. Update config with wandb configration for hyperparameter sweeps.
    """
    def init_wandb(self):
        if self.config.log_to_wandb:
            self.wandb_run = wandb.init(self.config.wandb_project)
            self.config = update_config(self.config, config_from_kwargs(wandb.config, convert=False))
        

    """ Set seeds for reproducibility
    """
    def reset_seeds(self):
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        # torch.backends.cudnn.deterministic=True
        # torch.backends.cudnn.benchmark = False


    """ Create checkpoint dirs, build tensorboard logger.
    """
    def prepare_logging(self):
        self.savestring = self.config.savestring
        self.checkpoint_dir = os.path.join(self.config.dirs.checkpoint_dir,self.savestring)
        if not os.path.exists(self.checkpoint_dir) and self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir)
        
        log_dir = os.path.join(self.config.dirs.log_dir,self.config.savestring)
        self.writer = SummaryWriter(log_dir=log_dir)

   
    """ Set or build model.
        INPUTS
            model: Can be a nn.Module. If it is None, it will be loaded from the configuration.
    """
    def set_model(self, model, extra_model_kwargs=None):
        if extra_model_kwargs is None:
            extra_model_kwargs = {}
        if model is None:
            # Load model from configuration
            model_class = NAME2MODEL[self.config.model.model_class]    # Get class from name
            self.model = model_class(self.config.model, **self.config.method.model_kwargs, **extra_model_kwargs)
        else:
            self.model = model

        self.print_v(self.model)
        self.print_v(f"Model number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}", verbosity=0)
        if getattr(self.model, "llm", None) is not None and getattr(self.model.llm, "print_trainable_parameters", None) is not None and self.accelerator.is_main_process and 0 >= self.verbosity:
            self.model.llm.print_trainable_parameters()

    """ Get the used columns of the dataset
    """
    def get_model_inputs(self):

        model_to_inspect = self.model
        # Access base model in case of a loaded peft adapter
        if getattr(self.model, "peft_type", None) is not None:
            if hasattr(self.model, "get_base_model"):
                model_to_inspect = self.model.get_base_model()
            else:
                model_to_inspect = self.model.base_model.model
        signature = inspect.signature(model_to_inspect.forward)
        self.model_inputs = list(signature.parameters.keys())
        

    """ Load dataset.
        INPUTS: 
            dataset: Can be a dataset object, the path to a json file or the name of a huggingface 
            dataset. If it is None, it will be loaded from configuration.
    """
    def set_dataset(self, dataset):
        if dataset is None:
            # Load dataset from configuration
            if self.config.data.hf_dataset_name:
                self.print_v(f"Loading hf dataset {self.config.hf_dataset_name}")
                self.dataset = load_dataset(self.config.data.hf_dataset_name)
            elif self.config.data.json_dataset_name:
                self.print_v(f"Loading dataset from json file {self.config.json_dataset_name}")
                self.dataset = json.load(open(self.config.data.json_dataset_name,"r"))
            else:
                raise Exception("No dataset provided")
        elif isinstance(dataset, str):
            # Load dataset from file or hf name
            try:
                self.dataset = load_dataset(dataset)
                self.print_v(f"Loading hf dataset {dataset}")
            except:
                try:
                    self.dataset = json.load(open(dataset,"r"))
                    self.print_v(f"Loading dataset from json file {dataset}")
                except:
                    raise Exception("Can't load dataset from provided path or name")
        else:
            self.dataset = dataset



    """ Create train and test dataloaders
    """
    def build_dataloaders(self):
        self.print_v("Building dataloaders", verbosity=0)
        # Get name of Dataset class
        dataset_class = NAME2DATASET[self.config.data.dataset_class]
        self.train_dataset = dataset_class(self.dataset[self.config.data.train_name], length=self.config.data.train_len, **self.config.method.dataset_kwargs)
        self.test_dataset = dataset_class(self.dataset[self.config.data.test_name], length=self.config.data.test_len, **self.config.method.dataset_kwargs)

        # ToDo Custom DataLoaders?
        self.train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, collate_fn=partial(pad_collate_fn, model_inputs=self.model_inputs, **self.config.method.dataloader_kwargs), batch_size=self.config.training.train_batch_size, pin_memory=True, drop_last=self.config.training.drop_last_train_dataloader,
        )

        self.test_dataloader = DataLoader(
            self.test_dataset, shuffle=self.config.training.shuffle_test_dataloader, collate_fn=partial(pad_collate_fn, model_inputs=self.model_inputs, **self.config.method.dataloader_kwargs), batch_size=self.config.training.test_batch_size, pin_memory=True, drop_last=self.config.training.drop_last_test_dataloader,
        )


    """ Create optimzier and learning rate scheduler
    """
    def build_optimizer_and_scheduler(self):
        self.print_v("Building optimizers", verbosity=0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.optimizer.lr, weight_decay=self.config.optimizer.wd, eps=self.config.optimizer.eps)
        # for pn, p in model.named_parameters():
        #     print(pn, p.requires_grad)

        if self.config.optimizer.scheduler == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=round(self.config.optimizer.warmup_pct*self.config.training.num_epochs*len(self.train_dataloader)//self.config.optimizer.gradient_accumulation_steps),
                num_training_steps=self.config.training.num_epochs*len(self.train_dataloader)//self.config.optimizer.gradient_accumulation_steps,
            )
        elif self.config.optimizer.scheduler == "cosine":
            self.lr_scheduler = OneCycleLR(
                optimizer=self.optimizer,
                total_steps=self.config.training.num_epochs*len(self.train_dataloader)//self.config.optimizer.gradient_accumulation_steps,
                max_lr=self.config.optimizer.lr,
                pct_start=self.config.optimizer.warmup_pct,
                div_factor=self.config.optimizer.div_factor,
            )
        elif self.config.optimizer.scheduler == "step":
            self.lr_scheduler = StepLR(
                self.optimizer, 
                step_size=1, 
                gamma=self.config.optimizer.gamma)
        else:
            raise Exception(f"Scheduler '{self.config.optimizer.scheduler}' not implemented")


    """ Accelerate method for distributed training
    """
    def prepare_for_distributed_training(self):
        self.print_v("Preparing for distributed training", verbosity=0)
        self.model, self.train_dataloader, self.test_dataloader, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
        self.model, self.train_dataloader, self.test_dataloader, self.optimizer, self.lr_scheduler
    )


    """ Evaluate the model with possible additional metric functions. 
        INPUTS:
            additional_metric_fns: dict of metric functions to use for evaluation apart from 
            the training metric functions
            eval_train_set: if set to True, the evaluation is performed over the train dataset
        OUTPUTS:
            Averaged test loss and averaged metrics.
    """
    def evaluate(
        self,
        additional_metric_fns: Optional[Dict[str,Callable]] = None,
        eval_train_set: Optional[bool] = False,
    ):
        metric_fns = dict(**self.metric_fns)
        metric_fns.update(additional_metric_fns if additional_metric_fns else {})
        

        # Test metrics
        test_loss = []
        test_examples = []
        test_metrics = {name: [] for name in metric_fns.keys()}
        
        self.model.eval()
        
        dataloader = self.test_dataloader if not eval_train_set else self.train_dataloader
        for test_step, (model_inputs, unused_inputs) in enumerate(tqdm(dataloader) if self.verbosity <= 1 else self.test_dataloader):
            
            with torch.no_grad() as A, self.accelerator.no_sync(self.model) as B:
                outputs = self.model(**model_inputs)
                loss = outputs.loss
                examples = outputs.n_examples

                 # Loss
                test_loss.append(self.accelerator.gather(loss).sum().detach().item())
                test_examples.append(self.accelerator.gather(examples).sum().detach().item())

                # Metrics
                for name, fn in metric_fns.items():
                    test_metrics[name].append(self.accelerator.gather(fn(self.model, model_inputs, unused_inputs, outputs.to_dict(), **self.metric_kwargs)).sum().detach().item())
            

        test_avg_loss = sum(test_loss) / sum(test_examples) if sum(test_examples) > 0 else 0
        test_avg_metrics = {k: sum(v)/len(v) for k, v in test_metrics.items()}

        return test_avg_loss, test_avg_metrics


    """ Training loop
    """
    def train(self):
        
        config = self.config

        self.print_v(f"Starting run {config.savestring}", verbosity=0)
    
        # Train
        global_step = 1
        
        # Train metrics
        train_loss = []
        train_examples = []
        train_metrics = {name: [] for name in self.metric_fns.keys()}

        for epoch in range(1, config.training.num_epochs+1):
            self.print_v(f"Epoch {epoch}", verbosity=1)
            self.model.train()

            for step, (model_inputs, unused_inputs) in enumerate(tqdm(self.train_dataloader) if self.verbosity <= 1 else self.train_dataloader):

                # Perform gradient accumulation
                if (global_step-1) % config.optimizer.gradient_accumulation_steps == 0:
                    outputs = self.model(**model_inputs)
                    loss = outputs.loss
                    examples = outputs.n_examples
                    self.accelerator.backward(loss / config.optimizer.gradient_accumulation_steps)
                    self.optimizer.step()
                    if config.optimizer.scheduler in ["linear","cosine"]:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                else:
                    with self.accelerator.no_sync(self.model):
                        outputs = self.model(**model_inputs)
                        loss = outputs.loss
                        examples = outputs.n_examples
                        self.accelerator.backward(loss / config.optimizer.gradient_accumulation_steps)

                
                # Loss
                train_loss.append(self.accelerator.gather(loss).sum().detach().item())
                train_examples.append(self.accelerator.gather(examples).sum().detach().item())
                if self.accelerator.is_main_process:
                    self.writer.add_scalar("Loss/train_iter",train_loss[-1] / train_examples[-1], global_step)

                # Metrics
                for name, fn in self.metric_fns.items():
                    train_metrics[name].append(self.accelerator.gather(fn(self.model, model_inputs, unused_inputs, outputs.to_dict(), **self.metric_kwargs)).sum().detach().item())
                    if self.accelerator.is_main_process:
                        self.writer.add_scalar(f"{name}/train_iter",train_metrics[name][-1], global_step)


                # Evaluation condition
                if config.training.eval_every and global_step % config.training.eval_every == 0:

                    self.print_v(f"Evaluation at step {global_step}", verbosity=1)
                    test_avg_loss, test_avg_metrics = self.evaluate(self.eval_metric_fns)
                    train_avg_loss = sum(train_loss) / sum(train_examples) if sum(train_examples) > 0 else 0
                    train_avg_metrics = {k: sum(v)/len(v) for k, v in train_metrics.items()}
                
                    self.print_v(f"{self.savestring=} {global_step=}:" + "\n" + \
                                  f"{train_avg_loss=} {train_avg_metrics=}" + "\n" + \
                                  f"{test_avg_loss=} {test_avg_metrics=}", verbosity=1)  

                    # Log to tensorboard/wandb
                    if self.accelerator.is_main_process:
                        self.writer.add_scalar("Loss/train",train_avg_loss,global_step)
                        for name, v in train_avg_metrics.items():
                            self.writer.add_scalar(f"{name}/train", v, global_step)
                        self.writer.add_scalar("Loss/test",test_avg_loss,global_step)
                        for name, v in test_avg_metrics.items():
                            self.writer.add_scalar(f"{name}/test", v, global_step)

                    # Log to wandb
                    if config.log_to_wandb:
                        wandb.log({
                            "step": global_step,
                            "train_avg_loss": train_avg_loss,
                            **train_avg_metrics,
                            "test_avg_loss": test_avg_loss,
                            **test_avg_metrics,
                        })

                    # Reset train metrics
                    train_loss = []
                    train_examples = []
                    train_metrics = {name: [] for name in self.metric_fns.keys()}

                    # End evaluation
                    self.model.train()     

                # Save checkpoints
                if config.training.save_every and global_step % config.training.save_every == 0:
                    save_to_path = os.path.join(self.checkpoint_dir,f"STEP{global_step}")
                    if not os.path.exists(save_to_path) and self.accelerator.is_main_process:
                        os.makedirs(save_to_path)

                    self.print_v(f"Saving checkpoint at step {global_step} to {save_to_path}", verbosity=1)
                    self.model.save_checkpoint(save_to_path)
                    if self.accelerator.is_main_process:
                        torch.save(dict(config), os.path.join(save_to_path,"trainer_config.pth"))
                        
                # Track step
                global_step += 1    

            if config.optimizer.scheduler in ["step"]:
                self.lr_scheduler.step()

        self.writer.flush()
        self.writer.close()

        self.print_v("Training done", verbosity=1)


                    


