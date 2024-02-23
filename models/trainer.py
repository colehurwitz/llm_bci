import os
import wandb 
import json
import inspect
from tqdm import tqdm

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, StepLR

from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from accelerate import Accelerator

from utils.config_utils import DictConfig, config_from_kwargs
from dataset.dataset import SpikingDataset, SpikingDatasetForCTC, pad_collate_fn


@dataclass
class ModelOutput():
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}


METHOD2DATASET = {"ssl": SpikingDataset, "ctc": SpikingDatasetForCTC}
# NAME2MODEL = {"iTransformer": iTransformer, "PatchTST": PatchTSTForSpikingActivity}
NAME2MODEL = {"PatchTST": PatchTSTForSpikingActivity}

class Trainer():

    def __init__(
        self,
        config:             DictConfig,
        model:              Optional[nn.Module]         = None,
        dataset:            Optional[Union[str,Dict[List[Any]]]]   = None,
        metric_fns:         Optional[Dict[Callable]]    = None,
        eval_metric_fns:    Optional[Dict[Callable]]    = None,
    ):  
        self.config = config
        self.reset_seeds(config.seed)
        self.verbosity = config.verbosity
        
        self.accelerator = Accelerator(
            step_scheduler_with_optimizer=config.optimizer.scheduler in ["linear","cosine"], 
            split_batches=True,
            gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps
        )

        self.prepare_logging()

        self.set_model(model)
        self.get_model_inputs()

        self.set_dataset(dataset)
        self.build_dataloaders()        

        self.build_optimizer_and_scheduler()

        self.prepare_for_distributed_training()
        
        self.metric_fns = metric_fns if metric_fns else {}
        self.eval_metric_fns = eval_metric_fns if eval_metric_fns else {}
        

    """ Print messages with the appropriate level of verbosity
        0: ALL
        1: TRAINING LOOP
        2: NOTHING
    """
    @staticmethod
    def print_v(*args, verbosity=3):
        if verbosity >= self.verbosity:
            accelerator.print(*args)


    """ Set seeds for reproducibility
    """
    @staticmethod
    def reset_seeds(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic=True
        # torch.backends.cudnn.benchmark = False


    """ Load dataset from configuration. It can be a dataset object, the path to a json file
        or the name of a huggingface dataset
    """
    def set_dataset(self, dataset):
        if dataset is None:
            # Load dataset from configuration
            if self.config.dataset.hf_dataset_name:
                self.print_v(f"Loading hf dataset {self.config.hf_dataset_name}")
                self.dataset = load_dataset(self.config.dataset.hf_dataset_name)
            elif self.config.dataset.json_dataset_name:
                self.print_v(f"Loading dataset from json file {self.config.json_dataset_name}")
                self.dataset = json.load(open(self.config.dataset.json_dataset_name,"r"))
            else:
                raise Exception("No dataset provided")
        elif isinstance(dataset, str):
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

   
    """ Set or build model. Model can be a nn.Module or None. In the second case it will be loaded from the configuration.
    """
    def set_model(self, model):
        if isinstance(model, nn.Module):
            self.model = model
        else:
            # Load model from configuration
            model_class = NAME2MODEL[self.config.model.name]    # Get class from name
            self.model = model_class(self.config.model, **self.config.method.model_kwargs)


    """ Get the used columns of the dataset
    """
    def get_model_inputs(self):

        model_to_inspect = self.model
        if getattr(self.model, "peft_type", None) is not None:
            if hasattr(self.model, "get_base_model"):
                model_to_inspect = self.model.get_base_model()
            else:
                model_to_inspect = self.model.base_model.model
        signature = inspect.signature(model_to_inspect.forward)
        self.model_inputs = list(signature.parameters.keys())
        

    """ Create checkpoint dirs, build tensorboard logger and prepare wandb run
    """
    def prepare_logging(self):
        self.savestring = config.savestring
        self.checkpoint_dir = os.path.join(self.config.dirs.checkpoint_dir,self.savestring)
        if not os.path.exists(self.checkpoint_dir) and self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir)
        
        log_dir = os.path.join(self.config.dirs.log_dir,self.config.savestring)
        self.writer = SummaryWriter(log_dir=log_dir)

        if self.config.log_to_wandb:
            self.wandb_run = wandb.init(self.config.wandb_project)
            self.config = update_config(self.config, config_from_kwargs(wandb.config, convert=False))
        

    """ Create train and test dataloaders
    """
    def build_dataloaders(self):
        self.print_v("Building dataloaders", verbosity=0)
        # Get name of Dataset class
        dataset_class = METHOD2DATASET[config.method.dataset_kwargs.dataset_name]
        train_dataset = dataset_class(self.dataset[config.trainer.train_name], config.trainer.train_len, config.method.name, **config.method.dataset_kwargs)
        test_dataset = dataset_class(self.dataset[config.trainer.test_name], config.trainer.test_len, config.method.name, **config.method.dataset_kwargs)

        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=partial(pad_collate_fn, model_inputs=self.model_inputs, **config.method.dataloader_kwargs), batch_size=config.trainer.train_batch_size, pin_memory=True, drop_last=True,
        )


        self.test_dataloader = DataLoader(test_dataset)#, shuffle=config.trainer.shuffle_test_dataloader, collate_fn=collate_fn=partial(pad_collate_fn, model_inputs=self.model_inputs, **config.method.dataloader_kwargs))#, batch_size=config.trainer.test_batch_size, pin_memory=True, drop_last=True)


    """ Create optimzier and learning rate scheduler
    """
    def build_optimizer_and_scheduler(self):
        self.print_v("Building optimizers", verbosity=0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)
        # for pn, p in model.named_parameters():
        #     print(pn, p.requires_grad)

        if config.optimizer.scheduler == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=round(config.optimizer.warmup_pct*config.trainer.num_epochs*len(train_dataloader)//config.optimizer.gradient_accumulation_steps),
                num_training_steps=config.trainer.num_epochs*len(train_dataloader)//config.optimizer.gradient_accumulation_steps,
            )
        elif config.optimizer.scheduler == "cosine":
            self.lr_scheduler = OneCycleLR(
                optimizer=self.optimizer,
                total_steps=config.trainer.num_epochs*len(train_dataloader)//config.optimizer.gradient_accumulation_steps,
                max_lr=config.optimizer.lr,
                pct_start=config.optimizer.warmup_pct,
                div_factor=config.optimizer.div_factor,
            )
        elif config.optimizer.scheduler == "step":
            self.lr_scheduler = StepLR(
                self.optimizer, 
                step_size=1, 
                gamma=config.optimizer.gamma)
        else:
            raise Exception(f"Scheduler '{config.optimizer.scheduler}' not implemented")


    """ Accelerate method for distributed training
    """
    def prepare_for_distributed_training(self):
        self.print_v("Preparing for distributed training", verbosity=0)
        self.model, self.train_dataloader, self.test_dataloader, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
        self.model, self.train_dataloader, self.test_dataloader, self.optimizer, self.lr_scheduler
    )


    """ Evaluate the model with possible additional metric functions. Returns average test
    loss and averaged metrics.
    """
    def evaluate(
        self,
        additional_metric_fns: Optional[Dict[Callable]] = None,
    ):
        metric_fns = additional_metric_fns if additional_metric_fns else {}
        metric_fns.update(self.metric_fns)

        # Test metrics
        test_loss = []
        test_examples = []
        test_metrics = {name: [] for name in metric_fns.keys()}
        
        model.eval()

        for test_step, (model_inputs, unused_inputs) in enumerate(tqdm(self.test_dataloader) if verbosity <= 1 else self.test_dataloader):
            
            with torch.no_grad() as A, self.accelerator.no_sync(model) as B:
                outputs = self.model(**model_inputs)
                loss = outputs.loss
                examples = outputs.n_examples

     
            # Loss
            test_loss.append(self.accelerator.gather(loss).sum().detach().item())
            test_examples.append(self.accelerator.gather(examples).sum().detach().item())

            # Metrics
            for name, fn in metric_fns.items():
                test_metrics[name].append(self.accelerator.gather(self.model, model_inputs, unused_inputs, outputs.to_dict()).sum().detach().item())


        test_avg_loss = sum(test_loss) / sum(test_examples)
        test_avg_metrics = {k: sum(v)/len(v) for k, v in test_metrics.items()}

        return test_avg_loss, test_avg_metrics


    """ Training loop 
    """
    def train(self):
        
        config = self.config

        self.print_v(f"Starting run {config.savestring} with config: ", verbosity=0)
        self.print_v(self.trainer_config, verbosity=0) 

        # Train
        global_step = 0
        
        # Train metrics
        train_loss = []
        train_examples = []
        train_metrics = {name: [] for name in self.metric_fns.keys()}

        for epoch in range(1, config.trainer.num_epochs+1):
            self.print_v(f"Epoch {epoch}", verbosity=1)
            self.model.train()

            for step, (model_inputs, unused_inputs) in enumerate(tqdm(self.train_dataloader) if verbosity <= 1 else self.train_dataloader):

                # Perform gradient accumulation
                if (global_step + 1) % config.optimizer.gradient_accumulation_steps == 0:
                    outputs = self.model(**model_inputs)
                    loss = outputs.loss
                    examples = outputs.n_examples
                    self.accelerator.backward(loss / config.optimizer.gradient_accumulation_steps)
                    self.optimizer.step()
                    if config.optimizer.scheduler in ["linear","cosine"]:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                else:
                    with self.accelerator.no_sync(model):
                        outputs = self.model(**model_inputs)
                        loss = outputs.loss
                        examples = outputs.n_examples
                        self.accelerator.backward(loss / config.optimizer.gradient_accumulation_steps)

                
                # Loss
                train_loss.append(self.accelerator.gather(loss).sum().detach().item())
                train_examples.append(self.accelerator.gather(examples).sum().detach().item())
                if accelerator.is_main_process:
                    self.writer.add_scalar("Loss/train_iter",train_loss[-1] / train_examples[-1], global_step)

                # Metrics
                for name, fn in self.metric_fns.items():
                    train_metrics[name].append(self.accelerator.gather(fn(self.model, model_inputs, unused_inputs, outputs.to_dict())).sum().detach().item())
                    if accelerator.is_main_process:
                        self.writer.add_scalar(f"{name}/train_iter",train_metrics[name][-1], global_step)


                # Evaluation condition
                if (global_step + 1) % config.trainer.eval_every == 0:

                    self.print_v(f"Evaluation at step {global_step}", verbosity=1)
                    train_avg_loss, train_avg_metrics = self.evaluate(self.eval_metric_fns)
                    train_avg_loss = sum(train_loss) / sum(train_examples)
                    train_avg_metrics = {k: sum(v)/len(v) for k, v in train_metrics.items()}
                
                    self.print_v(f"{self.savestring=} {global_step=}:" + "\n" + \
                                  f"{train_avg_loss=} {train_avg_metrics=}" + "\n" + \
                                  f"{test_avg_loss=} {test_avg_metrics=}", verbosity=1)  

                    # Log to tensorboard/wandb
                    if accelerator.is_main_process:
                        self.writer.add_scalar("Loss/train",train_avg_loss,global_step)
                        for k, v in train_avg_metrics:
                            self.writer.add_scalar(f"{name}/train", v, global_step)
                        self.writer.add_scalar("Loss/test",test_epoch_loss,global_step)
                        for k, v in test_avg_metircs:
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
                    model.train()     

                # Save checkpoints
                if (global_step + 1) % config.trainer.save_every == 0:
                    save_to_path = os.path.join(self.checkpoint_dir,f"STEP{global_step}")
                    if not os.path.exists(save_to_path) and accelerator.is_main_process:
                        os.makedirs(save_to_path)

                    self.print_v(f"Saving checkpoint at step {global_step} to {save_to_path}", verbosity=1)
                    self.model.save_checkpoint(save_to_path)
                    if accelerator.is_main_process:
                        torch.save(dict(config), os.path.join(save_to_path,"trainer_config.pth"))
                        
            
            global_step += 1    

            if config.optimizer.scheduler in ["step"]:
                self.lr_scheduler.step()

        self.writer.flush()
        self.writer.close()

        self.print_v("Training done", verbosity=1)


                    


