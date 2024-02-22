import os
import math
import yaml
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from peft import get_peft_model, PeftModel

from utils.config_utils import DictConfig, update_config

DEFAULT_CONFIG_FILE = "configs/phoneme_coupler.yaml"

@dataclass
class PhonemeLLMOutput():
    loss:       Optional[torch.FloatTensor]
    logits:     torch.FloatTensor
    labels:     Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor]  = None



""" PhonemeLLM trainer class
"""
class PhonemeLLM(nn.Module):

    def __init__(
            self, 
            llm: PreTrainedModel, 
            coupler_config_or_path: Union[DictConfig,str],
        ):
        
        super().__init__()

        # Assign llm
        self.llm = llm
        self.llm_config = llm.config
        

        # Create newly initialized coupler
        if isinstance(coupler_config_or_path, DictConfig):
            self.coupler_config = update_config(DEFAULT_CONFIG_FILE, coupler_config_or_path)
        else:
            coupler_config = os.path.join(coupler_config_or_path, "coupler_config.yaml")
            self.coupler_config = update_config(DEFAULT_CONFIG_FILE, coupler_config)

        if self.coupler_config.inter_size is not None:
            self.coupler = nn.Sequential(
                nn.Linear(self.coupler_config.input_size, self.coupler_config.inter_size, bias=self.coupler_config.bias),
                ACT2FN[self.coupler_config.act],
                nn.Linear(self.coupler_config.inter_size, self.llm_config.hidden_size, bias=self.coupler_config.bias)
            )
        else:
            self.coupler = nn.Linear(self.coupler_config.input_size, self.llm_config.hidden_size, bias=self.coupler_config.bias)        

        # Load pretrained coupler weights
        if isinstance(coupler_config_or_path, str):
            self.coupler.load_state_dict(torch.load(os.path.join(coupler_config_or_path,"coupler.bin")))

        self.loss_fn = nn.CrossEntropyLoss(reduction=self.coupler_config.loss_reduction)


    """ Prepare embeddings for LLM decoder
    """ 
    def prepare_embeds(
            self, 
            input_ids:          torch.LongTensor,   # (batch, seq_len)
            phoneme_logits:     torch.FloatTensor,  # (batch, seq_len_phon)
            phonemes_start:     torch.LongTensor,   # (batch)
            phonemes_end:       torch.LongTensor,   # (batch)
        ) -> torch.FloatTensor:                     # (batch, seq_len, hidden_size)
        
        # Embed tokens of sentence
        text_embeds = (self.llm.get_input_embeddings())(input_ids)    # (batch, seq_len, hidden_size)
        
        # Embed phoneme logits
        phoneme_embeds = [self.coupler(l) for l in phoneme_logits]  # (batch, seq_len_phon, hidden_size)

        # Substitute phoneme_embeds in input embeds
        return self.sub_embeds(text_embeds, phoneme_embeds, phonemes_start, phonemes_end)


    """ Substitute phoneme embeddings into text embeddings
    """ 
    def sub_embeds(
            self, 
            text_embeds:        torch.FloatTensor,  # (batch, seq_len, hidden_size)
            phoneme_embeds:     torch.FloatTensor,  # (batch, seq_len_phon, hidden_size)
            phonemes_start:     torch.LongTensor,   # (batch)
            phonemes_end:       torch.LongTensor,   # (batch)
        ) -> torch.FloatTensor:                     # (batch, seq_len, hidden_size)

        input_embeds = [
            torch.cat(
                (
                    t[:a], p, t[b:]
                ), dim=0
            )
        for t,p,a,b in zip(text_embeds, phoneme_embeds, phonemes_start, phonemes_end)]

        return torch.stack(input_embeds, dim=0)


    """ Compute Cross Entropy Loss
    """ 
    def loss(
            self,
            logits:     torch.FloatTensor,      # (batch, seq_len, vocab)
            labels:     torch.LongTensor,       # (batch, seq_len)
        ) -> Tuple[torch.FloatTensor, int]:     # ((batch), 1)
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.llm_config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.loss_fn(shift_logits, shift_labels)

        n_examples = (shift_labels != -100).sum().detach().cpu().item()

        return loss, n_examples


    """ Forward pass of the model
    """
    def forward(
            self,
            input_ids:          torch.LongTensor,                   # (batch, seq_len)
            attention_mask:     torch.LongTensor,                   # (batch, seq_len)
            phoneme_logits:     List[torch.FloatTensor],            # batch * [(seq_len_phon, vocab)]
            phonemes_start:     torch.LongTensor,                   # (batch)
            phonemes_end:       torch.LongTensor,                   # (batch)
            labels:             Optional[torch.LongTensor] = None,  # (batch, seq_len)
        ) -> PhonemeLLMOutput:
        
        # Embed logits and merge with text embeddings
        inputs_embeds = self.prepare_embeds(input_ids, phoneme_logits, phonemes_start, phonemes_end)
        
        # Forward LLM
        llm_outputs = self.llm(  
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = llm_outputs.logits   # (batch, seq_len, vocab)

        # Compute loss
        loss = None
        n_examples = None
        if labels is not None:
            loss, n_examples = self.loss(logits, labels)
        
        return PhonemeLLMOutput(
            loss=loss,
            logits=logits,
            labels=labels,
            n_examples=n_examples,
        )


    """ Open ended generation
    """
    def predict(
            self,
            input_ids:          torch.LongTensor,                   # (batch, seq_len)
            attention_mask:     torch.LongTensor,                   # (batch, seq_len)
            phoneme_logits:     List[torch.FloatTensor],            # batch * [(seq_len_phon, vocab)]
            phonemes_start:     torch.LongTensor,                   # (batch)
            phonemes_end:       torch.LongTensor,                   # (batch)
            synced_gpus:        Optional[bool]      = None,
            **gen_config:       DictConfig,
        ) -> List[torch.LongTensor]:  
         
        # Embed logits and merge with text embeddings
        inputs_embeds = self.prepare_embeds(input_ids, phoneme_logits, phonemes_start, phonemes_end)

        # LLM built-in generation
        return self.llm.generate(inputs_embeds=inputs_embeds, **gen_config, synced_gpus=synced_gpus)



    ##  ADAPTER METHODS  ## 
    """ Load trained LoRA adapter for the LLM
    """
    def load_lora_adapter(self, adapter_dir, is_trainable=False):
        self.llm = PeftModel.from_pretrained(self.llm, adapter_dir, is_trainable=is_trainable)

    
    """ Create new LoRA adapter for the LLM
    """
    def create_lora_adapter(self, peft_config):
        self.llm = get_peft_model(self.llm, peft_config)

    """ Merge LoRA weiths with original LLM weights
    """
    def merge_lora_adapter(self):
        self.llm = self.llm.merge_and_unload()

    """ Unload LoRA adapter from LLM
    """
    def unload_lora_adapter(self):
        self.llm = self.llm.unload()


    ##  SAVING METHODS  ##

    """ Save trained LoRA adapter
    """
    def save_lora_adapter(self, adapter_dir):
        if getattr(self.llm, "peft_type", None) is None:
            print("No adapter loaded. Nothing saved.")
            return
            
        if not os.path.exists(adapter_dir):
            os.makedirs(adapter_dir)
        self.llm.save_pretrained(adapter_dir)

    """ Save coupler weights
    """
    def save_coupler(self, coupler_dir):
        yaml.dump(dict(self.coupler_config), open(os.path.join(coupler_dir, "coupler_config.yaml"),"w"), default_flow_style=False)
        torch.save(self.coupler.state_dict(), os.path.join(coupler_dir,"coupler.bin"))
    
    """Save coupler and adapter weights
    """
    def save_checkpoint(self, checkpoint_dir):
        self.save_coupler(checkpoint_dir)
        self.save_lora_adapter(checkpoint_dir)
   
