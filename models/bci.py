import os
import math
import yaml
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, LlamaConfig
from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from models.ndt1 import NDT1
from models.model_output import ModelOutput

from utils.config_utils import DictConfig, update_config

DEFAULT_CONFIG = "configs/bci.yaml"

@dataclass
class BCIOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    mask: Optional[torch.LongTensor] = None
    preds: Optional[torch.FloatTensor] = None
    targets: Optional[torch.FloatTensor] = None


class BCI(nn.Module):

    def __init__(
        self,
        config: DictConfig, 
        llm_path: str,
        lora: Optional[Dict] = None,
        freeze_llm: Optional[bool] = False,
        **kwargs,
    ):

        super().__init__()

        config = update_config(DEFAULT_CONFIG, config)
        
        pt_path = dict(config).pop("from_pt", None)

        if "llm" in kwargs:
            llm = kwargs["llm"]
        else:
            if "debug" in kwargs and kwargs["debug"]:
                llm_config = LlamaConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=32,  num_attention_heads=4)
                llm = AutoModelForCausalLM.from_config(llm_config)
            else:
                llm = AutoModelForCausalLM.from_pretrained(pt_path or llm_path) 

            if lora is not None and pt_path is None:
                lora = DictConfig(lora)
                peft_config = LoraConfig(
                    inference_mode=False, r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                    target_modules=lora.target_modules, modules_to_save=lora.modules_to_save,
                )
                llm = get_peft_model(llm, peft_config)

            if freeze_llm:
                for param in llm.parameters():
                    param.requires_grad = False


        # Set LLM
        llm.to(torch.float16)
        self.llm = llm
        self.llm_config = llm.config

        # Build encoder
        ndt1_pt_path = pt_path or kwargs.pop("load_ndt1_from_pt", None)
        if ndt1_pt_path is not None:
            config["ndt1"]["encoder"]["from_pt"] = ndt1_pt_path
            config["ndt1"]["decoder"]["from_pt"] = ndt1_pt_path
        self.ndt1 = NDT1(config.ndt1, **kwargs)


        # Build projector
        if pt_path is not None:
            projector_config = torch.load(os.path.join(pt_path, "projector_config.pth"))
            config["projector"] = update_config(config.projector, projector_config)

        self.stacking = config.projector.stacking
        if config.projector.inter_size is not None:
            self.projector = nn.Sequential(
                nn.Linear(config.ndt1.encoder.transformer.hidden_size * self.stacking, config.projector.inter_size, bias=config.projector.bias),
                ACT2FN[config.projector.act],
                nn.Linear(config.projector.inter_size, llm.config.hidden_size, bias=config.projector.bias)
            )
        else:
            self.projector = nn.Linear(config.ndt1.encoder.transformer.hidden_size * self.stacking, llm.config.hidden_size, bias=config.projector.bias)

        # Load encoder weights
        if pt_path is not None:
            self.projector.load_state_dict(torch.load(os.path.join(pt_path,"projector.bin")))

        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

        self.config = config


    def prepare_embeds(
            self, 
            input_ids:          torch.LongTensor,   # (batch, seq_len)
            attention_mask:     torch.LongTensor,   # (batch, seq_len)
            input_split:        torch.LongTensor,   # (bs)
            spikes:             torch.FloatTensor,  # (batch, seq_len_spikes)
            spikes_mask:        torch.FloatTensor,  # (batch, seq_len_spikes)
            spikes_timestamp:   torch.FloatTensor,  # (batch, seq_len_spikes)
            spikes_lengths:     torch.LongTensor,   # (bs)
            block_idx:          Optional[torch.LongTensor]  = None,     # (bs)
            day_idx:            Optional[torch.LongTensor]  = None,     # (bs)
            targets:             Optional[torch.LongTensor]  = None,     # (bs, seq_len)
    ) -> torch.FloatTensor:                     # (batch, seq_len, hidden_size)
        

        # Embed tokens of sentence
        text_embeds = (self.llm.get_input_embeddings())(input_ids)    # (batch, seq_len, hidden_size)
        
        spikes_embeds, spikes_mask, _ = self.ndt1.encoder(spikes, spikes_mask, spikes_timestamp, block_idx, day_idx) # (batch_size, seq_len_spikes, hidden_size)

        B, T, H = spikes_embeds.size()

        # Pad to be evenly stacked
        if T % self.stacking != 0:
            new_T = math.ceil(T / self.stacking) * self.stacking
            spikes_embeds = torch.cat((spikes_embeds, torch.zeros(B, new_T - T, H).to(spikes_embeds)), 1)
            spikes_mask = torch.cat((spikes_mask, torch.zeros(B, new_T - T).to(spikes_mask)), 1)
            T = new_T


        # Stack and project
        spikes_embeds = spikes_embeds.view(B,T//self.stacking,H*self.stacking)
        spikes_embeds = self.projector(spikes_embeds)
        spikes_mask = spikes_mask.view(B, T//self.stacking, self.stacking)
        spikes_mask = (spikes_mask.sum(-1) == self.stacking).to(attention_mask) # only keep new features that contain no padding

        input_embeds = torch.stack([
            torch.cat(
                (
                    t[:d], s, t[d:]
                ), dim=0
            )
        for t,s,d in zip(text_embeds, spikes_embeds, input_split)], dim=0)

        attention_mask = torch.stack([
            torch.cat(
                (
                    a[:d], s, a[d:]
                ), dim=0
            )
        for a,s,d in zip(attention_mask, spikes_mask, input_split)], dim=0)

        if targets is not None:
            targets = torch.stack([
            torch.cat(
                (
                    t[:d], torch.ones_like(s).to(t)*(-100), t[d:]
                ), dim=0
            )
        for t,s,d in zip(targets, spikes_mask, input_split)], dim=0)

        return input_embeds, attention_mask, targets   
        # (batch_size, tot_seq_len, hidden_size), (batch_size, tot_seq_len), (batch_size, tot_seq_len)



    def forward(
        self,
        input_ids:          torch.LongTensor,                   # (batch_size, seq_len)
        attention_mask:     torch.LongTensor,                   # (batch_size, seq_len)
        input_split:       torch.LongTensor,                    # (bs)
        spikes:             torch.FloatTensor,                  # (batch_size, seq_len_spikes, n_channels)
        spikes_mask:        torch.LongTensor,                   # (batch_size, seq_len_spikes)
        spikes_timestamp:   torch.LongTensor,                   # (batch_size, seq_len_spikes)
        spikes_lengths:     torch.LongTensor,                   # (bs)
        block_idx:          Optional[torch.LongTensor] = None,  # (batch_size)
        day_idx:           Optional[torch.LongTensor] = None,  # (batch_size)
        targets:             Optional[torch.LongTensor] = None,  # (batch_size, seq_len)
    ) -> BCIOutput:

        inputs_embeds, attention_mask, targets = self.prepare_embeds(input_ids, attention_mask, input_split, spikes,
                                        spikes_mask, spikes_timestamp, spikes_lengths, block_idx, day_idx, targets)
        
        # Forward decoder
        logits = self.llm(  
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits   # (batch_size, tot_seq_len, vocab_size)
        

        loss = None
        n_examples = None
        if targets is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.llm_config.vocab_size)
            shift_targets = shift_targets.view(-1)
            # Enable model parallelism
            shift_targets = shift_targets.to(shift_logits.device)
            loss = self.loss_fn(shift_logits, shift_targets)

            n_examples=(shift_targets != -100).sum()
        
        return BCIOutput(
            loss=loss,
            n_examples=n_examples,
            preds=logits,
            targets=targets,
        )
    

    """ Open ended generation
    """
    def generate(
        self,
        input_ids:          torch.LongTensor,                   # (batch_size, seq_len)
        attention_mask:     torch.LongTensor,                   # (batch_size, seq_len)
        input_split:       torch.LongTensor,                   # (bs)
        spikes:             torch.FloatTensor,                  # (batch_size, seq_len_spikes, n_channels)
        spikes_mask:        torch.LongTensor,                   # (batch_size, seq_len_spikes)
        spikes_timestamp:   torch.LongTensor,                   # (batch_size, seq_len_spikes)
        spikes_lengths:     torch.LongTensor,                   # (bs)
        block_idx:          Optional[torch.LongTensor]  = None, # (batch_size)
        day_idx:           Optional[torch.LongTensor]  = None,  # (batch_size)
        inputs_embeds:      Optional[torch.FloatTensor] = None, # (batch, seq_len, hidden_size)
        **gen_config:       DictConfig,
    ) -> List[torch.LongTensor]:  
         
        # Embed logits and merge with text embeddings
        if inputs_embeds is None:
            inputs_embeds, attention_mask, _ = self.prepare_embeds(input_ids, attention_mask, input_split, spikes,
                                        spikes_mask, spikes_timestamp, spikes_lengths, block_idx, day_idx, targets=None)
        
        # LLM built-in generation
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_config)



    def save_checkpoint(self, save_dir):
        # Save llm 
        self.llm.save_pretrained(save_dir)
        # Save ndt1
        self.ndt1.save_checkpoint(save_dir)

        # Save projector
        torch.save(self.projector.state_dict(), os.path.join(save_dir,"projector.bin"))
        torch.save(dict(self.config.projector), os.path.join(save_dir,"projector_config.pth"))

    def load_checkpoint(self, load_dir):
        # Save llm 
        self.llm = AutoModelForCausalLM.from_pretrained(load_dir).to(self.llm.device)
        self.ndt1.load_checkpoint(load_dir)
        self.projector.load_state_dict(torch.load(os.path.join(load_dir,"projector.bin")))

