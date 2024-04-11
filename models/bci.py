import os
import math
import yaml
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from transformers import PreTrainedModel, LlamaConfig
from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from models.ndt1 import NDT1
from models.trainer import ModelOutput

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
        llm: PreTrainedModel,
    ):

        super().__init__()

        config = update_config(DEFAULT_CONFIG, config)
        
        # Set LLM
        self.llm = llm
        self.llm_config = llm.config

        # Build encoder
        ndt1_pt_path = config["load_ndt1_from_pt"]
        if ndt1_pt_path is not None:
            config["ndt1"]["encoder"]["from_pt"] = ndt1_pt_path
            config["ndt1"]["decoder"]["from_pt"] = ndt1_pt_path
        self.ndt1 = NDT1(config.ndt1)


        # Build projector
        projector_pt_path = config["projector"].pop("from_pt", None)
        if projector_pt_path is not None:
            projector_config = torch.load(os.path.join(projector_pt_path, "projector_config.pth"))
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
        if projector_pt_path is not None:
            self.projector.load_state_dict(torch.load(os.path.join(encoder_pt_path,"projector.bin")))

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
            labels:             Optional[torch.LongTensor]  = None,     # (bs, seq_len)
    ) -> torch.FloatTensor:                     # (batch, seq_len, hidden_size)
        
        # Embed tokens of sentence
        text_embeds = (self.llm.get_input_embeddings())(input_ids)    # (batch, seq_len, hidden_size)
        
        spikes_embeds, _ = self.encoder(spikes, spikes_mask, spikes_timestamp, block_idx, date_idx) # (batch_size, seq_len_spikes, hidden_size)

        B, T, H = spikes_embeds.size()
         
        # Pad to be evenly stacked
        if T % self.stacking != 0:
            new_T = math.ceil(T / self.stacking) * self.stacking
            spikes_embeds = torch.cat((torch.zeros(B, new_T - T, H).to(spikes_embeds), spikes_embeds), 1)
            spikes_mask = torch.cat((torch.zeros(B, new_T - T).to(spikes_mask), spikes_mask), 1)
            T = new_T

        # Stack and project
        spikes_embeds = spikes_embeds.view(B,T//self.stacking,H*self.stacking)
        spikes_embeds = self.stack_projector(spikes_embeds)
        spikes_mask = spikes_mask.view(B, T//self.stacking, self.stacking)
        spikes_mask = (spikes_mask.sum(-1) == self.stacking).to(attention_mask) # only keep new features that contain no padding

        input_embeds = torch.stack([
            torch.cat(
                (
                    t[:s], e, t[s:]
                ), dim=0
            )
        for t,e,s in zip(text_embeds, spikes_embeds, input_split)], dim=0)

        attention_mask = torch.stack([
            torch.cat(
                (
                    t[:s], e, t[s:]
                ), dim=0
            )
        for t,e,s in zip(attention_mask, spikes_mask, input_split)], dim=0)

        if labels is not None:
            labels = torch.stack([
            torch.cat(
                (
                    l[:s], torch.ones_like(e).to(l)*(-100), l[s:]
                ), dim=0
            )
        for l,e,s in zip(labels, spikes_embeds, input_split)], dim=0)


        return input_embeds, attention_mask, labels   
        # (batch_size, tot_seq_len, hidden_size), (batch_size, tot_seq_len), (batch_size, tot_seq_len)



    def forward(
        self,
        input_ids:          torch.LongTensor,                   # (batch_size, seq_len)
        attention_mask:     torch.LongTensor,                   # (batch_size, seq_len)
        inputt_split:       torch.LongTensor,                   # (bs)
        spikes:             torch.FloatTensor,                  # (batch_size, fea_len, n_channels)
        spikes_mask:        torch.LongTensor,                   # (batch_size, fea_len)
        spikes_timestamp:   torch.LongTensor,                   # (batch_size, fea_len)
        spikes_lengths:     torch.LongTensor,                   # (bs)
        block_idx:          Optional[torch.LongTensor] = None,  # (batch_size)
        date_idx:           Optional[torch.LongTensor] = None,  # (batch_size)
        labels:             Optional[torch.LongTensor] = None,  # (batch_size, seq_len)
    ) -> BCIOutput:

        
        inputs_embeds, attention_mask, labels = self.prepare_embeds(input_ids, attention_mask, input_split, 
                                        spikes_mask, spikes_timestamp, spikes_lengths, block_idx, day_idx, labels)
        
        # Forward decoder
        logits = self.decoder(  
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )   # (batch_size, tot_seq_len, vocab_size)
        

        loss = None
        n_examples = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.llm_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fn(shift_logits, shift_labels)

            n_examples=(shift_labels != -100).sum()
        
        return BCIOutput(
            loss=loss,
            n_examples=n_examples,
            preds=logits,
            targets=labels,
        )
    

    """ Open ended generation
    """
    def generate(
        self,
        input_ids:          torch.LongTensor,                   # (batch_size, seq_len)
        attention_mask:     torch.LongTensor,                   # (batch_size, seq_len)
        inputt_split:       torch.LongTensor,                   # (bs)
        spikes:             torch.FloatTensor,                  # (batch_size, fea_len, n_channels)
        spikes_mask:        torch.LongTensor,                   # (batch_size, fea_len)
        spikes_timestamp:   torch.LongTensor,                   # (batch_size, fea_len)
        spikes_lengths:     torch.LongTensor,                   # (bs)
        block_idx:          Optional[torch.LongTensor]  = None, # (batch_size)
        date_idx:           Optional[torch.LongTensor]  = None, # (batch_size)
        inputs_embeds:      Optional[torch.FloatTensor] = None, # (batch, seq_len, hidden_size)
        **gen_config:       DictConfig,
    ) -> List[torch.LongTensor]:  
         
        # Embed logits and merge with text embeddings
        if inputs_embeds is None:
            inputs_embeds, attention_mask, _ = self.prepare_embeds(input_ids, attention_mask, input_split, 
                                        spikes_mask, spikes_timestamp, spikes_lengths, block_idx, day_idx, labels=None)
        
        # LLM built-in generation
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_config)



    def save_checkpoint(self, save_dir):
        # Save llm 
        self.llm.save_pretrained(save_dir)
        self.ndt1.save_checkpoint(save_dir)
        torch.save(self.projector.state_dict(), os.path.join(save_dir,"projector.bin"))
        torch.save(dict(self.config.projector), os.path.join(save_dir,"projector_config.pth"))

