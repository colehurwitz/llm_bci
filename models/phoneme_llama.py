import os
import math
import yaml
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn

from transformers import LlamaPreTrainedModel, LlamaConfig
from transformers.utils import ModelOutput
from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from models.peft_wrapper import PeftModelWithoutLabels, PeftConfig
from models.llama_decoder import LlamaDecoderWithLMHead

from utils.config_utils import DictConfig, update_config

DEFAULT_CONFIG_FILE = "configs/phoneme_coupler.yaml"

@dataclass
class PhonemeLlamaOutput(ModelOutput):
    logits:     torch.FloatTensor
    labels:     Optional[torch.FloatTensor] = None
    loss:       Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor]  = None


""" PhonemeLlama class. Subclass  of LlamaPretrainedModel to acces all the hf code (from_pretained, generate, etc.)
"""
class PhonemeLlama(LlamaPreTrainedModel):

    def __init__(self, config: LlamaConfig, coupler_config: DictConfig):
        super().__init__(config)

        # Configuration
        self.config = config
        self.coupler_config = coupler_config

        self.vocab_size = config.vocab_size
        self._is_peft = False

        # Architecture
        if coupler_config.inter_size is not None:
            self.coupler = nn.Sequential(
                nn.Linear(coupler_config.input_size, coupler_config.inter_size, bias=coupler_config.bias),
                ACT2FN[coupler_config.act],
                nn.Linear(coupler_config.inter_size, config.hidden_size, bias=coupler_config.bias)
            )
        else:
            self.coupler = nn.Linear(coupler_config.input_size, config.hidden_size, bias=coupler_config.bias)

        self.decoder = LlamaDecoderWithLMHead(self.config)
        self.loss_fn = nn.CrossEntropyLoss(reduction=coupler_config.loss_reduction)


    def forward(
            self,
            input_ids:          torch.LongTensor,                   # (batch, seq_len)
            attention_mask:     torch.LongTensor,                   # (batch, seq_len)
            phoneme_logits:     List[torch.FloatTensor],            # batch * [(seq_len_phon, vocab)]
            phonemes_start:     torch.LongTensor,                   # (batch)
            phonemes_end:       torch.LongTensor,                   # (batch)
            labels:             Optional[torch.LongTensor] = None,  # (batch, seq_len)
        ) -> PhonemeLlamaOutput:
        

        inputs_embeds = self.prepare_embeds(input_ids, phoneme_logits, phonemes_start, phonemes_end)
        
        # Forward decoder
        logits = self.decoder(  
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits   # (batch, seq_len, vocab)
    
        loss = None
        n_examples = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fn(shift_logits, shift_labels)

            n_examples=(labels != -100).sum().detach().cpu().item()
        
        return PhonemeLlamaOutput(
            logits=logits,
            labels=labels,
            loss=loss,
            n_examples=n_examples,
        )
    

    """ Prepare embeddings for llm decoder
    """ 
    def prepare_embeds(self, input_ids, phoneme_logits, phonemes_start, phonemes_end):
        
        # Embed tokens of sentence
        text_embeds = self.decoder.transformer.embed_tokens(input_ids)  # (batch, seq_len, hidden_size)
        
        # Embed phoneme logits
        phoneme_embeds = [self.coupler(l) for l in phoneme_logits]      # (batch, seq_len_phon, hidden_size)

        # Substitute phoneme_embeds in input embeds
        return self.sub_embeds(text_embeds, phoneme_embeds, phonemes_start, phonemes_end)


    """ Substitute phoneme embeddings into text embeddings
    """ 
    def sub_embeds(self, text_embeds,phoneme_embeds, phonemes_start, phonemes_end):

        input_embeds = [
            torch.cat(
                (
                    t[:a], p, t[b:]
                ), dim=0
            )
        for t,p,a,b in zip(text_embeds, phoneme_embeds, phonemes_start, phonemes_end)]

        return torch.stack(input_embeds, dim=0)

    def predict(
            self,
            input_ids:          torch.LongTensor,                   # (batch, seq_len)
            attention_mask:     torch.LongTensor,                   # (batch, seq_len)
            phoneme_logits:     List[torch.FloatTensor],            # batch * [(seq_len_phon, vocab)]
            phonemes_start:     torch.LongTensor,                   # (batch)
            phonemes_end:       torch.LongTensor,                   # (batch)
            synced_gpus:        bool,
            **gen_config:       DictConfig,
        ) -> List[torch.LongTensor]:

        inputs_embeds = self.prepare_embeds(input_ids, phoneme_logits, phonemes_start, phonemes_end)

        return self.decoder.generate(inputs_embeds=inputs_embeds, **gen_config, synced_gpus=synced_gpus)


    ## LOADING METHODS ##

    """ Wrap from/save_pretrained method to set _is_peft and deal with config for coupler. When the coupler
        config file is not found, it is assumed that the state_dict doesn't contain the weights of the coupler 
        and it is expected to see a message from hf initializing the encoder weights.
    """
    @classmethod
    def from_pretrained(cls, model_dir, coupler_config=None, **kwargs):

        # Prepare default config
        default_config = update_config(DEFAULT_CONFIG_FILE, None)

        # Update default config with pretrained config or user config
        config_file = os.path.join(model_dir, "coupler_config.yaml")
        coupler_config = config_file if os.path.isfile(config_file) else coupler_config
        coupler_config = update_config(default_config, coupler_config)
        
        # Load with hf method
        model = super().from_pretrained(model_dir, coupler_config, **kwargs)
        model._is_peft = False

        return model


    def save_pretrained(self, model_dir, **kwargs):
        if self._is_peft:
            raise Exception("Peft adapter is loaded, merge before saving")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        yaml.dump(dict(self.coupler_config), open(os.path.join(model_dir, "coupler_config.yaml"), "w"), default_flow_style=False)
        super().save_pretrained(model_dir, **kwargs)

    @classmethod
    def from_config(cls, config, coupler_config=None):

        # Update default config with user config
        coupler_config = update_config(DEFAULT_CONFIG_FILE, coupler_config)
        
        # Load with hf method
        model = cls(config, coupler_config)
        model._is_peft = False

        return model

    ## ADAPTER METHODS ##
    def load_adapter(self, adapter_dir, is_trainable=False, adapter_name="default", **kwargs):

        # Get peft config
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        peft_config.inference_mode = not is_trainable

        # Load trained adapter for decoder
        self.decoder = PeftModelWithoutLabels(self.decoder, peft_config)
        self.decoder.load_adapter(adapter_dir, adapter_name, is_trainable=is_trainable)
        self._is_peft = True


    def create_adapter(self, peft_config):
        if self._is_peft:
            raise Exception("Peft adapter already loaded")

        self.decoder = PeftModelWithoutLabels(self.decoder, peft_config)
        self._is_peft = True


    def save_adapter(self, adapter_dir, **kwargs):
        
        if not self._is_peft:
            raise Exception("No peft adapter loaded")

        if not os.path.exists(adapter_dir):
            os.makedirs(adapter_dir)

        self.decoder.save_pretrained(adapter_dir, **kwargs)
        

    def merge_adapter(self):
        if not self._is_peft:
            raise Exception("No peft adapter loaded")

        self.decoder = self.decoder.merge_and_unload()
        self._is_peft = False


    def unload_adapter(self):
        if not self._is_peft:
            raise Exception("No peft adapter loaded")

        self.decoder = self.decoder.unload()
        self._is_peft = False

    
    # COUPLER METHODS
    def load_coupler(self, coupler_dir):
        config_file = os.path.join(coupler_dir, "coupler_config.yaml")
        coupler_config = update_config(DEFAULT_CONFIG_FILE, config_file)

        self.coupler = nn.Sequential(
            nn.Linear(coupler_config.input_size, coupler_config.inter_size, bias=coupler_config.bias),
            ACT2FN[coupler_config.act],
            nn.Linear(coupler_config.inter_size, self.config.hidden_size, bias=coupler_config.bias),
        )
        self.coupler.load_state_dict(torch.load(os.path.join(coupler_dir,"coupler.bin")))


    def save_coupler(self, coupler_dir):
        yaml.dump(dict(self.coupler_config), open(os.path.join(coupler_dir, "coupler_config.yaml"),"w"), default_flow_style=False)
        torch.save(self.coupler.state_dict(), os.path.join(coupler_dir,"coupler.bin"))
   

    ## INITIALIZATION ##

    # Override default method for initialization. This is called on parameters that are not in the saved state_dict,
    # i.e., the encoder parameters and the new part of the lm (because of resizing)
    # def _init_weights(self, module):
    #     print("aaa")
    #     # All copied from Llama
    #     std = self.config.initializer_range
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0,std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()

    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    
    # Another way of accessing the initialization of weights
    # def _init_encoder_weights(self):
    #     std = self.config.initializer_range
    #     for pn, p in self.named_parameters():
    #         if pn == 'encoder.fc.weight':
    #             pass



