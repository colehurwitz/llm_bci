import os
import math
import yaml
from typing import List, Optional, Tuple, Dict

from transformers import LlamaPreTrainedModel, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.peft_wrapper import PeftModelForBCI, PeftConfig
from models.llama_decoder import LlamaDecoderWithLMHead
from models.neural_encoder import NeuralEncoder

from utils.config_utils import DictConfig, update_config

DEFAULT_BCI_CONFIG_FILE = "configs/default_bci_config.yaml"
DEFAULT_NEURAL_CONFIG_FILE = "configs/default_neural_config.yaml"

# BCI class. Subclass  of LlamaPretrainedModel to acces all the hf code (from_pretained, etc)
class BCI(LlamaPreTrainedModel):

    def __init__(self, llama_config: LlamaConfig, bci_config: DictConfig):
        super().__init__(llama_config)

        # Configuration
        self.bci_config = bci_config
        self.neural_config = bci_config.neural_config
        self.llama_config = llama_config

        _no_split_modules = ["LlamaDecoderLayer", "NeuralEncoderLayer"]  # Override llama default because we may want to add the encoder or the encoder layer module here
        self.vocab_size = llama_config.vocab_size
        self._is_peft = False

        # Architecture
        self.encoder = NeuralEncoder(self.neural_config)
        self.stacking = self.bci_config.stacking
        self.stack_projector = nn.Linear(self.neural_config.embed_mult*self.neural_config.n_channels*self.stacking, self.llama_config.hidden_size, bias=self.bci_config.projector_bias)
        self.decoder = LlamaDecoderWithLMHead(self.llama_config)
        self.loss_fn = nn.CrossEntropyLoss(reduction=self.bci_config.loss_reduction)

        # init weights
        self.post_init() # from hf


    def forward(
            self,
            input_ids:          torch.LongTensor,                   # (batch_size, seq_len)
            attention_mask:     torch.FloatTensor,                  # (batch_size, seq_len)
            features:           torch.LongTensor,                   # (batch_size, fea_len, n_channels)
            features_mask:      torch.LongTensor,                  # (batch_size, fea_len)
            features_timestamp: torch.LongTensor,                   # (batch_size, fea_len)
            block_idx:          torch.LongTensor,                   # (batch_size, fea_len)
            date_idx:           torch.LongTensor,                   # (batch_size, fea_len)
            labels:             Optional[torch.LongTensor] = None,  # (batch_size, seq_len)
            **kwargs, # added for compatibility with hf model.generate

        ) -> CausalLMOutputWithPast:

        # Encode neural signal
        features_embeds = self.encoder(features, features_mask, features_timestamp, block_idx, date_idx) # (batch_size, fea_len, hidden_size)
        B, T, n = features_embeds.size()

        # Pad to be evenly stacked
        if T % self.stacking != 0:
            new_fea_len = math.ceil(T / self.stacking) * self.stacking
            features_embeds = torch.cat((torch.zeros(B, new_fea_len - T, n).to(features_embeds.device, features_embeds.dtype), features_embeds), 1)
            features_mask = torch.cat((torch.zeros(B, new_fea_len - T).to(features_mask.device, features_mask.dtype), features_mask), 1).to(features_mask.dtype)
            T = new_fea_len
        
        # Stack and project
        features_embeds = features_embeds.view(B,T//self.stacking,n*self.stacking)
        features_embeds = self.stack_projector(features_embeds)
        features_mask = features_mask.view(B, T//self.stacking, self.stacking)
        features_mask = (features_mask.sum(-1) == self.stacking).to(attention_mask.dtype) # only keep new features that contain no padding

        # Embed tokens of sentence
        sentence_embeds = self.decoder.transformer.embed_tokens(input_ids)  # (batch_size, seq_len, hidden_size)

        # Prepare inputs for decoder
        inputs_embeds = torch.cat((features_embeds, sentence_embeds), 1)   # (batch_size, fea+seq_len, hidden_size)
        attention_mask = torch.cat((features_mask, attention_mask), 1)   # (batch_size, fea+seq_len)

        # Forward decoder
        logits = self.decoder(  
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )   # (batch_size, fea+seq_len, vocab_size)
        



        loss = None
        if labels is not None:
            # Add features mask to match sizes
            labels = torch.cat((
                        torch.ones(features_embeds.shape[:2], device=labels.device, dtype=labels.dtype)*(-100), 
                        labels
                    ), 1)          # (batch_size, fea+seq_len)

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fn(shift_logits, shift_labels)

        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
    


    ## LOADING METHODS ##

    # Wrap from/save_pretrained method to set _is_peft and deal with neural_config. When neural_config is not
    # found, it is assumed that the state_dict doesn't contain the weights of an encoder and it is expected
    # to see a message from hf initializing the encoder weights.
    @classmethod
    def from_pretrained(cls, path_to_model, bci_config=None, **kwargs):
        
        print(f"Loading model from {path_to_model}")

        # Prepare default config
        bci_default_config = update_config(DEFAULT_BCI_CONFIG_FILE, None)


        # Update default config with pretrained config or user config
        bci_config_file = os.path.join(path_to_model, "bci_config.yaml")
        bci_config = bci_config_file if os.path.isfile(bci_config_file) else bci_config
        bci_config = update_config(bci_default_config, bci_config)
        
        # Load with hf method
        model = super().from_pretrained(path_to_model, bci_config, **kwargs)
        model._is_peft = False

        return model


    def save_pretrained(self, path_to_model, **kwargs):
        if self._is_peft:
            raise Exception("Peft adapter is loaded, merge before saving")
        print(f"Saving model to  {path_to_model}")

        if not os.path.exists(path_to_model):
            os.makedirs(path_to_model)

        yaml.dump(dict(self.bci_config), open(os.path.join(path_to_model, "bci_config.yaml"), "w"), default_flow_style=False)
        super().save_pretrained(path_to_model, **kwargs)

        

    # ENCODER METHODS

    def load_encoder(self, path_to_encoder):
        print(f"Loading encoder from {path_to_encoder}")
        neural_config_file = os.path.join(path_to_encoder, "neural_config.yaml")
        neural_config = update_config(DEFAULT_NEURAL_CONFIG_FILE, neural_config_file)

        self.encoder = NeuralEncoder(neural_config)
        self.encoder.load_state_dict(torch.load(os.path.join(path_to_encoder,"encoder.bin")))


    def save_encoder(self, path_to_encoder):
        print(f"Saving encoder to  {path_to_encoder}")
        yaml.dump(dict(self.neural_config), open(os.path.join(path_to_encoder, "neural_config.yaml"),"w"), default_flow_style=False)
        torch.save(self.encoder.state_dict(), os.path.join(path_to_encoder,"encoder.bin"))
        

    ## ADAPTER METHODS ##

    def load_adapter(self, path_to_adapter, is_trainable=False, adapter_name="default", **kwargs):
        
        print(f"Loading adapter from {path_to_adapter}")

        # Get peft config
        peft_config = PeftConfig.from_pretrained(path_to_adapter)
        peft_config.inference_mode = not is_trainable

        # Load trained adapter for decoder
        self.decoder = PeftModelForBCI(self.decoder, peft_config)
        self.decoder.load_adapter(path_to_adapter, adapter_name, is_trainable=is_trainable)
        self._is_peft = True


    def create_adapter(self, peft_config):
        if self._is_peft:
            raise Exception("Peft adapter already loaded")
        print("Creating new adapter")

        self.decoder = PeftModelForBCI(self.decoder, peft_config)
        self._is_peft = True


    def save_adapter(self, path_to_adapter, **kwargs):
        
        if not self._is_peft:
            raise Exception("No peft adapter loaded")
        print(f"Saving adapter to  {path_to_adapter}")

        if not os.path.exists(path_to_adapter):
            os.makedirs(path_to_adapter)

        self.decoder.save_pretrained(path_to_adapter, **kwargs)
        

    def merge_adapter(self):
        if not self._is_peft:
            raise Exception("No peft adapter loaded")
        print("Merging adapter")

        self.decoder = self.decoder.merge_and_unload()
        self._is_peft = False


    def unload_adapter(self):
        if not self._is_peft:
            raise Exception("No peft adapter loaded")

        print("Unloading adapter")
        self.decoder = self.decoder.unload()
        self._is_peft = False

       

   

    ## INITIALIZATION ##

    # Override default method for initialization. This is called on parameters that are not in the saved state_dict,
    # i.e., the encoder parameters and the new part of the lm (because of resizing)
    def _init_weights(self, module):

        # All copied from Llama
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0,std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    # Another way of accessing the initialization of weights
    def _init_encoder_weights(self):
        std = self.config.initializer_range
        for pn, p in self.named_parameters():
            if pn == 'encoder.fc.weight':
                pass


    # Override hf method (requirment for generation)
    def prepare_inputs_for_generation(
        self, input_ids, attention_mask, features, features_mask, features_timestamp, block_idx, date_idx, **kwargs
    ):
        
        model_inputs = {   
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "features": features,
                "features_mask": features_mask,
                "features_timestamp": features_timestamp,
                "block_idx": block_idx,
                "date_idx": date_idx,
            }
        
        return model_inputs


