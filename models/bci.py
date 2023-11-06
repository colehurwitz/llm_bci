import os
from typing import List, Optional, Tuple, Dict

from transformers import LlamaPreTrainedModel, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.peft_wrapper import PeftModelForBCI, PeftConfig
from models.llama_decoder import LlamaDecoderWithLMHead
from models.neural_encoder import NeuralEncoder, NeuralConfig



# BCI class. Subclass  of LlamaPretrainedModel to acces all the hf code (from_pretained, etc)
class BCI(LlamaPreTrainedModel):

    def __init__(self, llama_config: LlamaConfig, neural_config: NeuralConfig):
        super().__init__(llama_config)

        # Configuration
        self.neural_config = neural_config
        _no_split_modules = ["LlamaDecoderLayer", "NeuralEncoderLayer"]  # Override llama default because we may want to add the encoder or the encoder layer module here
        self.vocab_size = llama_config.vocab_size
        self._is_peft = False

        # Architecture
        self.encoder = NeuralEncoder(neural_config)
        self.decoder = LlamaDecoderWithLMHead(llama_config)

        # init weights
        self.post_init() # from hf


    def forward(
            self,
            input_ids: torch.LongTensor,                # (batch_size, seq_len)
            attention_mask: torch.FloatTensor,          # (batch_size, seq_len)
            features: torch.FloatTensor,                # (batch_size, fea_len, n_channels)
            features_mask: torch.FloatTensor,           # (batch_size, fea_len)
            features_timestamp: torch.LongTensor,       # (batch_size, fea_len)
            block_idx: torch.LongTensor,                # (batch_size, fea_len)
            date_idx: torch.LongTensor,                 # (batch_size, fea_len)
            labels: Optional[torch.LongTensor] = None,  # (batch_size, seq_len)
            **kwargs, # added for compatibility with hf model.generate

        ) -> CausalLMOutputWithPast:

        # Encode neural signal
        neural_embeds = self.encoder(features, features_mask, features_timestamp, block_idx, date_idx) # (batch_size, lat_len, hidden_size)

        # Embed tokens of sentence
        sentence_embeds = self.decoder.transformer.embed_tokens(input_ids)  # (batch_size, seq_len, hidden_size)

        # Prepare inputs for decoder
        neural_mask = torch.ones(neural_embeds.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype)
        inputs_embeds = torch.cat((neural_embeds, sentence_embeds), -2)   # (batch_size, lat+seq_len, hidden_size)
        attention_mask = torch.cat((neural_mask, attention_mask), -1)     # (batch_size, lat+seq_len, hidden_size)

        # Forward decoder
        logits = self.decoder(  
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )   # (batch_size, lat+seq_len, vocab_size)
        
        loss = None
        if labels is not None:
            # Add features mask to match sizes
            labels = torch.cat((
                        torch.ones(neural_embeds.shape[:2], device=labels.device, dtype=labels.dtype)*(-100), 
                        labels
                    ), -1)          # (batch_size, lat+seq_len)

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
    


    ## LOADING METHODS ##

    # Wrap from_pretrained method to set _is_peft and deal with neural_config. If neural_config is not 
    # provided, it will look for it in the checkpoint folder. This will raise an error when loading the llama
    # checkpoint 
    @classmethod
    def from_pretrained(cls, model_name_or_path, neural_config=None, **kwargs):
        if neural_config is None:
            neural_config_file = os.path.join(model_name_or_path, "neural_config.bin")
            neural_config = torch.load(neural_config_file) if neural_config is None else NeuralConfig()
        model = super().from_pretrained(model_name_or_path, neural_config, **kwargs)
        model._is_peft = False

        return model

    # Load pertrained model and create peft adapter. If neural_config is not provided, it will look for it in the
    # checkpoint folder. This will raise an error when loading the llama checkpoint 
    @classmethod
    def peft_from_pretrained(cls, model_name_or_path, peft_config, neural_config=None, **kwargs):

        # Create BCI model and load Llama2 weights to decoder and lm_head
        model = cls.from_pretrained(model_name_or_path, neural_config, **kwargs)
        
        # Add peft adapter to the decoder
        model.decoder = PeftModelForBCI(model.decoder, peft_config)
        model._is_peft = True

        return model

    # Load pretrained adapter. If neural_config is not provided, it will look for it in the adapter folder. 
    @classmethod
    def peft_from_adapter(cls, model_name_or_path, path_to_adapter, is_trainable=False, adapter_name="default", neural_config=None, **kwargs):
        
        neural_config_file = os.path.join(path_to_adapter, "neural_config.bin")
        assert os.path.isfile(neural_config_file), """Attempting to load pretrained config for NeuralEncoder but 
        neural_config was not found in {}""".format(path_to_adapter)
        neural_config = torch.load(neural_config_file) if neural_config is None else neural_config

        # Load pretrained Llama model
        model = cls.from_pretrained(model_name_or_path, neural_config, **kwargs)

        # Load trained weights for encoder
        print(f"Loading encoder weights from {path_to_adapter}")
        model.encoder.load_state_dict(torch.load(os.path.join(path_to_adapter,"encoder.bin")))

        # Get peft config
        peft_config = PeftConfig.from_pretrained(path_to_adapter)
        peft_config.inference_mode = not is_trainable

        # Load trained adapter for decoder
        print(f"Loading decoder adapter from {path_to_adapter}")
        model.decoder = PeftModelForBCI(model.decoder, peft_config)
        model.decoder.load_adapter(path_to_adapter, adapter_name, is_trainable=is_trainable)
        model._is_peft = True

        return model


    ## SAVING METHODS ##

    # Wrap save_pretrained method to avoid issues with the adapter and deal with neural_config
    def save_pretrained(self, path_to_model, **kwargs):
        if self._is_peft:
            raise Exception("Peft adapter is loaded, merge before saving")

        if not os.path.exists(path_to_model):
            os.makedirs(path_to_model)

        torch.save(self.neural_config, os.path.join(path_to_model, "neural_config.bin"))
        super().save_pretrained(path_to_model, **kwargs)


    # Save trained model and adapter
    def save_adapter(self, path_to_adapter, **kwargs):
        
        if not self._is_peft:
            raise Exception("No peft adapter, model was not saved")
        
        if not os.path.exists(path_to_adapter):
            os.makedirs(path_to_adapter)

        torch.save(self.neural_config, os.path.join(path_to_adapter, "neural_config.bin"))
        torch.save(self.encoder.state_dict(), os.path.join(path_to_adapter,"encoder.bin"))
        self.decoder.save_pretrained(path_to_adapter, **kwargs)


    ## ADAPTER METHODS ##

    # Merge adapter with weights of decoder
    def merge_decoder(self):
        if not self._is_peft:
            print("No peft adapter loaded, nothing was merged.")
            return
        
        self.decoder = self.decoder.merge_and_unload()
        self._is_peft = False

    # Remove peft adapter
    def unload_adapter(self):
        if not self._is_peft:
            print("No peft adapter loaded, nothing was unloaded.")
            return
        self.decoder = self.decoder.unload()
        self._is_peft = False

   

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

