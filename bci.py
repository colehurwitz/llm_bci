import os
from typing import List, Optional, Tuple, Dict

from transformers import LlamaPreTrainedModel
from transformers import LlamaModel, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft_wrapper import PeftModelForBCI, PeftConfig



# Dummy module that simulates the Encoder between the data and the Language Model
class Encoder(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.fc = nn.Linear(256, config.hidden_size)

    def forward(
            self, 
            features: torch.FloatTensor,
        ):

        return self.fc(features)
    

# Wrap transformer and lm_head of Llama
class LlamaDecoderWithLMHead(LlamaPreTrainedModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        # Architecture
        self.transformer = LlamaModel(config)
        self.lm_head     = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        
    def forward(
            self,
            inputs_embeds: torch.FloatTensor,
            attention_mask: torch.FloatTensor,

        ) -> torch.FloatTensor:


        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(  
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return logits
    

    ## COMPATIBILITY WITH HF METHODS ## 
    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.transformer
    
    def set_decoder(self, value):
        self.transformer = value



# BCI class. Subclass  of LlamaPretrainedModel to acces all the hf code (from_pretained, etc)
class BCI(LlamaPreTrainedModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
    

        # Configuration
        _no_split_modules = ["LlamaDecoderLayer"]  # Override llama default because we may want to add the encoder or the encoder layer module here
        self.vocab_size = config.vocab_size
        self._is_peft = False

        # Architecture
        self.encoder = Encoder(config)
        self.decoder = LlamaDecoderWithLMHead(config)

        # init weights
        self.post_init() # from hf



    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor,
            features: torch.FloatTensor,
            feature_mask: torch.FloatTensor,
            labels: Optional[torch.LongTensor] = None,
            **kwargs, # added for compatibility with hf model.generate

        ) -> CausalLMOutputWithPast:

 
        # Embed tokens of sentence
        sentence_embeds = self.decoder.transformer.embed_tokens(input_ids)

        # Embed neural signal
        neural_embeds = self.encoder(features)

        # Prepare inputs for decoder
        inputs_embeds = torch.cat((neural_embeds, sentence_embeds), -2)
        attention_mask = torch.cat((feature_mask, attention_mask), -1)

        # Forward decoder
        logits = self.decoder(  
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        
        loss = None
        if labels is not None:
            # Add features mask to match sizes
            labels = torch.cat((torch.ones_like(feature_mask, dtype=labels.dtype)*(-100), labels), -1)

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

    # Override from_pretrained method to set _is_peft
    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(model_name_or_path, *model_args, **kwargs)
        model._is_peft = False

        return model

    # Load pertrained model and create peft adapter
    @classmethod
    def peft_from_pretrained(cls, model_name_or_path, peft_config, *model_args, **kwargs):

        # Create BCI model and load Llama2 weights to decoder and lm_head
        model = cls.from_pretrained(model_name_or_path, *model_args, **kwargs)
        
        # Add peft adapter to the decoder
        model.decoder = PeftModelForBCI(model.decoder, peft_config)
        model._is_peft = True


        return model

    # Load pretrained adapter
    @classmethod
    def peft_from_adapter(cls, model_name_or_path, path_to_adapter, is_trainable=False, adapter_name = "default", *args, **kwargs):

        # Load pretrained Llama model
        model = cls.from_pretrained(model_name_or_path, *args, **kwargs)

        # Load trained weights for encoder
        model.encoder.load_state_dict(torch.load(os.path.join(path_to_adapter,"encoder.bin")))

        # Get peft config
        peft_config = PeftConfig.from_pretrained(path_to_adapter)
        peft_config.inference_mode = not is_trainable

        # Load trained adapter for decoder
        model.decoder = PeftModelForBCI(model.decoder, peft_config)
        model.decoder.load_adapter(path_to_adapter, adapter_name, is_trainable=is_trainable)
        model._is_peft = True

        return model


    ## SAVING METHODS ##

    # Override save_pretrained method to avoid issues with the adapter
    def save_pretrained(self, path_to_model, **kwargs):
        if self._is_peft:
            raise Exception("Peft adapter is loaded, merge before saving")
        super().save_pretrained(path_to_model, **kwargs)


    # Save trained model and adapter
    def save_adapter(self, path_to_adapter, **kwargs):
        
        if not self._is_peft:
            raise Exception("No peft adapter, model was not saved")
        
        if not os.path.exists(path_to_adapter):
            os.makedirs(path_to_adapter)
        
        # Save state all parameters of encoder and save decoder adapter
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
        self, input_ids, attention_mask, features, feature_mask, **kwargs
    ):
        
        model_inputs = {   
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "features": features,
                "feature_mask": feature_mask,
            }
        
        return model_inputs

