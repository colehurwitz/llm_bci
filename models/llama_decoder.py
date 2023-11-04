from transformers import LlamaPreTrainedModel, LlamaModel, LlamaConfig

import torch
import torch.nn as nn

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

    # Wrap hf default method to not mess with the lm_head and embeddings
    def resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of = None):
        input_requires_grad = self.get_input_embeddings().weight.requires_grad
        output_requires_grad = self.get_output_embeddings().weight.requires_grad
        value = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        for param in self.get_input_embeddings().parameters():
            param.requires_grad = input_requires_grad
        for param in self.get_output_embeddings().parameters():
            param.requires_grad = output_requires_grad
        return value


