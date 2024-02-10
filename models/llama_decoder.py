from typing import Optional, Union, Tuple, List
from transformers import LlamaPreTrainedModel, LlamaModel, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import torch
import torch.nn as nn

# Wrap transformer and lm_head of Llama
class LlamaDecoderWithLMHead(LlamaPreTrainedModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.config = config
        self.vocab_size = config.vocab_size

        # Architecture
        self.transformer = LlamaModel(config)
        self.lm_head     = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    

    def forward(
        self,
        input_ids:              torch.LongTensor                    = None,
        inputs_embeds:          Optional[torch.FloatTensor]         = None,
        attention_mask:         Optional[torch.Tensor]              = None,
        position_ids:           Optional[torch.LongTensor]          = None,
        past_key_values:        Optional[List[torch.FloatTensor]]   = None,
        use_cache:              Optional[bool] = None,
        output_attentions:      Optional[bool] = None,
        output_hidden_states:   Optional[bool] = None,
        return_dict:            Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
       
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    ## COMPATIBILITY WITH HF METHODS ## 
    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    # @staticmethod
    # def _reorder_cache(past_key_values, beam_idx):
    #     reordered_past = ()
    #     for layer_past in past_key_values:
    #         reordered_past += (
    #             tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
    #         )
    #     return reordered_past

    # def resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of = None):
    #     input_requires_grad = self.get_input_embeddings().weight.requires_grad
    #     output_requires_grad = self.get_output_embeddings().weight.requires_grad
    #     value = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
    #     for param in self.get_input_embeddings().parameters():
    #         param.requires_grad = input_requires_grad
    #     for param in self.get_output_embeddings().parameters():
    #         param.requires_grad = output_requires_grad
    #     return value