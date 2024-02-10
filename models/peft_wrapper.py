from typing import Optional, List

import torch

from peft import PeftModel, PeftConfig

# Peft wrapper that is compatible with models without LM head
# Only changes w.r.t PeftModelForCausalLM is that this model doesn't force "labels" in the module signature,
# and eliminated code for other use cases (prompt learning, generation)
class PeftModelWithoutLabels(PeftModel):
    """
    Peft model for causal language modeling without labels.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.

        ```
    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(
            self,
            input_ids:              Optional[torch.LongTensor]          = None,
            inputs_embeds:          Optional[torch.FloatTensor]         = None,
            attention_mask:         Optional[torch.Tensor]              = None,
            position_ids:           Optional[torch.LongTensor]          = None,
            past_key_values:        Optional[List[torch.FloatTensor]]   = None,
            use_cache:              Optional[bool] = None,
            output_attentions:      Optional[bool] = None,
            output_hidden_states:   Optional[bool] = None,
            return_dict:            Optional[bool] = None,
        ):

        return self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )