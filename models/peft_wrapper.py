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
            inputs_embeds: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
        ) -> torch.FloatTensor:

        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )