Launch script:
    - accelerate launch --config_file fsdp_config.yaml finetune.py


Loading:
    - From pretrained Llama and create adapter: peft_from_pretrained
    - From pretrained Llama and load trained adapter:  peft_from_adapter
    - From hf checkpoint (doesn't create adapter): from_pretrained

Saving:
    - Save decoder adapter and encoder separately with save_adapter.
    - Merge decoder and save_pretrained.






NEW FILES
    - bci: contains the Encoder and Decoder models 
    - finetune: script for finetuning BCI model
    - inference: script to test inference on BCI model
    - change_prefix: script to manipulate state_dict keys
    - fsdp_config.yaml: accelerate config for distributed training

CHANGES
    - peft/mapping.py to add PeftModelForOnlyLM to the Mapping dict
    - peft/peft_model.oy to add PeftModelForOnlyLM which is PeftModelForCausalLM without the labels
    - peft/utils/peft_types.py to add OnlyLM to TaskTypes
    - pretrained llama model state dict to match the new model architecture





OBS
- The transformer part of the decoder cannot be named model as in the original code because modules 
wrapped in peft have an attribute "model" that references the unwrapped model, so we would have to add
logic to call decoder.model.model only when the peft adapter is loaded
- _init_weights is called by hf whenever a key for a parameter is not found in the state_dict and when 
resizing the embedding and lm_head after adding tokens to the vocabulary
- the peft model class forces the signature of the underlying model to contain "labels", because it is 
supposed to wrap a ModelForCausalLM. I created a new peft model class to avoid this behaviour, as we may
want to compute loss and predicitions in a custom way.
- I had to change the keys in the original state_dict of llama to match the names of the parameters in the
BCI architecture

ARCHITECTURE
- Encoder: data embedding + transformer encoder
- Decoder: llama decoder + llama lm_head



TO DO:
- Try distributed training with DeepSpeed
- Try if full precision model fits in 1 GPU