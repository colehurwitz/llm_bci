Launch finetuning script:
    - accelerate launch --config_file deepspeed.yaml finetune.py

Launch inference script:
    - python inference.py 

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
    - change_prefix: script to adapt llama2 state_dict to BCI state_dict
    - fsdp_config.yaml: accelerate config for distributed training

CHANGES
    - adapted pretrained llama2 model state dict to match the new model architecture


TO DO
-which part of the model is under LORA ?
-Using higher lr for encoder and lower for decoder?
-pad_to_multiple_of ?
-data augmentation by choosing subset of channels?
-penalization proportional to the frequency of the token

OBS
-Max block and date index has to be specified in NeuralConfig, read from preprocess.py output
-Right now I am using the same timestamps for all the examples, which means that effectively some of them are slightly
compressed or expanded to match the "duration" of the latents. This is to ensure that all latents are carrying 
information. Another approach would be to use the actual duration of the inputs (scaled so that the maximum duration 
is equal or less than the "duration" of the latents), but then the last (or the first) latents would be left out
of the sequence (positionally). We could follow this approach and maybe mask the unused tokens
- BLOCK 25 is only in test set. Train blocks:  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24}
Test blocks: {8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25}. Heldout blocks: {1, 2, 3, 4, 6, 7}
- We may need to add the Encoder or EncoderLayer to the no split module of llama if there are any kind of residual connections,
this is to ensure that all the tensors are in the same rank for every operation in the module
- hf from_pretrained has the name/directory as args, *model_args to pass to the model __init__ method, and all the rest are **kwargs
hf save_pretrained has directory as arg and all the rest are **kwargs. peft load_adapter as directory and adapter name as args, and all the rest are **kwargs. 
We only need to take care of the is_trainable kwarg, so no more kwargs are passed.
- The transformer part of the decoder cannot be named model as in the original llama code because modules 
wrapped in peft have an attribute "model" that references the unwrapped model, so we would have to add
logic to call decoder.model.model only when the peft adapter is loaded
- Probably this is the same problem that causes resize_token_embeddings to fail when called on BCI model
instead of on the decoder
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
