First preprocess data:
    - python utils/preprocess.py

Launch finetuning script (model is not saved correctly in distributed):
    - accelerate launch --config_file deepspeed.yaml finetune.py --config_file kai_dirs.yaml
                        --kwargs k1=v1 k2=v2 

the config_file for finetuning updates the default_finetune_config.yaml, you can add only the 
fields you need (including nested fields, with dot notation, e.g. bci.neural_config.context_forward=1)
kai_dir.yaml contains the path to the relevant directories in kai's server


Launch inference script (still doesn't work dsitributed):
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
-CHECK CTC LOSS CUDNN REQURIEMENTS
-phoneme error rate in pretrain
-subwords instead of phonemes
- padding mode for gaussian smoothing
-disentangle relevant from irrelevant
-check weight initialization
- embedding gating and context
-visualize neural embeddings
-which part of the model is under LORA ?
-Using higher lr for encoder and lower for decoder?
-context span also for llama
-data augmentation by choosing subset of channels?
-penalization proportional to the frequency of the token
-generation strategy

OBS
-computing the phoneme error rate is 20 slower than the pretraining
- bins are 20 ms long
- WER in finetune.py is not very informative because the predictions are based in previous 
subtokens. -> maybe we want to use a BERT-like model instead of Llama?
-Max block and date index has to be specified in NeuralConfig, read from preprocess.py output
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
