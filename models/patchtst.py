from typing import List, Union, Optional
from dataclasses import dataclass

import torch
from torch import nn

from transformers import PatchTSTConfig, PatchTSTModel
from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from models.model_output import ModelOutput
from utils.config_utils import update_config, DictConfig



DEFAULT_CONFIG = "configs/patchtst.yaml"

@dataclass
class PatchTSTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None

class PredictHead(nn.Module):

    def __init__(
        self, 
        config: DictConfig, 
        num_input_channels: int, 
        d_model: int,
        patch_length: int,
        **kwargs,
    ):
        super().__init__()

        self.vocab_size = kwargs["vocab_size"]
        self.num_input_channels = num_input_channels
        self.share_projection = config.share_projection
        self.pooling_type = config.pooling_type
        self.mlp_decoder = config.mlp_decoder


        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        if not self.share_projection:
            self.projections = nn.ModuleList()
            for i in range(self.num_input_channels):
                self.projections.append(
                    nn.Linear(d_model, self.vocab_size) if not self.mlp_decoder else \
                    nn.Sequential(
                        nn.Linear(d_model, d_model),
                        ACT2FN[config.mlp_activation],
                        nn.Linear(d_model, self.vocab_size),
                    )
                )
        else:
            # all the channels share the same head
            self.projection = nn.Linear(d_model, self.vocab_size) if not self.mlp_decoder else \
                    nn.Sequential(
                        nn.Linear(d_model, d_model),
                        ACT2FN[config.mlp_activation],
                        nn.Linear(d_model, self.vocab_size),
                    )
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self, 
        embedding: torch.FloatTensor    # (bs, num_input_channels, num_patches, d_model)
    ) -> torch.FloatTensor:             # (bs, num_patches, vocab_size)

        if not self.share_projection:
            output = []
            for i in range(self.num_input_channels):
                embedding_i = self.dropout(embedding[:,i,:,:])
                embedding_i = self.projections[i](embedding_i)  # (bs, num_patches, vocab_size)
                output.append(embedding_i)
            output = torch.stack(output, dim=1)  # (bs, num_input_channels, num_patches, vocab_size)
            if self.pooling_type == "mean":
                pooled_embedding = embedding.mean(dim=1)    # (bs, num_patches, d_model)
            elif self.pooling_type == "max":
                pooled_embedding = embedding.max(dim=1)     # (bs, num_patches, d_model)
        else:
            if self.pooling_type == "mean":
                pooled_embedding = embedding.mean(dim=1)    # (bs, num_patches, d_model)
            elif self.pooling_type == "max":
                pooled_embedding = embedding.max(dim=1)     # (bs, num_patches, d_model)
            pooled_embedding = self.dropout(pooled_embedding)
            output = self.projection(pooled_embedding)      # (bs, num_patches, vocab_size)

        return self.softmax(output)                         # (bs, num_patches, vocab_size)


class PretrainHead(nn.Module):

    def __init__(
        self, 
        config: DictConfig,
        num_input_channels: int, 
        d_model: int,
        patch_length: int,
        **kwargs,
    ):

        super().__init__()


        self.num_input_channels = num_input_channels
        self.share_projection = config.share_projection
        self.mlp_decoder = config.mlp_decoder
        self.use_lograte = config.use_lograte

        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        if not self.share_projection:
            self.projections = nn.ModuleList()
            for i in range(self.num_input_channels):
                self.projections.append(
                    nn.Linear(d_model, patch_length) if not self.mlp_decoder else \
                    nn.Sequential(
                        nn.Linear(d_model, d_model),
                        ACT2FN[config.mlp_activation],
                        nn.Linear(d_model, patch_length),
                    )
                )
        else:
            # all the channels share the same head
            self.projection = nn.Linear(d_model, patch_length) if not self.mlp_decoder else \
                    nn.Sequential(
                        nn.Linear(d_model, d_model),
                        ACT2FN[config.mlp_activation],
                        nn.Linear(d_model, patch_length),
                    )
        self.post_proj = nn.ReLU() if not self.use_lograte else nn.Identity()


    def forward(
        self, 
        embedding: torch.FloatTensor,   # (bs, num_input_channels, num_patches, d_model)
    ) -> torch.Tensor:                  # (bs, num_input_channels, num_patches, patch_size)
       
        if not self.share_projection:
            output = []
            for i in range(self.num_input_channels):
                embedding_i = self.dropout(embedding[:,i,:,:])
                embedding_i = self.projections[i](embedding_i)  # (bs, num_patches, patch_size)
                output.append(embedding_i)
            output = torch.stack(output, dim=1)                 # (bs, num_input_channels, num_patches, patch_size)
        else:
            output = self.projection(self.dropout(embedding))   # (bs, num_input_channels, num_patches, patch_size)

        return self.post_proj(output)   # (bs, num_input_channels, num_patches, patch_size)


METHOD2HEAD = {"ctc": PredictHead, "ssl": PretrainHead}

class PatchTSTForSpikingActivity(nn.Module):

    def __init__(
        self, 
        config: DictConfig,
        **kwargs,
    ):
        super().__init__()

        config = update_config(DEFAULT_CONFIG, config)
        self.method = kwargs["method_name"]
        
        # Build encoder
        encoder_pt_path = config["encoder"].pop("from_pt", None)
        if encoder_pt_path is not None:
            encoder_config = os.path.join(encoder_pt_path, "encoder_config.yaml")
            config["encoder"] = update_config(config.encoder, encoder_config)

        self.encoder = PatchTSTModel(PatchTSTConfig.from_dict(config.encoder))
        
        if encoder_pt_path is not None:
            self.encoder.load_state_dict(torch.load(os.path.join(encoder_pt_path,"encoder.bin")))


        # Build decoder
        if config.decoder.from_pt is not None:
            decoder_pt_path = config.decoder.from_pt
            decoder_config = os.path.join(decoder_pt_path , "decoder_config.yaml")
            config["decoder"] = update_config(config.decoder, decoder_config)
        else:
            decoder_pt_path = None
        decoder_class = METHOD2HEAD[self.method]
        self.decoder = decoder_class(config.decoder, config.encoder.num_input_channels, config.encoder.d_model, config.encoder.patch_length, **kwargs)
        
        # Load decoder weights
        if decoder_pt_path is not None:
            self.decoder.load_state_dict(torch.load(os.path.join(decoder_pt_path,"decoder.bin")))


        # Build loss function
        if self.method == "ssl":
            if kwargs["loss"] == "poisson_nll":
                self.loss_fn = nn.PoissonNLLLoss(reduction="none", log_input=kwargs["use_lograte"])
            else:   
                raise Exception(f"Loss {kwargs['loss']} not implemented yet for ssl")
        elif self.method == "ctc":
            self.loss_fn = nn.CTCLoss(reduction="none", blank=kwargs["blank_id"], zero_infinity=kwargs["zero_infinity"])


    def forward(
        self,
        spikes: torch.LongTensor,                           # (bs, seq_len, n_input_channels)
        targets: Optional[torch.Tensor] = None,             # (bs, seq_len)
        targets_lengths: Optional[torch.LongTensor] = None, # (bs)
    ) -> PatchTSTOutput:

        outputs = self.encoder(spikes)   
        mask = outputs.mask
        patch_input = outputs.patch_input      
        embedding = outputs.last_hidden_state   # (bs, n_input_channels, num_patches, d_model)
        logits = self.decoder(embedding)        # (bs, num_patches, vocab_size) or (bs, num_input_channels, num_patches, patch_size)
        

        # Compute the loss over unmasked outputs
        if self.method == "ssl":
            loss = (self.loss_fn(outputs, patch_input) * mask).sum()
            n_examples = mask.sum()
        elif self.method == "ctc":
            loss = self.loss_fn(logits.transpose(0,1), targets, target_lengths=targets_lengths)
            n_examples = torch.Tensor(len(targets)).to(loss.device, torch.long)

        return PatchTSTOutput(
            loss=loss,
            n_examples=n_examples,
        )