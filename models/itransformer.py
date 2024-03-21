import os
from typing import List, Union, Optional
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torchvision.ops import MLP

from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign()

from utils.config_utils import DictConfig, update_config
from models.model_output import ModelOutput
from models.masker import Masker
DEFAULT_CONFIG = "configs/itransformer.yaml"


@dataclass
class iTransformerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    mask: Optional[torch.LongTensor] = None
    preds: Optional[torch.FloatTensor] = None
    targets: Optional[torch.FloatTensor] = None


class AverageTokens(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.sum(dim=self.dim)


class iTransformerEncoder(nn.Module):

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()

        self.embed = MLP(
            in_channels=config.max_n_bins, 
            hidden_channels=[config.hidden_size, config.hidden_size],
            activation_layer=ACT2FN[config.activation].__class__,
            bias=config.bias,
            dropout=config.dropout,
        )

        transformer_layer = TransformerEncoderLayer(
            d_model = config.hidden_size,
            nhead = config.n_heads,
            dim_feedforward = 4*config.hidden_size,
            activation=ACT2FN[config.activation],
            # bias=config.bias,
            dropout=config.dropout,
            batch_first=True,
        )

        self.transformer = TransformerEncoder(
            encoder_layer = transformer_layer,
            num_layers = config.n_layers,
            norm = nn.LayerNorm(config.hidden_size),
            enable_nested_tensor = True,
        )

        self.project = MLP(
            in_channels=config.hidden_size, 
            hidden_channels=[config.hidden_size, config.max_n_bins],
            activation_layer=ACT2FN[config.activation].__class__,
            bias=config.bias,
            dropout=config.dropout,
        )

    def forward(
        self, 
        spikes: torch.LongTensor,  # (batch, max_n_bins, n_channels)
    ) -> torch.FloatTensor:   # (batch, n_channels, hidden_size)

        tokens = self.embed(spikes.transpose(1,2))           # (batch, n_channels, hidden_size)
        # torch.save(tokens,"t.pth")
        x = self.transformer(tokens)              # (batch, n_channels, hidden_size)

        return x




class iTransformer(nn.Module):

    def __init__(
        self, 
        config: DictConfig,
        **kwargs,
    ):
        super().__init__()
        self.method = kwargs["method_name"]
        
        config = update_config(DEFAULT_CONFIG, config)

        # Build masker
        self.mask = config.masker.active
        if self.mask:
            self.masker = Masker(config.masker)
        
        # Build encoder
        encoder_pt_path = config["encoder"].pop("from_pt", None)
        if encoder_pt_path is not None:
            encoder_config = torch.load(os.path.join(encoder_pt_path, "encoder_config.pth"))
            config["encoder"] = update_config(config.encoder, encoder_config)
        self.encoder = iTransformerEncoder(config.encoder)

        # Load encoder weights
        if encoder_pt_path is not None:
            self.encoder.load_state_dict(torch.load(os.path.join(encoder_pt_path,"encoder.bin")))


        # Build decoder
        decoder_pt_path = config["decoder"].pop("from_pt", None)
        if decoder_pt_path is not None:
            decoder_config = torch.load(os.path.join(decoder_pt_path, "decoder_config.pth"))
            config["decoder"] = update_config(config.decoder, decoder_config)

        
        if self.method == "mlm":
            assert config.masker.active, "Can't pretrain with inactive masking"
            n_outputs = config.encoder.max_n_bins
        elif self.method == "ctc":
            n_outputs = kwargs["vocab_size"] * config.encoder.max_n_bins
            self.output_shape = (config.encoder.max_n_bins, kwargs["vocab_size"])
        else:
            raise Exception(f"Method {self.method} not implemented")


        decoder_layers = []

        if self.method == "ctc":
            decoder_layers.append(AverageTokens(dim=1)) # Get rid of the channel dimension
    
        if config.decoder.mlp_decoder:
            decoder_layers.append(nn.Linear(config.encoder.hidden_size, config.encoder.hidden_size))
            decoder_layers.append(ACT2FN[config.decoder.activation])
        decoder_layers.append(nn.Linear(config.encoder.hidden_size, n_outputs))

        if self.method == "mlm" and not kwargs["log_input"]:
            decoder_layers.append(nn.ReLU()) # If we're not using lograte, we need to feed positive rates
        if self.method == "ctc":
            decoder_layers.append(nn.LogSoftmax(dim=-1))  # CTC loss receives log-softmax-normalized logits
        self.decoder = nn.Sequential(*decoder_layers)

        # Load decoder weights
        if decoder_pt_path is not None:
            self.decoder.load_state_dict(torch.load(os.path.join(decoder_pt_path,"decoder.bin")))


        # Build loss function
        if self.method == "mlm":
            self.loss_name = kwargs["loss"]
            self.log_input = kwargs["log_input"]
            if self.loss_name == "poisson_nll":
                self.loss_fn = nn.PoissonNLLLoss(reduction="none", log_input=self.log_input)
            elif self.loss_name == "mse":
                self.loss_fn = nn.MSELoss(reduction="none")
            else:   
                raise Exception(f"Loss {kwargs['loss']} not implemented yet for mlm")
        elif self.method == "ctc":
            self.loss_fn = nn.CTCLoss(reduction="none", blank=kwargs["blank_id"], zero_infinity=kwargs["zero_infinity"])
        
        # Save config
        self.config = config

    def forward(
        self,
        spikes:           torch.FloatTensor,  # (bs, seq_len, n_channels)
        spikes_mask:      torch.LongTensor,   # (bs, seq_len)
        spikes_lengths:   torch.LongTensor,   # (bs)
        targets:          Optional[torch.FloatTensor] = None,  # (bs, tar_len) 
        targets_lengths:  Optional[torch.LongTensor] = None,   # (bs)
        block_idx:        Optional[torch.LongTensor] = None,   # (bs)
        date_idx:         Optional[torch.LongTensor] = None,   # (bs)
    ) -> iTransformerOutput:

        if self.method == "mlm":
            assert targets is None, "No targets needed for ssl"
            targets = spikes.clone()
        
        # Encode neural data. x is the masked embedded spikes. targets_mask is True for masked bins
        if self.mask:
            spikes, targets_mask = self.masker(spikes)
        else:
            targets_mask = torch.zeros_like(spikes, dtype=torch.int64)
        x = self.encoder(spikes)    # (batch, n_channels, hidden_size)

        # Predict rates/ctc-logits from embeddedings
        preds = self.decoder(x)    # (bs, n_channels, max_n_bins) / (bs, max_n_bins*vocab_size)

        if self.method == "mlm":
            preds = preds.transpose(1,2)
            # Include padding in mask
            targets_mask = targets_mask & spikes_mask.unsqueeze(2) 
            # Compute the loss only over masked timesteps that are not padded 
            loss = (self.loss_fn(preds, targets) * targets_mask).sum()
            n_examples = targets_mask.sum()

            return iTransformerOutput(
                loss=loss,
                n_examples=n_examples,
                preds=preds,
                targets=targets,
                mask=targets_mask,
            )

        elif self.method == "ctc":
            preds = preds.view(preds.shape[:1] + self.output_shape)
            loss = self.loss_fn(log_probs=preds.transpose(0,1), targets=targets, input_lengths=spikes_lengths, target_lengths=targets_lengths).sum()
            n_examples = torch.tensor(len(targets), device=loss.device, dtype=torch.long)

            return iTransformerOutput(
                loss=loss,
                n_examples=n_examples,
                preds=preds,
                targets=targets,
            )


    def save_checkpoint(self, save_dir):
        torch.save(self.encoder.state_dict(), os.path.join(save_dir,"encoder.bin"))
        torch.save(dict(self.config.encoder), os.path.join(save_dir,"encoder_config.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir,"decoder.bin"))
        torch.save(dict(self.config.decoder), os.path.join(save_dir,"decoder_config.pth"))

    def load_checkpoint(self, load_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(load_dir,"encoder.bin")))
        self.decoder.load_state_dict(torch.load(os.path.join(load_dir,"decoder.bin")))