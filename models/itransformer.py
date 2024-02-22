import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torchvison.ops import MLP

from typing import List, Union


from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from models.trainer import ModelOutput
from utils.config_utils import update_config

DEFAULT_CONFIG = "configs/itransformer.yaml"

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
            activation_layer=ACT2FN[config.activation],
            bias=config.bias,
            dropout=config.dropout,
        )

        transformer_layer = TransformerEncoderLayer(
            d_model = config.hidden_size,
            nhead = config.n_heads,
            dim_feedforward = 4*config.hidden_size,
            activation=ACT2FN[config.activation],
            bias=config.bias,
            dropout=config.dropout,
            batch_first=True,
        )

        self.transformer = TransformerEncoder(
            encoder_layer = transformer_layer,
            num_layers = config.n_layers,
            norm = nn.LayerNorm,
            enable_nested_tensor = True,
        )

        self.project = MLP(
            in_channels=config.hidden_size, 
            hidden_channels=[config.hidden_size, config.max_n_bins],
            activation_layer=ACT2FN[config.activation],
            bias=config.bias,
            dropout=config.dropout,
        )

    def forward(
        self, 
        x: torch.LongTensor,  # (batch, max_n_bins, n_channels)
    ) -> torch.FloatTensor:   # (batch, n_channels, hidden_size)

        tokens = self.embed(x.transpose(1,2))           # (batch, n_channels, hidden_size)
        latents = self.transformer(tokens)              # (batch, n_channels, hidden_size)

        return latents




class iTransformer(nn.Module):

    def __init__(
        self, 
        config: DictConfig,
    ):
        super().__init__()

        config = update_config(DEFAULT_CONFIG, config)

        # Build encoder
        if config.encoder.from_pt is not None:
            encoder_pt_path = config.encoder.from_pt
            encoder_config = os.path.join(encoder_pt_path, "encoder_config.yaml")
            config["encoder"] = update_config(config.encoder, encoder_config)
        else:
            encoder_pt_path = None
        self.model = iTransformerModel(config.encoder)
        
        if encoder_pt_path is not None:
            self.model.load_state_dict(torch.load(os.path.join(encoder_pt_path,"encoder.bin")))

        # Build decoder
        if config.decoder.from_pt is not None:
            decoder_pt_path = config.decoder.from_pt
            decoder_config = os.path.join(decoder_pt_path , "decoder_config.yaml")
            config["decoder"] = update_config(config.decoder, decoder_config)
        else:
            decoder_pt_path = None
        
        if config.decoder.method == "ssl":
            n_outputs = config.encoder.max_n_bins
            self.output_shape = (config.encoder.max_n_bins)
        elif config.decoder.method == "sft":
            n_outputs = config.decoder.vocab_size * config.encoder.max_n_bins
        else:
            raise Exception(f"Method {config.decoder.method} not implemented")


        decoder_layers = []

        if config.decoder.method == "sft":
            decoder_layers.append(AverageTokens(dim=1)) # Get rid of the channel dimension
    
        if config.decoder.mlp_decoder:
            decoder_layers.append(nn.Linear(config.encoder.hidden_size, config.encoder.hidden_size))
            decoder_layers.append(ACT2FN[config.decoder.activation])
        decoder_layers.append(nn.Linear(config.encoder.hidden_size, n_outputs))

        if config.decoder.method == "ssf" and not config.decoder.use_lograte:
            decoder_layers.append(nn.ReLU()) # If we're not using lograte, we need to feed positive rates
        if config.decoder.method == "sft":
            decoder_layers.append(nn.LogSoftmax(dim=-1))  # CTC loss receives log-softmax-normalized logits
        self.decoder = nn.Sequential(*decoder_layers)

        # Load decoder weights
        if decoder_pt_path is not None:
            self.decoder.load_state_dict(torch.load(os.path.join(decoder_pt_path,"decoder.bin")))


        # Build loss function
        if config.decoder.method == "sst":
            self.loss_fn = nn.PoissonNLLLoss(reduction="none", log_input=config.decoder.use_lograte)
        elif config.decoder.method == "sft":
            self.loss_fn = nn.CTCLoss(reduction="none", blank=blank_id, zero_infinity=config.decoder.zero_infinity)


    def forward(
        self,
        x: torch.LongTensor,  # (batch, max_n_bins, n_channels)
    ) -> ModelOutput:

        latents = self.model(x)         # (batch, n_channels, hidden_size)
        otuputs = self.decoder(latents) # (batch, ) 

        return outputs