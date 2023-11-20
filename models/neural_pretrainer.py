import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Union, Tuple

from transformers.utils import ModelOutput

from models.neural_encoder import NeuralEncoder

from utils.config_utils import DictConfig


@dataclass
class NeuralPretrainerOutput(ModelOutput):
    outputs: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    targets_mask: Optional[torch.LongTensor] = None



class NeuralPretrainer(nn.Module):

    def __init__(self, encoder: NeuralEncoder, config: DictConfig, vocab_size: int = None, blank_id: int = None):
        super().__init__()


        self.encoder = encoder
        self.loss_type = config.loss.type
        self.reduction = config.loss.reduction

        if config.loss.type == "poisson":
            n_outputs = encoder.config.embedder.n_channels
        elif config.loss.type == "ctc":
            n_outputs = vocab_size
        else:
            raise Exception(f"Loss {config.loss.type} not implemented")

        # Build decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(self.encoder.out_proj.out_size, n_outputs))

        if config.loss.type == "poisson" and not config.use_lograte:
            decoder_layers.append(nn.ReLU()) # If we're not using lograte, we need to feed positive rates
        if config.loss.type == "ctc":
            decoder_layers.append(nn.LogSoftmax(dim=-1))  # CTC loss asks for log-softmax-normalized logits
        self.decoder = nn.Sequential(*decoder_layers)

        # Loss function
        if config.loss.type == "poisson":
            self.loss = nn.PoissonNLLLoss(reduction="none", log_input=config.use_lograte)
        elif config.loss.type == "ctc":
            self.loss = nn.CTCLoss(reduction="none", blank=blank_id)
        


    def forward(
            self, 
            features:           torch.FloatTensor,  # (batch_size, fea_len, n_channels)
            features_mask:      torch.LongTensor,   # (batch_size, fea_len)
            features_timestamp: torch.LongTensor,   # (batch_size, fea_len)
            targets:            Union[torch.LongTensor, torch.FloatTensor],   # (batch_size, tar_len)
            features_len:       Optional[torch.LongTensor] = None,   # (batch_size)
            targets_len:        Optional[torch.LongTensor] = None,   # (batch_size)
            block_idx:          Optional[torch.LongTensor] = None,   # (batch_size)
            date_idx:           Optional[torch.LongTensor] = None,   # (batch_size)
        ) -> NeuralPretrainerOutput:                    

        if self.loss_type == "poisson" and self.encoder.int_features:
            targets = targets.to(torch.int64)

        # Encode neural data
        x, targets_mask = self.encoder(features, features_mask, features_timestamp, block_idx, date_idx)

        # Transform neural embeddings into rates/logits
        outputs = self.decoder(x)

        
        # Compute the loss over unmasked outputs
        if self.loss_type == "poisson":
            loss = self.loss(outputs, targets) * targets_mask
            n_examples = targets_mask.sum()
        elif self.loss_type == "ctc":
            loss = self.loss(outputs.transpose(0,1), targets, features_len, targets_len)
            n_examples = len(features)

        # Reduce loss
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return NeuralPretrainerOutput(
            outputs=outputs,
            loss=loss,
            n_examples=n_examples,
            targets_mask=targets_mask,
        )
