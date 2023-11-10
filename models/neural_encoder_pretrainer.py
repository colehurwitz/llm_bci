import torch
import torch.nn as nn

from models.neural_encoder import NeuralEncoder

from utils.config_utils import DictConfig

    

class NeuralEncoderPretrainer(nn.Module):

    def __init__(self, encoder: NeuralEncoder, config: DictConfig):
        super().__init__()


        self.encoder = encoder
        self.rate_dropout = nn.Dropout(config.rate_dropout)
        n_channels = encoder.n_channels

        # Build decoder
        decoder_layers = []
        if config.multi_layer_decoder == False:
            decoder_layers.append(nn.Linear(self.encoder.hidden_size, n_channels))
        else:
            decoder_layers.append(nn.Linear(self.encoder.hidden_size, config.inter_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(config.inter_dim, n_channels))
        if not config.use_lograte:
            decoder_layers.append(nn.ReLU()) # If we're not using lograte, we need to feed positive rates
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.loss = nn.PoissonNLLLoss(reduction='none', log_input=config.use_lograte)

    def forward(
            self, 
            features:           torch.FloatTensor,  # (batch_size, fea_len, n_channels)
            features_mask:      torch.FloatTensor,  # (batch_size, fea_len)
            features_timestamp: torch.LongTensor,   # (batch_size, fea_len)
            block_idx:          torch.LongTensor,   # (batch_size, fea_len)
            date_idx:           torch.LongTensor,   # (batch_size, fea_len)
        ) -> torch.FloatTensor:                     # (1,)

        # encode neural data
        x = self.encoder(features, features_mask, features_timestamp, block_idx, date_idx)

        # dropout for regularization
        x = self.rate_dropout(x)

        # transform neural embeddings into rates
        log_rates = self.decoder(x)

        # compute the loss over the unmasked outputs
        masked_loss = self.loss(log_rates, features) * features_mask

        # sum the masked loss
        loss = masked_loss.sum()

        return (log_rates, loss)
        
