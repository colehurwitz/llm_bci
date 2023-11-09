import torch
import torch.nn as nn

from models.neural_encoder import NeuralEncoder

from utils.config_utils import DictConfig

    

class NeuralEncoderPretrainer(nn.Module):

    def __init__(self, encoder: NeuralEncoder, config: DictConfig):
        super().__init__()


        self.encoder = encoder
        self.rate_dropout = nn.Dropout(config.rate_dropout)

        # Build decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(self.encoder.hidden_size, config.inter_dim))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(config.inter_dim, num_neurons))
        if not config.use_lograte:
            decoder_layers.append(nn.ReLU()) # If we're not using lograte, we need to feed positive ratesw
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.classifier = nn.PoissonNLLLoss(reduction='none', log_input=config.use_lograte)

    def forward(
            self, 
            features:           torch.FloatTensor,  # (batch_size, fea_len, n_channels)
            features_mask:      torch.FloatTensor,  # (batch_size, fea_len)
            features_timestamp: torch.LongTensor,   # (batch_size, fea_len)
            block_idx:          torch.LongTensor,   # (batch_size, fea_len)
            date_idx:           torch.LongTensor,   # (batch_size, fea_len)
        ) -> torch.FloatTensor:                     # (1,)
        