from copy import deepcopy
from typing import List, Optional, Tuple, Dict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign

from utils.config_utils import DictConfig

# Create buffer of biggest possible context mask 
def create_context_mask(context_forward, context_backward, max_F) -> torch.LongTensor: # (max_fea_len, max_fea_len)

        if context_forward == -1 and context_backward == -1:
            return torch.ones(max_F, max_F).to(torch.int64)

        context_forward = context_forward if context_forward >= 0 else max_F
        context_backward = context_backward if context_backward >= 0 else max_F
        mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_forward).to(torch.int64)).transpose(0, 1)
        if context_backward > 0:
            back_mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_backward).to(torch.int64))
            mask = mask & back_mask
        return mask

# Copied from hf Llama
# Precompute cos and sin for RoPE
def get_cos_sin(dim, max_latent_positions, base=10000, dtype=torch.get_default_dtype(), device=None):

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        t = torch.arange(max_latent_positions, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

# Rotates half the hidden dims of the input.
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), -1)

# Applies RoPE to the query and key tensors.
def apply_rotary_pos_emb(q, k, pos_ids, cos, sin, unsqueeze_dim=1):

    cos = cos[pos_ids].unsqueeze(unsqueeze_dim)
    sin = sin[pos_ids].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    
    return q_embed, k_embed



# Mask features
class Masker(nn.Module):

    def __init__(self, embed_mode, config: DictConfig):
        super().__init__()

        self.mode = config.mode
        self.ratio = config.ratio
        self.zero_ratio = config.zero_ratio
        self.random_ratio = config.random_ratio
        self.expand_prob = config.expand_prob
        self.max_timespan = config.max_timespan
        self.embed_mode = embed_mode

    def forward(
        self, 
        features: torch.FloatTensor,                    # (batch_size, fea_len, n_channels)
    ) -> Tuple[torch.FloatTensor,torch.LongTensor]:     # (batch_size, fea_len, n_channels), (batch_size, fea_len, n_channels)

        mask_ratio = deepcopy(self.ratio)

        # Expand mask
        if self.mode == "timestep":
            if torch.bernoulli(torch.tensor(self.expand_prob).float()):
                timespan = torch.randint(1, self.max_timespan+1, (1, )).item() 
            else:
                timespan = 1
            mask_ratio = mask_ratio/timespan

        # Get masking probabilities
        if self.mode == "full":
            mask_probs = torch.full(features.shape, mask_ratio)     # (batch_size, fea_len, n_channels)
        elif self.mode == "timestep":
            mask_probs = torch.full(features[:, :, 0].shape, mask_ratio) # (batch_size, fea_len)
        elif self.mode == "neuron":
            mask_probs = torch.full(features[:, 0].shape, mask_ratio)    # (batch_size, n_channels)
        else:
            raise Exception(f"Masking mode {self.mode} not implemented")
        
        # Create mask
        mask = torch.bernoulli(mask_probs).to(features.device)

        # Expand mask
        if self.mode == "timestep":
            mask = self.expand_timesteps(mask, timespan)
            mask = mask.unsqueeze(2).expand_as(features).bool()    # (batch_size, fea_len, n_channels)
        elif self.mode == "neuron":
            mask = mask.unsqueeze(1).expand_as(features).bool()    # (batch_size, fea_len, n_channels)
        
        # Mask data
        zero_idx = torch.bernoulli(torch.full(features.shape, self.zero_ratio)).to(features.device).bool() & mask
        features[zero_idx] = 0
        random_idx = torch.bernoulli(torch.full(features.shape, self.random_ratio)).to(features.device).bool() & mask & ~zero_idx
        random_spikes = (features.max() * torch.rand(features.shape, device=features.device) )
        if self.embed_mode == "embed":
            random_spikes = random_spikes.to(torch.int64)
        elif self.embed_mode == "identity":
            random_spikes = random_spikes.round()
        else:
            random_spikes = random_spikes.float()

        features[random_idx] = random_spikes[random_idx]

        return features, mask.to(torch.int64)

    @staticmethod
    def expand_timesteps(mask, width=1):
        kernel = torch.ones(width, device=mask.device).view(1, 1, -1)
        expanded_mask = F.conv1d(mask.unsqueeze(1), kernel, padding="same")
        return (expanded_mask.squeeze(1) >= 1)
        

# Normalize and add noise
class NormAndNoise(nn.Module): 

    def __init__(self, input_size, config):
        super().__init__()
        self.normalize = config.normalize
        if self.normalize:
            if config.norm == "layernorm":
                self.norm = nn.LayerNorm(input_size)
            elif config.norm == "scalenorm":
                self.norm = ScaleNorm(input_size ** 0.5)
            elif config.norm == "zscore":
                self.norm = None
            else:
                raise Exception(f"Norm layer {config.norm} not implemented")
        self.eps = config.eps
        self.white_noise_sd = config.white_noise_sd
        self.constant_offset_sd = config.constant_offset_sd

    def forward(self, features):
        B, T, N = features.size()

        if self.normalize:  
            if self.norm is None:
                features = (features - features.mean(-1).unsqueeze(-1)) / (features.std(-1).unsqueeze(-1) + self.eps)
            else:
                features = self.norm(features)

        if self.white_noise_sd is not None:
            features += self.white_noise_sd*torch.randn(B,T,N, dtype=features.dtype, device=features.device)

        if self.constant_offset_sd is not None:
            features += self.constant_offset_sd*torch.randn(B,1,N, dtype=features.dtype, device=features.device)

        return features


# Embed and stack
class NeuralEmbeddingLayer(nn.Module):

    def __init__(self, hidden_size, config: DictConfig):
        super().__init__()

        self.adapt = config.adapt
        self.bias = config.mlp_bias
        

        if self.adapt:
             # One embedding layer for each day
            if config.mode == "linear":
                self.embed_spikes = nn.ModuleList([
                    nn.Linear(config.n_channels, config.n_channels * config.mult, bias=config.bias) 
                for i in range(config.n_dates)])

            elif config.mode == "embed":
                self.embed_spikes = nn.ModuleList([
                    nn.Sequential(
                        nn.Embedding(config.max_spikes, config.mult),
                        nn.Flatten(start_dim=-2)
                    )
                for i in range(config.n_dates)])
            else:
                raise Exception(f"Embedding mode {config.mode} cannot be adaptative")
        else:
            # One common embedding layer
            if config.mode == "linear":
                self.embed_spikes = nn.Linear(config.n_channels, config.n_channels * config.mult, bias=config.bias)
            elif config.mode == "embed":
                self.embed_spikes = nn.Sequential(
                    nn.Embedding(config.max_spikes, config.mult),
                    nn.Flatten(start_dim=-2)
                )
            elif config.mode == "identity":
                self.embed_spikes = nn.Identity()
            else:
                raise Exception(f"Invalid embed mode {config.mode}.")

        if config.mode == "embed" and config.fixup_init:
            self.fixup_initialization(config.init_range, config.spike_log_init, config.max_spikes, adapt=self.adapt)

        # Stacking
        self.stack = config.stack.active
        if self.stack:
            self.stack_size = config.stack.size
            self.stack_stride = config.stack.stride
            self.stacking = nn.Unfold(kernel_size=(config.stack.size, config.n_channels),stride=(config.stack.stride,1))
            self.stacking_mask = nn.Unfold(kernel_size=(config.stack.size, 1),stride=(config.stack.stride,1))
            self.stack_projection = nn.Linear(config.n_channels*config.mult*config.stack.size,hidden_size)

        # Activation after embedding
        self.act = ACT2FN[config.act] if config.act != "identity" else nn.Identity()

        # Embedding scale
        self.scale = hidden_size ** 0.5 if config.scale == None else config.scale

        # Embed postion
        self.pos = config.pos
        if self.pos:
            self.embed_pos = nn.Embedding(config.max_F, hidden_size)

        # Regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(
            self, 
            features:           torch.FloatTensor,      # (batch_size, fea_len, n_channels)
            features_mask:      Optional[torch.LongTensor],          # (batch_size, fea_len)
            features_timestamp: Optional[torch.LongTensor],          # (batch_size, fea_len)
            block_idx:          Optional[torch.LongTensor] = None,   # (batch_size)
            date_idx:           Optional[torch.LongTensor] = None,   # (batch_size)
        ) -> Tuple[torch.FloatTensor,torch.LongTensor,torch.LongTensor]:   # (batch_size, new_fea_len, hidden_size),  (batch_size, new_fea_len), (batch_size, new_fea_len)

        # Embed spikes
        if self.adapt:
            x = torch.stack([self.embed_spikes[date_idx[i]](f) for i, f in enumerate(features)], 0)
        else:
            x = self.embed_spikes(features)

        # Rescaling
        x = self.act(x) * self.scale

        # Stacking
        if self.stack:
            x = self.stack_projection(self.stacking(x.unsqueeze(1)).transpose(1,2))
            features_timestamp = features_timestamp[:,:x.size(1)] # keep the first positions
            features_mask = self.stacking_mask(features_mask.unsqueeze(-1).unsqueeze(1).float()).transpose(1,2).prod(-1).to(features_mask.dtype) # unmask only features tha come from unmasked features

        # Embed position
        if self.pos:
            x += self.embed_pos(features_timestamp)

        return self.dropout(x), features_mask, features_timestamp


    # Compute new lens after stacking
    def get_stacked_lens(self, lens):
        return lens if not self.stack else (1 + (lens - self.stack_size) / self.stack_stride).to(lens.dtype)

    # Initialization methods copied from NDT
    def fixup_initialization(self, init_range, spike_log_init, max_spikes, adapt):
        if adapt:
            for i in range(len(self.embed_spikes)):
                if spike_log_init:
                    # Use a log scale, since we expect spike semantics to follow compressive distribution
                    log_scale = torch.arange(1, max_spikes+1).float().log() # 1 to lg
                    log_scale = (log_scale - log_scale.mean()) / (log_scale[-1] - log_scale[0])
                    log_scale = log_scale * init_range
                    # Add some noise
                    self.embed_spikes[i][0].weight.data.uniform_(-init_range / 10, init_range / 10)
                    self.embed_spikes[i][0].weight.data += log_scale.unsqueeze(1).expand_as(self.embed_spikes[i][0].weight.data)
                else:
                    self.embed_spikes[i][0].weight.data.uniform_(-init_range, init_range)
        else:
            if spike_log_init:
                # Use a log scale, since we expect spike semantics to follow compressive distribution
                log_scale = torch.arange(1, max_spikes+1).float().log() # 1 to lg
                log_scale = (log_scale - log_scale.mean()) / (log_scale[-1] - log_scale[0])
                log_scale = log_scale * init_range
                # Add some noise
                self.embed_spikes[0].weight.data.uniform_(-init_range / 10, init_range / 10)
                self.embed_spikes[0].weight.data += log_scale.unsqueeze(1).expand_as(self.embed_spikes[0].weight.data)
            else:
                self.embed_spikes[0].weight.data.uniform_(-init_range, init_range)


# MLP
class NeuralMLP(nn.Module):

    def __init__(self, hidden_size, inter_size, act, use_bias, dropout):
        super().__init__()

        self.up_proj    = nn.Linear(hidden_size, inter_size, bias=use_bias)
        self.act        = ACT2FN[act]
        self.down_proj  = nn.Linear(inter_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        x = self.act(self.up_proj(x))
        return self.dropout(self.down_proj(x))



# Attention module.
class NeuralAttention(nn.Module):

    def __init__(self, idx, hidden_size, n_heads, use_bias, dropout, use_rope=False, base=10000.):
        super().__init__()
        
        self.idx = idx

        # Architecture config
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        assert self.hidden_size % self.n_heads == 0, f"Hidden dim is not multiple of head size"
        self.head_size = self.hidden_size // self.n_heads
        self.use_rope = use_rope

        # Attention parameters
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.value  = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)

        # Flash attention
        # torch.backends.cuda.enable_flash_sdp(True)
        self.flash_attention = partial(torch.nn.functional.scaled_dot_product_attention, dropout_p=dropout, is_causal=False)

        # Final projection
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)


        # RoPE parameters
        if use_rope:
            cos, sin = get_cos_sin(self.head_size, max_n_latents, base=base, dtype=self.query.weight.dtype, device=self.query.weight.device)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,       
        x:          torch.FloatTensor,                      # (batch_size, fea_len, hidden_size)
        attn_mask:  torch.LongTensor,                       # (batch_size, fea_len, fea_len)
        timestamp:  Optional[torch.LongTensor] = None,      # (batch_size, fea_len)
    ) -> torch.FloatTensor:                                 # (batch_size, fea_len, hidden_size)

        B, F, _  = x.size()     # batch size and fea len

        # Create batched bool attention mask 
        assert attn_mask.max() == 1 and attn_mask.min() == 0, ["assertion", attn_mask.max(), attn_mask.min()]
        attn_mask = attn_mask.unsqueeze(1).expand(B,self.n_heads,F,F).bool()            # (B,n_heads,F,F)
        
        # Compute query, key, value for attention
        q = self.query(x).view(B, F, self.n_heads, self.head_size).transpose(1, 2)      #(B,n_heads,F,head_size)
        k = self.key(x).view(B, F, self.n_heads, self.head_size).transpose(1, 2)        #(B,n_heads,F,head_size)
        v = self.value(x).view(B, F, self.n_heads, self.head_size).transpose(1, 2)      #(B,n_heads,F,head_size)

        # Apply rotations to encode relative positions
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, timestamp, self.cos, self.sin, 1)  # (B,n_heads,F,head_size)

        # Compute attention efficiently
        out = self.flash_attention(q, k, v, attn_mask=attn_mask)                 # (B,n_heads,F,head_size)
        out = out.transpose(1, 2).contiguous().view(B,F, self.hidden_size)       # (B, F, hidden_size)

        return self.out_proj(self.dropout(out)) # (B, F, hidden_size)

    
    

# Encoder layer: bidirectional self-attention + mlp
class NeuralEncoderLayer(nn.Module):
    
    def __init__(self, idx, config: DictConfig):
        super().__init__()

        self.idx = idx
        
        # Architecture config
        self.use_rope = config.use_rope

        # Encoder block
        self.ln1 = ScaleNorm(config.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(config.hidden_size, bias=config.norm_bias) 
        self.attn = NeuralAttention(idx, config.hidden_size, config.n_heads, config.attention_bias, config.dropout, config.use_rope, config.rope_theta)
        self.ln2 = ScaleNorm(config.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(config.hidden_size, bias=config.norm_bias) 
        self.mlp = NeuralMLP(config.hidden_size, config.inter_size, config.act, config.mlp_bias, config.dropout)

        if config.fixup_init:
            self.fixup_initialization(config.n_layers)

    def forward(
        self, 
        x:          torch.FloatTensor,                  # (batch_size, fea_len, hidden_size)
        attn_mask:  torch.LongTensor,                   # (batch_size, fea_len, fea_len)
        timestamp:  Optional[torch.LongTensor] = None,  # (batch_size, fea_len)          
    ) -> torch.FloatTensor :                            # (batch_size, fea_len, hidden_size)
        
        # LN -> Attention -> Residual connectiob
        x = x + self.attn(self.ln1(x), attn_mask, timestamp if self.use_rope else None)

        # LN -> MLP -> Residual connection
        x = x + self.mlp(self.ln2(x))

        return x

    def fixup_initialization(self, n_layers):
        temp_state_dic = {}
        for name, param in self.named_parameters():
            if name.endswith("_proj.weight"):
                temp_state_dic[name] = (0.67 * (n_layers) ** (- 1. / 4.)) * param
            elif name.endswith("value.weight"):
                temp_state_dic[name] = (0.67 * (n_layers) ** (- 1. / 4.)) * (param * (2**0.5))


        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)   





class NeuralFactorsProjection(nn.Module):

    def __init__(self, hidden_size, config):
        
        super().__init__()
        
        self.out_size = config.size if config.project_to_factors else hidden_size
        # self.out_space = "factors" if config.project_to_factors else "hidden"
        
        self.dropout = nn.Dropout(config.dropout)

        if config.project_to_factors:
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, config.size, config.bias),
                ACT2FN[config.act]
            )
            # Renitialize weights
            if config.fixup_init:
                self.proj[0].weight.data.uniform_(-config.init_range, config.init_range)
                if config.bias:
                    self.proj[0].bias.data.zero_()
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(self.dropout(x))
        



# Encoder for time binned neural data
class NeuralEncoder(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        
        self.int_features = config.embedder.mode == "embed"
        self.hidden_size = config.transformer.hidden_size
        self.n_layers = config.transformer.n_layers

        # Masker
        self.mask = config.masker.active
        if self.mask:
            self.masker = Masker(config.embedder.mode, config.masker)

        # Context span mask
        context_mask = create_context_mask(config.context.forward, config.context.backward, config.embedder.max_F)
        self.register_buffer("context_mask", context_mask, persistent=False)

        # Normalization and noising layer
        self.norm_and_noise = NormAndNoise(config.n_channels, config.norm_and_noise)

        # Embedding layer
        self.embedder = NeuralEmbeddingLayer(self.hidden_size, config.embedder)

        # Transformer
        self.layers = nn.ModuleList([NeuralEncoderLayer(idx, config.transformer) for idx in range(self.n_layers)])
        self.out_norm = ScaleNorm(self.hidden_size ** 0.5) if config.transformer.use_scalenorm else nn.LayerNorm(self.hidden_size) 
       
        # Out projection
        self.out_proj = NeuralFactorsProjection(self.hidden_size, config.factors)



    def forward(
            self, 
            features:           torch.FloatTensor,  # (batch_size, fea_len, n_channels)
            features_mask:      torch.LongTensor,   # (batch_size, fea_len)
            features_timestamp: torch.LongTensor,   # (batch_size, fea_len)
            block_idx:          Optional[torch.LongTensor] = None,   # (batch_size)
            date_idx:           Optional[torch.LongTensor] = None,   # (batch_size)
        ) -> torch.FloatTensor:                     # (batch_size, fea_len, hidden_size)
        
        B, F, N = features.size() # batch size, fea len, n_channels
        
        if self.int_features:
            features = features.to(torch.int64)
        
        # Normalize across channels and add noise
        features = self.norm_and_noise(features)

        # Mask neural data
        if self.mask:
            features, targets_mask = self.masker(features)
            targets_mask = targets_mask & features_mask.unsqueeze(-1).expand(B,F,N)
        else:
            targets_mask = None

        # Embed neural data
        x, features_mask, features_timestamp = self.embedder(features, features_mask, features_timestamp, block_idx, date_idx)

        _, F, _ = x.size() # feature len may have changed after stacking

        # Prepare 
        context_mask = self.context_mask[:F,:F].to(x.device).unsqueeze(0).expand(B,F,F)
        features_mask = features_mask.unsqueeze(1).expand(B,F,F)
        self_mask = torch.eye(F).to(x.device, torch.int64).expand(B,F,F) # hack so that even padded features attend to themselves and avoid attention issues
        attn_mask = self_mask | (context_mask & features_mask)
        
        # Forward transformer
        for idx, layer in enumerate(self.layers):
            x = layer(x, attn_mask=attn_mask, timestamp=features_timestamp)
        x = self.out_norm(x)

        return self.out_proj(x), targets_mask

    

class ScaleNorm(nn.Module):

    def __init__(self, scale, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm