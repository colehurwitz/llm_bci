from typing import List, Optional, Tuple, Dict
from functools import partial

import torch
import torch.nn as nn

from transformers.activations import ACT2FN

from utils.config_utils import DictConfig


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

# Embed
class NeuralEmbeddingLayer(nn.Module):

    def __init__(self, hidden_size, config: DictConfig):
        super().__init__()

        self.adapt = config.adapt
        self.bias = config.bias
        if self.adapt:
            # One embedding layer for each day
            self.embed_spikes = nn.Parameter(torch.zeros(config.n_dates, hidden_size, config.n_channels))
            sd = 1. / (config.n_channels ** 0.5)
            self.embed_spikes.data.uniform_(-sd, sd) # default pytorch linear layer initialization
            if self.bias:
                self.embed_spikes_bias = nn.Parameter(torch.zeros(config.n_dates))
                self.embed_spikes_bias.data.uniform_(-sd, sd)  # default pytorch linear layer initialization
        else:
            # One common embedding layer
            self.embed_spikes = nn.Linear(config.n_channels, hidden_size, bias=config.bias)

        # Embedding scale
        self.scale = hidden_size ** 0.5

        # Embed postion
        self.pos = config.pos
        if self.pos:
            self.embed_pos = nn.Embedding(config.max_F, hidden_size)

        # Regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(
            self, 
            features:           torch.FloatTensor,      # (batch_size, fea_len, n_channels)
            features_timestamp: Optional[torch.LongTensor],          # (batch_size, fea_len)
            block_idx:          Optional[torch.LongTensor] = None,   # (batch_size)
            date_idx:           Optional[torch.LongTensor] = None,   # (batch_size)
        ) -> torch.FloatTensor:                     # (batch_size, fea_len, hidden_size)

        if self.adapt:
            weight = self.embed_spikes[date_idx]
            x = (weight @ features.transpose(-1,-2)).transpose(-1,-2)
            if self.bias:
                x += self.embed_spikes_bias[date_idx].unsqueeze(-1).unsqueeze(-1)
        else:
            x = self.embed_spikes(features)

        x = x * self.scale

         # Embed position
        if self.pos:
            x += self.embed_pos(features_timestamp)

        return self.dropout(x)

# MLP
class NeuralMLP(nn.Module):

    def __init__(self, size_in, size_out, act, use_bias, dropout):
        super().__init__()

        self.up_proj    = nn.Linear(size_in, 4 * size_in, bias=use_bias)
        self.act        = ACT2FN[act]
        self.down_proj  = nn.Linear(4 * size_in, size_out, bias=use_bias)
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
        mask:       torch.LongTensor,                       # (batch_size, fea_len, fea_len)
        timestamp:  Optional[torch.LongTensor] = None,      # (batch_size, fea_len)
    ) -> torch.FloatTensor:                                 # (batch_size, fea_len, hidden_size)

        B, F, _  = x.size()     # batch size and fea len

        # Create batched bool attention mask 
        attn_mask = mask.unsqueeze(1).expand(B,self.n_heads,F,F).bool()            # (B,n_heads,F,F)

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
    
    def __init__(self, idx, hidden_size, config: DictConfig):
        super().__init__()

        self.idx = idx
        
        # Architecture config
        self.use_rope = config.use_rope

        # Encoder block
        self.ln1 = ScaleNorm(hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(hidden_size) 
        self.attn = NeuralAttention(idx, hidden_size, config.n_heads, config.attention_bias, config.dropout, config.use_rope, config.rope_theta)
        self.ln2 = ScaleNorm(hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(hidden_size) 
        self.mlp = NeuralMLP(hidden_size, hidden_size, config.act, config.mlp_bias, config.dropout)

        if config.fixup_init:
            self.fixup_initialization(config.n_layers)

    def forward(
        self, 
        x:          torch.FloatTensor,                  # (batch_size, fea_len, hidden_size)
        mask:       torch.LongTensor,                   # (batch_size, fea_len, fea_len)
        timestamp:  Optional[torch.LongTensor] = None,  # (batch_size, fea_len)          
    ) -> torch.FloatTensor :                            # (batch_size, fea_len, hidden_size)
        
        # LN -> Attention -> Residual connectiob
        x = x + self.attn(self.ln1(x), mask, timestamp if self.use_rope else None)

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
        self.out_space = "factors" if config.project_to_factors else "hidden"
        
        self.dropout = nn.Dropout(config.dropout)

        if config.project_to_factors:
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, config.size, config.bias),
                ACT2FN[config.act]
            )
            # Renitialize weights
            if config.re_init:
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

        self.n_channels = config.embedder.n_channels
        self.hidden_size = config.embedder.mult * self.n_channels
        self.use_rope = config.use_rope

        # Embedding layer
        self.embedder = NeuralEmbeddingLayer(self.hidden_size, config.embedder)

        # Context span mask
        context_mask = self.create_context_mask(config.context_forward, config.context_backward, config.embedder.max_F)
        self.register_buffer("context_mask", context_mask, persistent=False)

        # Transformer
        self.n_layers = config.transformer.n_layers
        self.layers = nn.ModuleList([NeuralEncoderLayer(idx, self.hidden_size, config.transformer) for idx in range(self.n_layers)])
        self.out_norm = ScaleNorm(hidden_size ** 0.5) if config.transformer.use_scalenorm else nn.LayerNorm(self.hidden_size) 
       
        # Out projection
        self.out_proj = NeuralFactorsProjection(self.hidden_size, config.factors)

        # Initialization
        if config.fixup_init:
            self.fixup_initialization()



    def forward(
            self, 
            features:           torch.FloatTensor,   # (batch_size, fea_len, n_channels)
            features_mask:      torch.LongTensor,   # (batch_size, fea_len)
            features_timestamp: torch.LongTensor,   # (batch_size, fea_len)
            block_idx:          Optional[torch.LongTensor] = None,   # (batch_size)
            date_idx:           Optional[torch.LongTensor] = None,   # (batch_size)
        ) -> torch.FloatTensor:                     # (batch_size, fea_len, hidden_size)
        
        B, F, _ = features.size() # batch size and seq len
        
        # Embed neural data
        x = self.embedder(features, features_timestamp, block_idx, date_idx)

        
        # Prepare attention mask
        context_mask = self.context_mask[:F,:F].to(x.device).unsqueeze(0).expand(B,F,F)
        features_mask = features_mask.unsqueeze(1).expand(B,F,F)
        self_mask = torch.eye(F).to(x.device, context_mask.dtype) # hack so that even padded features attend to themselves and avoid attention issues
        mask = self_mask | (context_mask & features_mask) if context_mask is not None else features_mask
        
        
        # Forward transformer
        for idx, layer in enumerate(self.layers):
            x = layer(x, mask=mask, timestamp=features_timestamp)
        x = self.out_norm(x)

        return self.out_proj(x)

    
    # Limit context for encoding
    def create_context_mask(
            self, 
            context_forward,
            context_backward,
            max_F,  
        ) -> torch.LongTensor:          # (max_fea_len, max_fea_len )

        if context_forward == -1 and context_backward == -1:
            return None

        context_forward = context_forward if context_forward >= 0 else max_F
        mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_forward) == 1).transpose(0, 1)
        if context_backward > 0:
            back_mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_backward) == 1)
            mask = mask & back_mask

        return mask


    

class ScaleNorm(nn.Module):

    def __init__(self, scale, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm