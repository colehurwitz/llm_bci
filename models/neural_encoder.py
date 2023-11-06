from typing import List, Optional, Tuple, Dict
from functools import partial

import torch
import torch.nn as nn

from transformers.activations import ACT2FN

class NeuralConfig:
    
    # data related
    n_channels = 256
    n_blocks = 25
    n_dates = 24
    embed_context = False
    embed_gate = False
    embed_act = "sigmoid"

    # architecture
    n_cross_layers = 2
    n_self_layers = 0
    # n_latents =   [256,256,128,64,32,16,16]
    # hidden_size = [128,128,256,512,1024,2048,4096]
    n_latents =   [256,256,16]
    hidden_size = [128,128,4096]
    n_heads = [4,4]
    hidden_act = "silu"

    dropout = 0.5
    embed_bias = False
    attention_bias = False
    mlp_bias = False

    rope_theta = 10000.0


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
def apply_rotary_pos_emb(q, k, q_pos_ids, k_pos_ids, cos, sin, unsqueeze_dim=1):

    cos_q = cos[q_pos_ids].unsqueeze(unsqueeze_dim)
    sin_q = sin[q_pos_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)

    cos_k = cos[k_pos_ids].unsqueeze(unsqueeze_dim)
    sin_k = sin[k_pos_ids].unsqueeze(unsqueeze_dim)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)

    
    return q_embed, k_embed


# Gated MLP
class NeuralMLP(nn.Module):

    def __init__(self, size_in, size_out, act, use_bias, dropout):
        super().__init__()

        self.up_proj    = nn.Linear(size_in, 4 * size_in, bias=use_bias)
        self.gate_proj  = nn.Linear(size_in, 4*size_in, bias=use_bias)
        self.act        = ACT2FN[act]
        self.down_proj  = nn.Linear(4 * size_in, size_out, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, latents):
        
        latents = self.up_proj(latents) * self.act(self.gate_proj(latents))
        return self.dropout(self.down_proj(latents))


# Attention module. Target can be features (cross-attention) or None, in which case defaults to latents (self-attention)
class NeuralAttention(nn.Module):

    def __init__(self, n_latents, hidden_size, n_heads, use_bias, dropout, max_n_latents, base, idx):
        super().__init__()
        
        self.idx = idx

        # Architecture config
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        assert self.hidden_size % self.n_heads == 0, f"Hidden dim is not multiple of head size at layer {idx}"
        self.head_size = self.hidden_size // self.n_heads

        # Attention parameters
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.value  = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)

        # Flash attention
        # torch.backends.cuda.enable_flash_sdp(True)
        self.flash_attention = partial(torch.nn.functional.scaled_dot_product_attention, dropout_p=dropout, is_causal=False)

        # Final projection
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=use_bias)


        # RoPE parameters
        cos, sin = get_cos_sin(self.head_size, max_n_latents, base=base, dtype=self.query.weight.dtype, device=self.query.weight.device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        latents_timestamp = torch.arange(n_latents) * (max_n_latents-1) / (n_latents-1) 
        self.register_buffer("latents_timestamp", latents_timestamp.to(device=self.query.weight.device), persistent=False)


    def forward(
        self,       
        latents,                    # (batch_size, latents_len, hidden_size)
        target           = None,    # (batch_size, target_len, hidden_size)
        target_mask      = None,    # (batch_size, target_len)
        target_timestamp = None,    # (batch_size, target_len)
        int_dtype        = None            
    ) -> torch.FloatTensor:         # (batch_size, latents_len, hidden_size)

        B, L, _  = latents.size() # batch size and number of input bins

        # Deduce int compute type
        if int_dtype is None:
            assert target_timestamp is not None or target_mask is not None, "int dtype not provided and could not deduce it from target "
        int_dtype = (target_timestamp.dtype if target_timestamp is not None else target_mask.dtype) if int_dtype is None else int_dtype
        
        # Cast latents to appropriate int type and batch 
        latents_timestamp = self.latents_timestamp.to(int_dtype).unsqueeze(0).expand(B,L)       # (B, L)

        # Default target to latents
        target = latents if target is None else target                                          # (B, T, hidden_size)
        target_timestamp = latents_timestamp if target_timestamp is None else target_timestamp  # (B, T)

        # Create batched bool attention mask 
        _, T, _ = target.size()
        attn_mask = None if target_mask is None else target_mask.unsqueeze(1).unsqueeze(2).expand(B,self.n_heads,L,T).bool()                         # (B, T)
        
        # Compute query, key, value for attention
        q = self.query(latents).view(B, L, self.n_heads, self.head_size).transpose(1, 2)        #(B,n_heads,L,head_size)
        k = self.key(target).view(B, T, self.n_heads, self.head_size).transpose(1, 2)           #(B,n_heads,T,head_size)
        v = self.value(target).view(B, T, self.n_heads, self.head_size).transpose(1, 2)         #(B,n_heads,T,head_size)

        # Apply rotations to encode relative positions
        q, k = apply_rotary_pos_emb(q, k, latents_timestamp, target_timestamp, self.cos, self.sin, 1)  # (B, n_heads, L/T, head_size)

        # Compute attention efficiently
        out = self.flash_attention(q, k, v, attn_mask=attn_mask)            # (B, n_heads, L, head_size)
        out = out.transpose(1, 2).contiguous().view(B,L, self.hidden_size)  # (B, L, hidden_size)

        return self.output_projection(self.dropout(out)) # (B, L, hidden_size)



# Encoder layer. Target can be features (cross-attention) or None, in which case defaults to latents (self-attention)
class NeuralEncoderLayer(nn.Module):
    
    def __init__(self, config: NeuralConfig, idx):
        super().__init__()

        self.idx = idx

        # Architecture config
        self.n_latents_in = config.n_latents[idx]
        self.n_latents_out = config.n_latents[idx+1]
        self.hidden_size_in = config.hidden_size[idx]
        self.hidden_size_out = config.hidden_size[idx+1]
        self.hidden_size_inter = self.n_latents_in * self.hidden_size_in // self.n_latents_out

        # Encoder block
        self.ln1 = nn.LayerNorm(self.hidden_size_in)
        self.attention = NeuralAttention(self.n_latents_in, self.hidden_size_in, config.n_heads[idx], config.attention_bias, config.dropout, max(config.n_latents), config.rope_theta, idx)
        self.ln2 = nn.LayerNorm(self.hidden_size_inter)
        self.residual_projection = nn.Linear(self.hidden_size_inter, self.hidden_size_out, bias=config.mlp_bias)
        self.mlp = NeuralMLP(self.hidden_size_inter, self.hidden_size_out, config.hidden_act, config.mlp_bias, config.dropout)
        
        

    def forward(
        self, 
        latents: torch.FloatTensor,                             # (batch_size, latents_len, hidden_size)
        target: Optional[torch.FloatTensor]          = None,    # (batch_size, target_len, hidden_size)
        target_mask: Optional[torch.LongTensor]      = None ,   # (batch_size, target_len)
        target_timestamp: Optional[torch.LongTensor] = None,     # (batch_size, target_len)
        int_dtype                                    = None            
    ) -> torch.FloatTensor :                                    # (batch_size, latents_len, hidden_size)
        
        B, _, _ = latents.size()

        # LN -> Attention -> Residual projection
        target = self.ln1(target) if target is not None else None
        latents = self.ln1(latents)
        latents = latents + self.attention(latents, target, target_mask, target_timestamp, int_dtype=int_dtype)

        # LN -> MLP -> Residual projection
        latents = latents.view(B, self.n_latents_out, self.hidden_size_inter)
        latents = self.residual_projection(latents) + self.mlp(self.ln2(latents))

        return latents

        


# Encoder for time binned neural data
class NeuralEncoder(nn.Module):

    def __init__(self, config: NeuralConfig):
        super().__init__()
        # The different layers and sizes should be consistent at least in lenght. This does not ensure full consistency
        # andsome operations will rise an error if there is a mismatch
        self.n_layers = config.n_cross_layers + config.n_self_layers
        self.n_cross_layers = config.n_cross_layers
        self.n_self_layers = config.n_self_layers
        assert len(config.hidden_size) == self.n_layers + 1, "Hidden size pattern and number of layers don't match"
        assert len(config.n_latents) == self.n_layers + 1, "Number of latents pattern and number of layers don't match"
        assert len(config.n_heads) == self.n_layers, "Head size pattern and number of layers don't match"
        self.n_latents = config.n_latents[0]
        self.hidden_size = config.hidden_size[0]
        
        # Levers
        self.embed_context = config.embed_context
        self.embed_gate = config.embed_gate


        # Embed latents and context
        if config.embed_context:
            self.embed_block = nn.Embedding(config.n_blocks, self.hidden_size)
            self.embed_date = nn.Embedding(config.n_dates, self.hidden_size)
            self.latents = nn.Parameter(torch.randn(self.n_latents,self.hidden_size))
            self.embed_latents = nn.Linear(3*self.hidden_size, self.hidden_size, bias=config.embed_bias)
        else:
            self.latents = nn.Parameter(torch.randn(self.n_latents,self.hidden_size))
            

        # Embed neural data
        self.embed_spikes = nn.Linear(config.n_channels, self.hidden_size, bias=config.embed_bias)
        if config.embed_gate:
            self.gate_spikes = nn.Sequential(
                nn.Linear(config.n_channels, self.hidden_size, bias=config.embed_bias),
                ACT2FN[config.embed_act],
            )

        self.dropout = nn.Dropout(config.dropout)

        # Attention+MLP layers
        self.layers = nn.ModuleList([NeuralEncoderLayer(config, idx) for idx in range(self.n_layers)])


    def forward(
            self, 
            features: torch.FloatTensor,            # (batch_size, seq_len, n_channels)
            features_mask: torch.FloatTensor,       # (batch_size, seq_len)
            features_timestamp: torch.LongTensor,   # (batch_size, seq_len)
            block_idx: torch.LongTensor,            # (batch_size, seq_len)
            date_idx: torch.LongTensor,             # (batch_size, seq_len)
        ) -> torch.FloatTensor:                     # (batch_size, latents_len, hidden_size)
        
        B, _, _ = features.size() # batch size
        
        # Embed latents together with context; and batch
        if self.embed_context:
            block_embd = self.embed_block(block_idx).unsqueeze(1).expand(B,self.n_latents,self.hidden_size)
            date_embd = self.embed_block(date_idx).unsqueeze(1).expand_as(block_embd)
            latents = self.embed_latents(torch.cat((self.latents.unsqueeze(0).expand_as(block_embd),block_embd,date_embd),-1))
        else:
            latents = self.latents.unsqueeze(0).expand(B,self.n_latents,self.hidden_size)

        # Embed neural data
        if self.embed_gate:
            features = self.embed_spikes(features) * self.gate_spikes(features)
        else:
            features = self.embed_spikes(features)
            
        # Dropout
        latents = self.dropout(latents)
        features = self.dropout(features)

        # Forward cross-attention layers
        for idx, layer in enumerate(self.layers):
            if idx < self.n_cross_layers:
                latents = layer(latents, target=features, target_mask=features_mask, target_timestamp=features_timestamp)
            else:
                latents = layer(latents, int_dtype=features_timestamp.dtype) 
        return latents