import math
from typing import List, Optional, Tuple, Dict
from functools import partial

import torch
import torch.nn as nn

from transformers.activations import ACT2FN

from utils.config_utils import DictConfig

EMBED_MODES = "linear, identity, embed"



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
        mask:       torch.LongTensor,                      # (batch_size, fea_len, fea_len)
        timestamp:  Optional[torch.LongTensor] = None,      # (batch_size, fea_len)
    ) -> torch.FloatTensor:                                 # (batch_size, fea_len, hidden_size)

        B, T, _  = x.size()     # batch size and fea len

        # Create batched bool attention mask 
        attn_mask = mask.unsqueeze(1).expand(B,self.n_heads,T,T).bool()            # (B,n_heads,T,T)

        # Compute query, key, value for attention
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)      #(B,n_heads,T,head_size)
        k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)        #(B,n_heads,T,head_size)
        v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)      #(B,n_heads,T,head_size)

        # Apply rotations to encode relative positions
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, timestamp, self.cos, self.sin, 1)  # (B,n_heads,T,head_size)

        # Compute attention efficiently
        out = self.flash_attention(q, k, v, attn_mask=attn_mask)                 # (B,n_heads,T,head_size)
        out = out.transpose(1, 2).contiguous().view(B,T, self.hidden_size)       # (B, T, hidden_size)

        return self.out_proj(self.dropout(out)) # (B, T, hidden_size)



# Encoder layer: bidirectional self-attention + mlp
class NeuralEncoderLayer(nn.Module):
    
    def __init__(self, idx, config: DictConfig):
        super().__init__()

        self.idx = idx
        
        # Architecture config
        self.hidden_size = config.n_channels * config.embed_mult
        self.use_rope = config.use_rope

        # Encoder block
        self.ln1 = ScaleNorm(self.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(self.hidden_size) 
        self.attn = NeuralAttention(idx, self.hidden_size, config.n_heads, config.attention_bias, config.layers_dropout, config.use_rope, config.rope_theta)
        self.ln2 = ScaleNorm(self.hidden_size ** 0.5) if config.use_scalenorm else nn.LayerNorm(self.hidden_size) 
        self.mlp = NeuralMLP(self.hidden_size, self.hidden_size, config.hidden_act, config.mlp_bias, config.layers_dropout)

    def forward(
        self, 
        x:          torch.FloatTensor,                  # (batch_size, fea_len, hidden_size)
        mask:       torch.LongTensor,                  # (batch_size, fea_len, fea_len)
        timestamp:  Optional[torch.LongTensor] = None,  # (batch_size, fea_len)          
    ) -> torch.FloatTensor :                            # (batch_size, fea_len, hidden_size)
        
        # LN -> Attention -> Residual connectiob
        x = x + self.attn(self.ln1(x), mask, timestamp if self.use_rope else None)

        # LN -> MLP -> Residual connection
        x = x + self.mlp(self.ln2(x))

        return x


# Encoder for time binned neural data
class NeuralEncoder(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        self.hidden_size = config.embed_mult * config.n_channels

        # Embed neural data
        if config.embed_mode == "linear":
            self.embed_spikes = nn.Linear(config.n_channels, self.hidden_size, bias=config.embed_bias)

        elif config.embed_mode == "embed":
            self.embed_spikes = nn.Sequential(
                nn.Embedding(config.max_spikes, config.embed_mult),
                nn.Flatten(start_dim=-2)
            )
            self.init_embed_weights(config.spike_log_init, config.max_spikes)

        elif config.embed_mode == "identity":
            self.embed_spikes = nn.Identity()

        else:
            raise Exception(f"Invalid embed mode {config.embed_mode}. Available modes are {EMBED_MODES}")

        # Optional gating of the embedding
        self.embed_gate = config.embed_gate and config.linear_embed
        if self.embed_gate:
            self.gate_spikes = nn.Sequential(
                nn.Linear(config.n_channels, self.hidden_size, bias=config.embed_bias),
                ACT2FN[config.embed_act],
            )

        # Embedding scale
        self.scale = math.sqrt(self.hidden_size)
        

        # Embed postion
        self.use_rope = config.use_rope
        if not self.use_rope:
            self.embed_pos = nn.Embedding(config.max_T, self.hidden_size)


        # Embed context
        self.embed_context = config.embed_context
        if self.embed_context:
            self.embed_block = nn.Embedding(config.n_blocks, self.hidden_size)
            self.embed_date = nn.Embedding(config.n_dates, self.hidden_size)
            
        # Regularization
        self.dropout = nn.Dropout(config.embed_dropout)

        # Context span mask
        context_mask = self.create_context_mask(config.context_forward, config.context_backward, config.max_T)
        self.register_buffer("context_mask", context_mask, persistent=False)

        # Attention+MLP layers
        self.n_layers = config.n_layers
        self.layers = nn.ModuleList([NeuralEncoderLayer(idx, config) for idx in range(self.n_layers)])

        
        if config.fixup_init:
            self.fixup_initialization()


    def forward(
            self, 
            features:           torch.LongTensor,   # (batch_size, fea_len, n_channels)
            features_mask:      torch.LongTensor,   # (batch_size, fea_len)
            features_timestamp: torch.LongTensor,   # (batch_size, fea_len)
            block_idx:          torch.LongTensor,   # (batch_size, fea_len)
            date_idx:           torch.LongTensor,   # (batch_size, fea_len)
        ) -> torch.FloatTensor:                     # (batch_size, fea_len, hidden_size)
        
        B, T, _ = features.size() # batch size and seq len
        
        # Embed neural data
        x = self.embed_spikes(features)
        print
        if self.embed_gate:
            x = x * self.gate_spikes(features)
        x = x * self.scale
        
        # Embed position
        if not self.use_rope:
            x += self.embed_pos(features_timestamp)

        # Embed context
        if self.embed_context:
            x += self.embed_block(block_idx) + self.embed_date(date_idx)

        x = self.dropout(x)
        
        # Prepare attention mask
        context_mask = self.context_mask[:T,:T].to(x.device).unsqueeze(0).expand(B,T,T)
        features_mask = features_mask.unsqueeze(1).expand(B,T,T)
        self_mask = torch.eye(T).to(x.device, context_mask.dtype) # hack so that even padded features attend to themselves and avoid attention issues
        mask = self_mask | (context_mask & features_mask) if context_mask is not None else features_mask
        

        
        # Forward attention layers
        for idx, layer in enumerate(self.layers):
            x = layer(x, mask=mask, timestamp=features_timestamp if self.use_rope else None)

        return x

    

    # Initialization methods copied from NDT
    def init_embed_weights(self, spike_log_init, max_spikes):
        init_range = 0.1
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

        # self.decoder[0].bias.data.zero_()
        # self.decoder[0].weight.data.uniform_(-initrange, initrange)
        # nn.init.xavier_uniform_(m.weight)


    def fixup_initialization(self):

        temp_state_dic = {}
        for name, param in self.named_parameters():
            if name.endswith("_proj.weight"):
                temp_state_dic[name] = (0.67 * (self.n_layers) ** (- 1. / 4.)) * param
            elif name.endswith("attn.value.weight"):
                temp_state_dic[name] = (0.67 * (self.n_layers) ** (- 1. / 4.)) * (param * (2**0.5))

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)      

    
    # Limit context for encoding
    def create_context_mask(
            self, 
            context_forward,
            context_backward,
            max_T,  
        ) -> torch.LongTensor:          # (max_fea_len, max_fea_len )

        if context_forward == -1 and context_backward == -1:
            return None

        context_forward = context_forward if context_forward >= 0 else max_T
        mask = (torch.triu(torch.ones(max_T, max_T), diagonal=-context_forward) == 1).transpose(0, 1)
        if context_backward > 0:
            back_mask = (torch.triu(torch.ones(max_T, max_T), diagonal=-context_backward) == 1)
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