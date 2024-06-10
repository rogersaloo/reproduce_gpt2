import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import Functional as F


class CasualSelfAttention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #key, query and calue projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a bias but more of a mask while folloeitn the openAI/HF naming though
        self.register_buffer("bias", 
                             torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        #attention ( materialize s the larhe (T, T ) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 /math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, T, T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contigous().view(B, T, C)
        #output projection
        y = self.c_proj(y)
        return y


        return super()._slow_forward(*input, **kwargs)


class MLP(nn.Module):
    def __init__(self, config) -> None:

        super().__init__()
        self.c_fc = nn.Linear(config.n_ambd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Modele):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_heads: int = 6
    n_embed: int  = 384


class GPT(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

