import torch

from dataclasses import dataclass

@dataclass
class TransformerConfig:
    num_layers: int = 4 # number of transformer blocks
    num_heads: int = 8 # number of attention heads per layer

    # 31 for FEN vocabulary + 1968 unique positions
    vocab_size: int = 1999

    # token stuff
    embed_dim: int = 64 # C
    num_embed: int = 79 # positions

    # mlp
    widening_factor: int = 4

class Attention(torch.nn.Module):
    """multi-head attention"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.qkv_linear = torch.nn.Linear(config.embed_dim, 3*config.embed_dim) # [B,T,C] -> [B,T,3C]
        self.proj_linear = torch.nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C = x.shape

        qkv = self.qkv_linear(x) # [B,T,3C]
        q,k,v = torch.split(qkv, self.config.embed_dim, dim=-1) # 3x [B,T,C]

        H = self.config.num_heads
        D = self.config.embed_dim // H

        # views s.t. [B,T,C] -> [B,T,H,D] (unpack embed dim into HxD matrices)
        q = q.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)

        # transpose dims=1,2 s.t. [B,T,H,D] -> [B,H,T,D] 
        # q[b, h] now contains entire TxD sequence
        q,k,v = map(lambda x: x.transpose(1,2), (q,k,v)) 
        
        # attn(q,k,v) = softmax(q*k^T/sqrt(d_k)+M)*v
        # using torch's sdpa for optimized attention (non-causal; no masking)
        attn = torch.nn.functional.scaled_dot_product_attention(q,k,v) # [B,H,T,D]
        attn = attn.transpose(1,2).contiguous().view(B, T, C) # [B,T,C]
        attn = self.proj_linear(attn)

        return attn

class MLP(torch.nn.Module):
    """gated MLP block for transformer"""

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        ffn_dim = config.embed_dim * config.widening_factor

        self.linear1 = torch.nn.Linear(config.embed_dim, ffn_dim, bias=False) # 64 -> 256
        self.linear2 = torch.nn.Linear(config.embed_dim, ffn_dim, bias=False) # 64 -> 256
        self.silu = torch.nn.SiLU()
        self.proj = torch.nn.Linear(ffn_dim, config.embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split_1 = self.linear1(x) # [B,T,C=64] --> [B,T,4*C=256]
        split_2 = self.linear2(x)

        gate_out = self.silu(split_1) * split_2
        return self.proj(gate_out) # [B,T,C=64]

class Block(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.Ln1 = torch.nn.LayerNorm(self.config.embed_dim)
        self.Ln2 = torch.nn.LayerNorm(self.config.embed_dim)

        self.Attn = Attention(config)
        self.MLP = MLP(config)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = [B,T,C]
        x = x + self.Attn(self.Ln1(x)) # [B,T,C]
        x = x + self.MLP(self.Ln2(x))  # [B,T,C]
        return x

class Transformer(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.blocks = torch.nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        # token ID --> d vec
        self.embed_table = torch.nn.Embedding(config.vocab_size, config.embed_dim)   # [1999, 64]

        # token position 0...79 --> d vec
        self.position_table = torch.nn.Embedding(config.num_embed, config.embed_dim) # [79, 64]

    def forward(self, inputs: torch.Tensor):
        # input tokens: [ BOS(1) ][ FEN(77) ][ ACTION(1) ] 79 sequence len 
        #   * ACTION part of 1968 UCI-notation moves, 
        # output tokens: 128 k-bin probability distribution (winrates evaluating ACTION)

        # T = 79 (sequence length)
        # C = 64 (embed dim)

        B,T = inputs.shape # [B, T]

        positions = torch.arange(T, device=inputs.device)

        embeddings = self.embed_table(inputs)
        positions = self.position_table(positions)

        h = embeddings + positions # [B,T,C]

        for b in self.blocks: h = b(h)

