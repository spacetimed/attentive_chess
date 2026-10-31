import torch

from dataclasses import dataclass

@dataclass
class TransformerConfig:
    num_layers: int = 4 # number of transformer blocks
    num_heads: int = 8 # number of heads per layer

    # 31 for FEN vocabulary + 1968 unique positions
    vocab_size: int = 1999

    # token stuff
    embed_dim: int = 64
    num_embed: int = 78 # positions

    # mlp
    widening_factor: int = 4

class Attention(torch.nn.Module):
    pass

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

        #self.Attn = Attention()
        self.Ln1 = torch.nn.LayerNorm(self.config.embed_dim)
        self.Ln2 = torch.nn.LayerNorm(self.config.embed_dim)
        self.MLP = MLP(config)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x += self.Attn(self.Ln1(x))
        x = self.MLP(self.Ln2(x))
        return x

class Transformer(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.blocks = torch.nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        # token ID --> d vec
        self.embed_table = torch.nn.Embedding(config.vocab_size, config.embed_dim) # [1999, 64]

        # token position 0...79 --> d vec
        self.position_table = torch.nn.Embedding(config.num_embed, config.embed_dim) # [79, 64]

    def forward(self, inputs: torch.Tensor):
        B,T = inputs.shape # [B, 79]

        h = self.embed_table(inputs)     # [B,79,D]
        h += self.position_table(inputs) # [B,79,D]

        for b in self.blocks: h = b(h)

