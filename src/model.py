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

class Attention(torch.nn.Module):
    pass

class MLP(torch.nn.Module):
    pass

class Block(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        #self.Attn = Attention()
        self.Ln1 = torch.nn.LayerNorm(self.config.embed_dim)
        self.Ln2 = torch.nn.LayerNorm(self.config.embed_dim)
        #self.MLP + MLP()

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += self.Attn(self.Ln1(x))
        x += self.MLP(self.Ln2(x))
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

