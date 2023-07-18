import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 1,
        batch_first: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=batch_first, bias=bias
        )  # 4 * embed_dim * embed_dim

    def forward(self, x):
        attended_values, _ = self.attention(x, x, x)
        return attended_values


# Multi-head attention transformer
class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 1,
        batch_first: bool = True,
        layers: int = 12,
        return_sequences: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [Attention(embed_dim, num_heads, batch_first, bias) for _ in range(layers)]
        )
        self.return_sequences = return_sequences

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        if self.return_sequences:
            return x
        else:
            return x[:, -1, :]


# Maybe consider multi-query transformer
# TODO

if __name__ == "__main__":
    batch_size = 1
    input_length = 100
    embed_dim = 10

    test_model = Attention(embed_dim=embed_dim, bias=False)

    inputs = torch.zeros(batch_size, input_length, embed_dim)
    outputs = test_model(inputs)
    assert outputs.shape == torch.Size([batch_size, input_length, embed_dim])

    print("Test passed.")

    test_model = Transformer(embed_dim=embed_dim, layers=4, bias=False)

    inputs = torch.zeros(batch_size, input_length, embed_dim)
    outputs = test_model(inputs)
    assert outputs.shape == torch.Size([batch_size, input_length, embed_dim])

    print("Test passed.")
