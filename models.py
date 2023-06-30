import torch
import torch.nn as nn


class LinearTemporalConvNet(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_sizes=[2],
        kernel_size=128,
        bias=True,
    ):
        super(LinearTemporalConvNet, self).__init__()

        self.num_layers = len(hidden_sizes)
        self.convs = nn.ModuleList()

        # First convolutional layer
        self.convs.append(
            nn.Conv1d(input_size, hidden_sizes[0], kernel_size, bias=bias)
        )

        # Intermediate convolutional layers
        for i in range(1, self.num_layers):
            self.convs.append(
                nn.Conv1d(hidden_sizes[i - 1], hidden_sizes[i], kernel_size, bias=bias)
            )

    def forward(self, x):
        x = x.permute(
            0, 2, 1
        )  # Convert input to (batch_size, input_size, sequence_length)
        # print("In models, after perm", x)

        for i in range(self.num_layers):
            padding = (self.convs[i].kernel_size[0] - 1) * self.convs[i].dilation[0]

            # Apply causal padding to the input
            x = nn.functional.pad(x, (padding, 0))
            # print("In models, after pad ", x)

            x = self.convs[i](x)
            # print("In models, after conv", x)

        x = x.permute(
            0, 2, 1
        )  # Convert back to (batch_size, sequence_length, hidden_size)

        return x
