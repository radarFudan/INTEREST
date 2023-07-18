# https://github.com/locuslab/TCN
import torch
import torch.nn.utils.parametrize as P
from torch import nn
from torch.nn.utils import weight_norm


# remove the last component
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                dtype=torch.float64,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                dtype=torch.float64,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1, dtype=torch.float64)
            if n_inputs != n_outputs
            else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(
        self,
        num_channels,
        kernel_size,
        dropout=0.0,
        return_sequences=True,
        input_size=1,
        output_size=1,
        bias=True,
    ):
        super().__init__()
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.linear = nn.Linear(
            num_channels[-1], output_size, bias=bias, dtype=torch.float64
        )
        self.init_weights()

        self.return_sequences = return_sequences

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.transpose(1, 2)

        # print("tcn", x.dtype)
        y1 = self.tcn(x)

        y = y1.transpose(1, 2)
        y = self.linear(y)

        if self.return_sequences:
            return y
        else:
            return y[:, -1, :]


if __name__ == "__main__":
    batch_size = 1
    input_length = 100
    embed_dim = 10

    test_model = TCN(
        [30],
        7,
        bias=False,
        input_size=10,
        output_size=30,
    )

    inputs = torch.zeros(1, 100, 10)
    inputs[:, 0, :] = 1
    outputs = test_model(inputs)

    assert outputs.shape == torch.Size([1, 100, 30])
    print("Test passed.")

    squeezed_output = torch.squeeze(outputs)
    # print(squeezed_output[1:] - squeezed_output[:-1])
