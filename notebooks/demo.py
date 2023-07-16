import numpy as np
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn

from models import LinearTemporalConvNet
from utils import output_metric_fns

# kernel_size_list = [4, 16, 64, 256, 1024, 4096]
kernel_size_list = [4, 16, 64, 512, 1024, 4096]
ratio_growth_dict = {"p=0": [], "p=1": [], "p=2": [], "p=10": []}

for kernel_size in kernel_size_list:
    # kernel_size = 1024

    model = LinearTemporalConvNet(hidden_sizes=[1], kernel_size=kernel_size, bias=False)
    # For convolution kernels, the memory length is bounded by kernel_size

    state_dict = model.state_dict()

    pol_weights = torch.tensor(
        [(1 / (kernel_size - i)) ** 1.2 for i in range(kernel_size)],
        dtype=torch.float64,
    ).reshape(1, 1, kernel_size)
    exp_weights = torch.tensor(
        [(1 / 1.2) ** (kernel_size - i) for i in range(kernel_size)],
        dtype=torch.float64,
    ).reshape(1, 1, kernel_size)
    state_dict[list(state_dict.keys())[0]] = pol_weights

    model.load_state_dict(state_dict)

    T = kernel_size

    inputs = torch.zeros(1, T, 1)  # B * T * D
    inputs[:, 0, :] = 1
    # print(inputs)

    # print(dict(model.named_parameters()))
    # print(list(model.parameters()))
    outputs = model(inputs)
    # print("Outputs shape is", outputs.shape)
    # print("Outputs is", outputs)

    squeezed_outputs = torch.squeeze(outputs).detach().numpy()
    # print(squeezed_outputs)
    # plt.plot(squeezed_outputs)
    # memory = squeezed_outputs[1:] - squeezed_outputs[:-1]
    # memory = np.abs(memory)
    # print("Memory shape is", memory.shape)
    # print(memory)
    # plt.plot(memory)

    # plt.yscale("log")
    # plt.show()

    pol_weights[:, :, -1] = pol_weights[:, :, -1] + 1  # rho[0] perturbed by 1
    state_dict[list(state_dict.keys())[0]] = pol_weights
    model_head = copy.deepcopy(model)
    model_head.load_state_dict(state_dict)
    pol_weights[:, :, -1] = pol_weights[:, :, -1] - 1

    pol_weights[:, :, 0] = pol_weights[:, :, 0] + 1  # rho[T-1] perturbed by 1
    state_dict[list(state_dict.keys())[0]] = pol_weights
    model_tail = copy.deepcopy(model)
    model_tail.load_state_dict(state_dict)
    pol_weights[:, :, 0] = pol_weights[:, :, 0] - 1

    inputs = torch.randn(16, T, 1)  # B * T * D

    for p in [0, 1, 2, 10]:
        # print(
        #     output_metric_fns[f"mse_wl{p}"](model(inputs), model_head(inputs)),
        #     output_metric_fns[f"mse_wl{p}"](model(inputs), model_tail(inputs)),
        # )
        # print(
        #     "Ratio",
        #     output_metric_fns[f"mse_wl{p}"](model(inputs), model_head(inputs))
        #     / output_metric_fns[f"mse_wl{p}"](model(inputs), model_tail(inputs)),
        # )
        ratio_growth_dict[f"p={p}"].append(
            float(
                (
                    output_metric_fns[f"mse_wl{p}"](model(inputs), model_head(inputs))
                    / output_metric_fns[f"mse_wl{p}"](model(inputs), model_tail(inputs))
                ).detach()
            )
        )

print(ratio_growth_dict)
# plot ratio_growth_dict
plt.figure(figsize=(8, 6))
for p in [0, 1, 2, 10]:
    plt.plot(kernel_size_list, ratio_growth_dict[f"p={p}"], label=f"p={p}")
plt.legend()
plt.xlabel("sequence length")
plt.ylabel("ratio")
plt.xscale("log")
plt.yscale("log")
plt.show()
