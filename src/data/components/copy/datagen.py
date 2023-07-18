import numpy as np
import torch


def data_generator(
    data_dir: str, size: int, seq_length: int, mem_length: int, input_dim: int = 1
):
    """
    Generate data for the copying memory task

    :param size: the batch size
    :param seq_length: the total blank time length
    :param mem_length: the length of the memory to be recalled
    :return: Input and target data tensor
    1253614124000099999999999->
    0000000000000001253614124
    """
    seq = torch.from_numpy(np.random.randint(1, 9, size=(size, mem_length))).double()

    zeros = torch.zeros((size, seq_length))
    marker = 9 * torch.ones((size, mem_length + 1))
    placeholders = torch.zeros((size, mem_length))

    x = torch.cat((seq, zeros[:, :-1], marker), 1).unsqueeze(-1)
    y = torch.cat((placeholders, zeros, seq), 1).unsqueeze(-1)
    # y = torch.cat((placeholders, zeros, seq), 1).long()

    if data_dir is not None:
        input_file_path = data_dir + f"copy_{mem_length}_inputs.npy"
        output_file_path = data_dir + f"copy_{mem_length}_outputs.npy"

        np.save(
            input_file_path,
            x.numpy(),
        )
        np.save(
            output_file_path,
            y.numpy(),
        )

    return x, y


if __name__ == "__main__":
    x, y = data_generator(data_dir="./", size=1, seq_length=20, mem_length=10)
    print(torch.squeeze(x))
    print(torch.squeeze(y))
    print(x.shape)
    print(y.shape)
    print(x.dtype)
    print(y.dtype)

    print("Test passed.")
