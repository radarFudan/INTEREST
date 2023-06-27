import torch
import torch.nn.functional as F

from functools import partial


def cross_entropy(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return F.cross_entropy(logits, y)


def cross_entropy_wl(logits, y, p=2, time_dim=1):
    """_summary_

    Args:
        logits (_type_): B * T * ...
        y (_type_): B * T * ...
        p (int, optional): _description_. Defaults to 2.
            TODO, later to support np.infinity.
            0 is the vanilla mean squared error.
        time_dim (int, optional): Time weighted dimension over dim 1.
            TODO, generalize it to general dimension

    Returns:
        _type_: _description_
    """
    assert time_dim == 1

    shape_len = len(logits.shape)
    shape = tuple(1 if i != 1 else logits.shape[1] for i in range(shape_len))

    weight = torch.ones(shape, device=logits.device)
    for i in range(shape[1]):
        weight[:, i] = ((i + 1) / (shape[1] + 1)) ** p
    weight /= weight.sum()

    return cross_entropy(logits * weight, y)


def mse(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.mse_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        # TODO document the use case of this
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.mse_loss(outs_masked, y_masked)


def mse_wl(outs, y, len_batch=None, p=2, time_dim=1):
    """_summary_

    Args:
        outs (_type_): B * T or B*T*d_i...
        y (_type_): B * T or B*T*d_i...
        len_batch (_type_, optional): _description_. Defaults to None.
        p (int, optional): _description_. Defaults to 2.
            TODO, later to support np.infinity.
            0 is the vanilla mean squared error.
        time_dim (int, optional): Time weighted dimension over dim 1.
            TODO, generalize it to general dimension

    Returns:
        _type_: _description_
    """
    assert time_dim == 1

    shape_len = len(outs.shape)
    shape = tuple(1 if i != 1 else outs.shape[1] for i in range(shape_len))

    weight = torch.ones(shape, device=outs.device)
    for i in range(shape[1]):
        weight[:, i] = ((i + 1) / (shape[1] + 1)) ** p
    weight /= weight.sum()

    return F.mse_loss(outs * weight, y * weight)


def mae(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.l1_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.l1_loss(outs_masked, y_masked)


def mae_wl(outs, y, len_batch=None, p=2, time_dim=1):
    """_summary_

    Args:
        outs (_type_): B * T or B*T*d_i...
        y (_type_): B * T or B*T*d_i...
        len_batch (_type_, optional): _description_. Defaults to None.
        p (int, optional): _description_. Defaults to 2.
            TODO, later to support np.infinity.
            0 is the vanilla mean squared error.
        time_dim (int, optional): Time weighted dimension over dim 1.
            TODO, generalize it to general dimension

    Returns:
        _type_: _description_
    """
    assert time_dim == 1

    shape_len = len(outs.shape)
    shape = tuple(1 if i != 1 else outs.shape[1] for i in range(shape_len))

    weight = torch.ones(shape, device=outs.device)
    for i in range(shape[1]):
        weight[:, i] = ((i + 1) / (shape[1] + 1)) ** p
    weight /= weight.sum()

    return mae(outs * weight, y * weight)


# should have a better way to do this
output_metric_fns = {
    "cross_entropy": cross_entropy,
    "cross_entropy_wl0": partial(cross_entropy_wl, p=0),
    "cross_entropy_wl1": partial(cross_entropy_wl, p=1),
    "cross_entropy_wl2": partial(cross_entropy_wl, p=2),
    "cross_entropy_wl10": partial(cross_entropy_wl, p=10),
    "mse": mse,
    "mse_wl0": partial(mse_wl, p=0),
    "mse_wl1": partial(mse_wl, p=1),
    "mse_wl2": partial(mse_wl, p=2),
    "mse_wl10": partial(mse_wl, p=10),
    "mae": mae,
    "mae_wl0": partial(mae_wl, p=0),
    "mae_wl1": partial(mae_wl, p=1),
    "mae_wl2": partial(mae_wl, p=2),
    "mae_wl10": partial(mae_wl, p=10),
}
