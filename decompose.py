import tensorly as tl
from torch.autograd import Variable
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
import VBMF

tl.set_backend("pytorch")


def cp_decomposition(layer, rank):
    """Gets a conv layer,
    returns a nn.Sequential object with the CP decomposition.
    """

    W = layer.weight.data

    last, first, vertical, horizontal = parafac(W, rank=rank, init="random")[1]

    pointwise_s_to_r_layer = nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        padding=0,
        bias=False,
    )

    depthwise_r_to_r_layer = nn.Conv2d(
        in_channels=rank,
        out_channels=rank,
        kernel_size=vertical.shape[0],
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=rank,
        bias=False,
    )

    pointwise_r_to_t_layer = nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        padding=0,
        bias=True,
    )

    if layer.bias is not None:
        pointwise_r_to_t_layer.bias.data = layer.bias.data

    sr = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    rt = last.unsqueeze_(-1).unsqueeze_(-1)
    rr = torch.stack(
        [
            vertical.narrow(1, i, 1) @ torch.t(horizontal).narrow(0, i, 1)
            for i in range(rank)
        ]
    ).unsqueeze_(1)

    pointwise_s_to_r_layer.weight.data = sr
    pointwise_r_to_t_layer.weight.data = rt
    depthwise_r_to_r_layer.weight.data = rr

    new_layers = [
        pointwise_s_to_r_layer,
        depthwise_r_to_r_layer,
        pointwise_r_to_t_layer,
    ]
    return nn.Sequential(*new_layers)
    # return


def estimate_ranks(layer):
    """Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF
    """
    weights = layer.weight.data
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks


def tucker_decomposition(layer):
    """Gets a conv layer,
    returns a nn.Sequential object with the Tucker decomposition.
    The ranks are estimated with a Python implementation of VBMF
    https://github.com/CasvandenBogaard/VBMF
    """

    ranks = estimate_ranks(layer)
    if 0 in ranks:
        ranks.remove(0)
        ranks = [ranks[0], ranks[0]]
    weights = layer.weight.data
    core, [last, first] = partial_tucker(weights, modes=[0, 1], rank=ranks, init="svd")

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(
        in_channels=core.shape[1],
        out_channels=core.shape[0],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=True,
    )

    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)


def decompose_layer(type, layer):
    """
    return the new decomposed conv2d layer
    """
    if type == "CP":
        new_layer = cp_decomposition(layer)
    else:
        new_layer = tucker_decomposition(layer)
    return new_layer
