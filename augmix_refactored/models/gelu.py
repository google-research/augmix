import torch
import torch.nn as nn


GELU_CONSTANT = 1.702
"""The GELU Constant"""


class GELU(nn.Module):
    """This is the GELU Module."""

    def forward(self, x):
        return torch.sigmoid(GELU_CONSTANT * x) * x