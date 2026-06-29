from numbers import Number
from typing import Union

import torch
import numpy as np


def to_scalar(value):
    if isinstance(value, Number):
        return value
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    return None