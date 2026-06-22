from typing import Union
import torch
import numpy as np


def to_torch(data: Union[np.ndarray, dict]):
    if isinstance(data, dict):
        return {k: from_numpy(v) for k, v in data.items()}
    else:
        data = torch.from_numpy(data)
        if data.dtype == torch.float64:
            data = data.float()
        return data.to(device)


def to_numpy(tensor: Union[torch.Tensor, dict]):
    if isinstance(tensor, dict):
        return {k: to_numpy(v) for k, v in tensor.items()}
    else:
        return tensor.to("cpu").detach().numpy()

