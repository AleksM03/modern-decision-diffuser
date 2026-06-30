from numbers import Number
from pathlib import Path
from typing import Union

import torch
import numpy as np
import json


def to_scalar(value):
    if isinstance(value, Number):
        return value
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    return None


def load_checkpoint_config(checkpoint_dir):
    config_path = Path(checkpoint_dir) / "flags.json"
    with open(config_path) as f:
        return json.load(f)


def load_model_from_checkpoint_dir(model, checkpoint_dir, device):
    checkpoint_path = Path(checkpoint_dir) / "agent.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return checkpoint_path
