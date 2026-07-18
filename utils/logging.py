import json
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.pytorch_utils import to_scalar


class Logger:
    """Logger backed by TensorBoard and a CSV file."""

    def __init__(self, log_dir: str, csv_path: str):
        self.log_dir = log_dir
        self.csv_path = csv_path
        self.path = csv_path
        self.header = None
        self.file = None
        self.writer = SummaryWriter(log_dir=log_dir)
        self.rows = []

    def log(self, row, step):
        scalar_row = {
            key: scalar
            for key, value in {**row, "step": step}.items()
            if (scalar := to_scalar(value)) is not None
        }

        new_keys = [
            key for key in scalar_row if self.header is None or key not in self.header
        ]
        if self.header is None:
            self.header = list(scalar_row.keys())
            self.file = open(self.csv_path, "w")
            self.file.write(",".join(self.header) + "\n")
        elif new_keys:
            self.header.extend(new_keys)
            self.file.close()
            self.file = open(self.csv_path, "w")
            self.file.write(",".join(self.header) + "\n")
            for prev_row in self.rows:
                prev_scalar_row = {
                    key: scalar
                    for key, value in prev_row.items()
                    if (scalar := to_scalar(value)) is not None
                }
                self.file.write(
                    ",".join(str(prev_scalar_row.get(key, "")) for key in self.header)
                    + "\n"
                )

        self.file.write(
            ",".join(str(scalar_row.get(key, "")) for key in self.header) + "\n"
        )
        self.file.flush()

        for key, value in scalar_row.items():
            if key != "step":
                self.writer.add_scalar(key, value, step)
        self.writer.flush()
        self.rows.append(scalar_row)

    def log_scalar(self, scalar, name, step):
        """Log a single scalar value."""
        value = to_scalar(scalar)
        if value is not None:
            self.writer.add_scalar(name, value, step)
            self.writer.flush()

    def log_trajs_as_videos(
        self, trajs, step, max_videos_to_save=2, fps=30, video_title="video"
    ):
        videos = [
            traj["image_obs"]
            for traj in trajs
            if "image_obs" in traj and len(traj["image_obs"]) > 0
        ][:max_videos_to_save]
        if not videos:
            return

        max_length = max(len(video) for video in videos)
        for idx, video in enumerate(videos):
            if len(video) < max_length:
                pad = np.repeat(video[-1:], max_length - len(video), axis=0)
                videos[idx] = np.concatenate([video, pad], axis=0)

        video_tensor = np.stack(videos)  # (N, T, H, W, C)
        video_tensor = np.transpose(video_tensor, (0, 1, 4, 2, 3))  # (N, T, C, H, W)
        self.writer.add_video(video_title, video_tensor, step, fps=fps)
        self.writer.flush()

    def log_paths_as_videos(
        self, paths, step, max_videos_to_save=2, fps=10, video_title="video"
    ):
        """Alias for log_trajs_as_videos for compatibility."""
        self.log_trajs_as_videos(paths, step, max_videos_to_save, fps, video_title)

    def flush(self):
        """Flush pending CSV and TensorBoard writes."""
        if self.file is not None:
            self.file.flush()
        self.writer.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
        self.writer.close()


def remove_functions(obj):
    if isinstance(obj, dict):
        return {k: remove_functions(v) for k, v in obj.items() if not callable(v)}
    if isinstance(obj, list):
        return [remove_functions(v) for v in obj if not callable(v)]
    if callable(obj):
        return None
    return obj


def dump_log(agent: nn.Module, logger: Logger, args, save_dir: str, ema_model: nn.Module):
    """Dump the log to a pkl file and checkpoint the agent."""
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = remove_functions(vars(args))

    data = {
        "log": logger.rows,
        "log_hash": hash(json.dumps(str(logger.rows), sort_keys=True)),
        "config": config,
        "config_hash": hash(json.dumps(str(config), sort_keys=True)),
        "time": cur_time,
    }

    with open(os.path.join(save_dir, "flags.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(save_dir, "log.pkl"), "wb") as f:
        pickle.dump(data, f)

    torch.save(agent.state_dict(), os.path.join(save_dir, "raw_model.pt"))
    torch.save(ema_model.state_dict(), os.path.join(save_dir, "ema_model.pt"))
    logger.flush()
