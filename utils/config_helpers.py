import argparse

from utils.pytorch_utils import load_checkpoint_config


def apply_checkpoint_config(args: argparse.Namespace) -> None:

    config = load_checkpoint_config(args.checkpoint_dir)
    checkpoint_dataset_id = config.get("dataset_id")
    if checkpoint_dataset_id is not None and checkpoint_dataset_id != args.dataset_id:
        raise ValueError(
            "Checkpoint dataset_id "
            f"{checkpoint_dataset_id!r} does not match {args.dataset_id!r}."
        )

    for key in (
        "horizon",
        "diffusion_steps",
        "dim",
        "hidden_dim",
        "returns_condition",
        "returns_scale",
        "max_episodes",
    ):
        if key in config:
            setattr(args, key, config[key])
