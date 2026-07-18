import argparse
import copy
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from utils.config_helpers import apply_checkpoint_config
from utils.eval_helpers import run_eval
from utils.logging import Logger, dump_log
from utils.pytorch_utils import load_model_from_checkpoint_dir
from utils.model_helpers import EMA

from data import MinariSequenceDataset, collate_trajectory_batches
from models import DecisionDiffuser, TemporalUnet
from models.diffusion import GaussianDiffusion
from models.inv_dynamics import ARInverseDynamics, BasicInverseDynamics


def move_batch(batch, device):
    return (
        batch.trajectories.to(device),
        {step: value.to(device) for step, value in batch.conditions.items()},
        batch.returns.to(device),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--returns-condition", action="store_true")
    parser.add_argument("--condition-guidance-w", type=float, default=0.0)
    parser.add_argument("--returns-scale", type=float, default=1000.0)
    parser.add_argument("--ema-beta", type=float, default=0.995)
    parser.add_argument("--step-start-ema", type=int, default=2000)
    parser.add_argument("--eval-return", type=float)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-length", type=int, default=1000)
    parser.add_argument("--eval-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--checkpoint-dir")
    return parser.parse_args()


def main(logger: Logger, args: argparse.Namespace):
    if args.checkpoint_dir is not None:
        apply_checkpoint_config(args)

    device = torch.device(args.device)

    dataset = MinariSequenceDataset(
        args.dataset_id,
        horizon=args.horizon,
        download=args.download,
        returns_scale=args.returns_scale,
        include_returns=args.returns_condition,
        max_episodes=args.max_episodes,
    )
    if args.returns_condition:
        return_stats = dataset.get_sequence_return_stats()
        if args.eval_return is None:
            args.eval_return = return_stats["max"]
            print(
                "Using --eval-return "
                f"{args.eval_return:.4f} from dataset max scaled horizon return "
                f"(min={return_stats['min']:.4f}, mean={return_stats['mean']:.4f}, "
                f"max={return_stats['max']:.4f})."
            )
        elif args.eval_return < return_stats["min"] or args.eval_return > return_stats["max"]:
            print(
                "Warning: --eval-return "
                f"{args.eval_return:.4f} is outside the scaled training return range "
                f"[{return_stats['min']:.4f}, {return_stats['max']:.4f}]."
            )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_trajectory_batches,
    )

    denoiser = TemporalUnet(
        horizon=args.horizon,
        transition_dim=dataset.observation_dim,
        dim=args.dim,
        dim_mults=(1, 2, 4),
        returns_condition=args.returns_condition,
    )
    diffusion = GaussianDiffusion(
        n_timesteps=args.diffusion_steps,
        clip_denoised=True,
        predict_epsilon=True,
    )
    #inverse_dynamics = ARInverseDynamics(hidden_dim=args.hidden_dim, observation_dim=dataset.observation_dim, action_dim=dataset.action_dim,)

    inverse_dynamics = BasicInverseDynamics(
        hidden_dim=args.hidden_dim,
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
    )

    model = DecisionDiffuser(
        denoiser=denoiser,
        diffusion=diffusion,
        inverse_dynamics=inverse_dynamics,
        horizon=args.horizon,
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        batch_size=args.batch_size,
        device=device,
        returns_condition=args.returns_condition,
        condition_guidance_w=args.condition_guidance_w,
    ).to(device)

    ema = EMA(args.ema_beta, args.step_start_ema)

    ema_model = copy.deepcopy(model)

    if args.checkpoint_dir is not None:
        try:
            checkpoint_path = load_model_from_checkpoint_dir(
                model,
                args.checkpoint_dir,
                device,
            )
            print(f"Loaded checkpoint from {checkpoint_path}")
            model = torch.compile(model)
        except:
            model = torch.compile(model)
            checkpoint_path = load_model_from_checkpoint_dir(
                model,
                args.checkpoint_dir,
                device,
            )
            print(f"Loaded checkpoint from {checkpoint_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ema.reset_parameters(model, ema_model)

    model.train()
    data_iter = iter(dataloader)
    progress = tqdm(range(args.steps), desc="training", dynamic_ncols=True)
    for step in progress:
        step_start_time = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        trajectories, conditions, returns = move_batch(batch, device)
        if not args.returns_condition:
            returns = None

        loss, info = model.loss(trajectories, conditions, returns)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step%10 == 0:
            ema.step_ema(model, ema_model, step)

        train_row = {f"train/{key}": value for key, value in info.items()}
        train_row["train/lr"] = optimizer.param_groups[0]["lr"]
        train_row["train/step_time"] = time.perf_counter() - step_start_time
        if returns is not None:
            train_row["train/returns_mean"] = returns.mean()

        if args.eval_interval > 0 and (step + 1) % args.eval_interval == 0:
            eval_row = run_eval(ema_model, dataset, device, args, logger, step + 1)
            train_row.update(eval_row)

        logger.log(train_row, step + 1)

        if (
            args.checkpoint_interval > 0
            and (step + 1) % args.checkpoint_interval == 0
        ):
            dump_log(model, logger, args, logger.log_dir)

        progress.set_postfix(
            {key: f"{value.item():.4f}" for key, value in info.items()}
        )

    dump_log(model, logger, args, logger.log_dir, ema_model)


def make_logger(args: argparse.Namespace) -> Logger:
    logdir = "{}_{}_{}".format(
        args.dataset_id,
        args.diffusion_steps,
        time.strftime("%Y%m%d_%H%M%S"),
    )
    logdir = os.path.join("exp", logdir)
    os.makedirs(logdir, exist_ok=True)

    return Logger(log_dir=logdir, csv_path=os.path.join(logdir, "log.csv"))


if __name__ == "__main__":
    args = parse_args()
    logger = make_logger(args)
    try:
        main(logger, args)
    finally:
        logger.close()
