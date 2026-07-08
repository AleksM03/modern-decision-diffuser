import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import minari
import numpy as np
from tqdm.auto import tqdm


IMAGE_KEYS = (
    "image",
    "image_obs",
    "pixels",
    "rgb",
    "rgb_array",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a video of the highest-return trajectory in a Minari dataset."
    )
    parser.add_argument("dataset_id")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--discount",
        type=float,
        default=1.0,
        help="Use discounted return for selecting the best episode.",
    )
    parser.add_argument(
        "--no-state-reset",
        action="store_true",
        help="Do not try to seed MuJoCo state from the first recorded observation.",
    )
    return parser.parse_args()


def episode_return(rewards: np.ndarray, discount: float) -> float:
    rewards = np.asarray(rewards, dtype=np.float32)
    discounts = discount ** np.arange(len(rewards), dtype=np.float32)
    return float(np.sum(discounts * rewards))


def iter_episodes(dataset: Any, max_episodes: int | None):
    for index, episode in enumerate(dataset.iterate_episodes()):
        if max_episodes is not None and index >= max_episodes:
            break
        yield index, episode


def find_best_episode(dataset: Any, max_episodes: int | None, discount: float):
    best = None
    for index, episode in tqdm(
        iter_episodes(dataset, max_episodes),
        desc="scanning episodes",
        dynamic_ncols=True,
    ):
        ret = episode_return(episode.rewards, discount)
        if best is None or ret > best[0]:
            best = (ret, index, episode)

    if best is None:
        raise ValueError("No episodes found in dataset.")
    return best


def get_observation_field(observations: Any, key: str) -> Any | None:
    if isinstance(observations, Mapping):
        if key in observations:
            return observations[key]
        return None
    return observations if key == "observation" else None


def find_recorded_frames(episode: Any) -> np.ndarray | None:
    observations = episode.observations
    for key in IMAGE_KEYS:
        values = get_observation_field(observations, key)
        if values is None:
            continue

        frames = np.asarray(values)
        if frames.ndim == 4 and frames.shape[-1] in (3, 4):
            return frames.astype(np.uint8)
        if frames.ndim == 4 and frames.shape[1] in (3, 4):
            return np.moveaxis(frames, 1, -1).astype(np.uint8)

    return None


def first_state_observation(episode: Any) -> np.ndarray | None:
    values = get_observation_field(episode.observations, "observation")
    if values is None:
        return None

    observation = np.asarray(values[0], dtype=np.float64).reshape(-1)
    return observation


def unwrap_env(env: Any) -> Any:
    current = env
    while hasattr(current, "env"):
        current = current.env
    return getattr(current, "unwrapped", current)


def try_set_mujoco_state_from_observation(env: Any, observation: np.ndarray) -> bool:
    env = unwrap_env(env)
    model = getattr(env, "model", None)
    if model is None or not hasattr(env, "set_state"):
        return False

    nq = int(getattr(model, "nq", 0))
    nv = int(getattr(model, "nv", 0))
    if nq <= 0 or nv <= 0:
        return False

    obs_len = int(observation.shape[0])
    if obs_len >= nq + nv:
        qpos = observation[:nq].copy()
        qvel = observation[nq : nq + nv].copy()
    elif obs_len >= (nq - 1) + nv:
        qpos = np.zeros(nq, dtype=np.float64)
        qpos[1:] = observation[: nq - 1]
        qvel = observation[nq - 1 : nq - 1 + nv].copy()
    else:
        return False

    env.set_state(qpos, qvel)
    return True


def reset_env_for_episode(env: Any, episode: Any) -> None:
    seed = getattr(episode, "seed", None)
    options = getattr(episode, "options", None)
    try:
        env.reset(seed=seed, options=options)
    except TypeError:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()


def render_frame(env: Any) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "env.render() returned None. The recovered environment may not support "
            "render_mode='rgb_array'."
        )
    return np.asarray(frame, dtype=np.uint8)


def recover_render_env(dataset: Any) -> Any:
    try:
        return dataset.recover_environment(render_mode="rgb_array")
    except TypeError:
        return dataset.recover_environment()


def step_env(env: Any, action: np.ndarray) -> tuple[bool, bool]:
    step_result = env.step(action)
    if len(step_result) == 5:
        _, _, terminated, truncated, _ = step_result
        return bool(terminated), bool(truncated)

    _, _, done, _ = step_result
    return bool(done), False


def replay_episode(dataset: Any, episode: Any, *, max_steps: int | None, set_state: bool):
    env = recover_render_env(dataset)
    frames = []
    try:
        reset_env_for_episode(env, episode)
        if set_state:
            observation = first_state_observation(episode)
            if observation is not None:
                restored = try_set_mujoco_state_from_observation(env, observation)
                if restored:
                    print("Restored MuJoCo qpos/qvel from the first observation.")
                else:
                    print("Could not restore MuJoCo state; replaying from env reset.")

        frames.append(render_frame(env))
        actions = np.asarray(episode.actions)
        if max_steps is not None:
            actions = actions[:max_steps]

        for action in tqdm(actions, desc="replaying actions", dynamic_ncols=True):
            terminated, truncated = step_env(env, action)
            frames.append(render_frame(env))
            if terminated or truncated:
                break
    finally:
        env.close()

    return np.stack(frames)


def write_video(frames: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def main() -> None:
    args = parse_args()
    dataset = minari.load_dataset(args.dataset_id, download=args.download)
    best_return, best_index, best_episode = find_best_episode(
        dataset,
        max_episodes=args.max_episodes,
        discount=args.discount,
    )

    output = args.output
    if output is None:
        safe_dataset_id = args.dataset_id.replace("/", "_")
        output = Path("exp") / f"{safe_dataset_id}_best_episode.mp4"

    frames = find_recorded_frames(best_episode)
    if frames is not None:
        if args.max_steps is not None:
            frames = frames[: args.max_steps + 1]
        source = "recorded image observations"
    else:
        frames = replay_episode(
            dataset,
            best_episode,
            max_steps=args.max_steps,
            set_state=not args.no_state_reset,
        )
        source = "environment action replay"

    write_video(frames, output, args.fps)
    print(
        f"Wrote {len(frames)} frames from episode {best_index} "
        f"(return={best_return:.4f}, source={source}) to {output}"
    )


if __name__ == "__main__":
    main()
