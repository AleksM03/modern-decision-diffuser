from collections import namedtuple
from dataclasses import dataclass
from typing import Any
from gymnasium.spaces.dict import Dict

import minari
import numpy as np
import torch
from torch.utils.data import Dataset


TrajectoryBatch = namedtuple("TrajectoryBatch", "trajectories conditions returns")


@dataclass(frozen=True)
class SequenceIndex:
    episode: int
    start: int
    end: int


class StandardNormalizer:
    def __init__(self, values: np.ndarray, eps: float = 1e-6) -> None:
        self.mean = values.mean(axis=0).astype(np.float32)
        self.std = values.std(axis=0).astype(np.float32)
        self.std = np.maximum(self.std, eps)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class MinariSequenceDataset(Dataset):
    def __init__(
        self,
        dataset_id: str,
        horizon: int,
        *,
        download: bool = False,
        discount: float = 0.99,
        returns_scale: float = 1.0,
        include_returns: bool = True,
        max_episodes: int | None = None,
        normalize_observations: bool = True,
    ) -> None:
        self.dataset_id = dataset_id
        self.horizon = horizon
        self.discount = discount
        self.returns_scale = returns_scale
        self.include_returns = include_returns
        self.normalize_observations = normalize_observations

        self.minari_dataset = minari.load_dataset(dataset_id, download=download)
        self.env = self.minari_dataset.recover_environment()
        if type(self.minari_dataset.observation_space) is Dict:
            self.observation_dim = int(np.prod(self.minari_dataset.observation_space["observation"].shape))
        else:
            self.observation_dim = int(np.prod(self.minari_dataset.observation_space.shape))
        self.action_dim = int(np.prod(self.minari_dataset.action_space.shape))

        episodes = []
        for i, episode in enumerate(self.minari_dataset.iterate_episodes()):
            if max_episodes is not None and i >= max_episodes:
                break

            if type(self.minari_dataset.observation_space) is Dict:
                observations = self._as_flat_float_array(episode.observations["observation"])
            else:
                observations = self._as_flat_float_array(episode.observations)
            actions = self._as_flat_float_array(episode.actions)
            rewards = np.asarray(episode.rewards, dtype=np.float32)

            if observations.shape[0] != actions.shape[0] + 1:
                raise ValueError(
                    "Expected Minari observations to contain the initial and final "
                    "observation, so len(observations) == len(actions) + 1."
                )

            episodes.append(
                {
                    "observations": observations,
                    "actions": actions,
                    "rewards": rewards,
                }
            )

        if not episodes:
            raise ValueError(f"No episodes loaded from Minari dataset {dataset_id!r}.")

        self.observation_normalizer = None
        if normalize_observations:
            all_observations = np.concatenate(
                [episode["observations"] for episode in episodes],
                axis=0,
            )
            self.observation_normalizer = StandardNormalizer(all_observations)
            for episode in episodes:
                episode["observations"] = self.observation_normalizer.normalize(
                    episode["observations"]
                ).astype(np.float32)

        self.episodes = episodes
        self.indices = self._make_indices()

    def _as_flat_float_array(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim < 2:
            array = array.reshape(array.shape[0], 1)
        return array.reshape(array.shape[0], -1)

    def _make_indices(self) -> list[SequenceIndex]:
        indices = []
        for episode_id, episode in enumerate(self.episodes):
            n_steps = episode["actions"].shape[0]
            max_start = n_steps - self.horizon + 1
            for start in range(max(0, max_start)):
                indices.append(
                    SequenceIndex(
                        episode=episode_id,
                        start=start,
                        end=start + self.horizon,
                    )
                )
        if not indices:
            raise ValueError(
                f"No sequence windows of horizon {self.horizon} are available."
            )
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> TrajectoryBatch:
        sequence_index = self.indices[index]
        episode = self.episodes[sequence_index.episode]
        start = sequence_index.start
        end = sequence_index.end

        observations = episode["observations"][start:end]
        actions = episode["actions"][start:end]
        trajectories = np.concatenate([actions, observations], axis=-1)
        conditions = {0: observations[0]}

        rewards = episode["rewards"][start:end]
        discounts = self.discount ** np.arange(len(rewards), dtype=np.float32)
        returns = np.array(
            [(discounts * rewards).sum() / self.returns_scale],
            dtype=np.float32,
        )

        return TrajectoryBatch(
            trajectories=torch.as_tensor(trajectories, dtype=torch.float32),
            conditions={
                step: torch.as_tensor(value, dtype=torch.float32)
                for step, value in conditions.items()
            },
            returns=torch.as_tensor(returns, dtype=torch.float32),
        )

    def get_sequence_return_stats(self) -> dict[str, float]:
        returns = []
        for sequence_index in self.indices:
            rewards = self.episodes[sequence_index.episode]["rewards"][
                sequence_index.start : sequence_index.end
            ]
            discounts = self.discount ** np.arange(len(rewards), dtype=np.float32)
            returns.append(float((discounts * rewards).sum() / self.returns_scale))

        returns = np.asarray(returns, dtype=np.float32)
        return {
            "min": float(returns.min()),
            "max": float(returns.max()),
            "mean": float(returns.mean()),
        }

    def unnormalize_observations(self, observations: np.ndarray) -> np.ndarray:
        if self.observation_normalizer is None:
            return observations
        return self.observation_normalizer.unnormalize(observations)

    def recover_environment(self, **kwargs):
        try:
            return self.minari_dataset.recover_environment(**kwargs)
        except TypeError:
            if kwargs:
                return self.minari_dataset.recover_environment()
            raise


def collate_trajectory_batches(batches: list[TrajectoryBatch]) -> TrajectoryBatch:
    condition_steps = batches[0].conditions.keys()
    conditions = {
        step: torch.stack([batch.conditions[step] for batch in batches])
        for step in condition_steps
    }

    return TrajectoryBatch(
        trajectories=torch.stack([batch.trajectories for batch in batches]),
        conditions=conditions,
        returns=torch.stack([batch.returns for batch in batches]),
    )
