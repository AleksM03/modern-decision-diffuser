import numpy as np
import torch
from tqdm.auto import tqdm



def format_action(action, action_space):
    action = action.detach().cpu().numpy()
    action = np.clip(action, action_space.low.reshape(-1), action_space.high.reshape(-1))
    return action.reshape(action_space.shape)


def render_frame(env):
    frame = env.render()
    if frame is None:
        return None
    return np.asarray(frame, dtype=np.uint8)

def get_observation_from_dict(observation):
    if isinstance(observation, dict):
        return observation["observation"]
    return observation



@torch.no_grad()
def rollout_policy(model, env, dataset, device, args, *, record_video=False):
    observation, _ = env.reset()
    observation = get_observation_from_dict(observation)
    rewards = []
    actions = []
    frames = []
    success = None
    terminated = False
    truncated = False
    target_return = args.eval_return

    if record_video:
        frame = render_frame(env)
        if frame is not None:
            frames.append(frame)

    for _ in tqdm(range(args.eval_length), desc="eval", dynamic_ncols=True, leave=False):
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        if dataset.observation_normalizer is not None:
            observation = dataset.observation_normalizer.normalize(observation[None])[0]

        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        returns = None
        if args.returns_condition:
            returns = torch.full((1, 1), target_return, device=device)

        sampled_observations = model.sample(
            {0: observation},
            returns=returns,
            horizon=args.horizon,
        )
        next_observation = sampled_observations[:, 1]
        action = model.inverse_dynamics(
            torch.cat([observation, next_observation], dim=-1),
            deterministic=True,
        )[0]
        action = format_action(action, env.action_space)

        observation, reward, terminated, truncated, info = env.step(action)
        observation = get_observation_from_dict(observation)
        rewards.append(float(reward))
        actions.append(action.reshape(-1))

        if args.returns_condition:
            target_return = (target_return - float(reward) / args.returns_scale) / dataset.discount

        if "success" in info:
            success = float(info["success"])
        elif "is_success" in info:
            success = float(info["is_success"])

        if record_video:
            frame = render_frame(env)
            if frame is not None:
                frames.append(frame)

        if terminated or truncated:
            break

    actions = np.concatenate(actions)
    result = {
        "return": float(np.sum(rewards)),
        "length": len(rewards),
        "action_mean": float(actions.mean()),
        "action_std": float(actions.std()),
    }
    if success is not None:
        result["success"] = success
    if record_video and frames:
        result["trajectory"] = {"image_obs": np.stack(frames)}
    return result


def run_eval(model, dataset, device, args, logger, step):
    if args.eval_video:
        env = dataset.recover_environment(render_mode="rgb_array")
    else:
        env = dataset.recover_environment()

    was_training = model.training
    model.eval()

    try:
        results = [
            rollout_policy(
                model,
                env,
                dataset,
                device,
                args,
                record_video=args.eval_video,
            )
            for _ in range(args.eval_episodes)
        ]
    finally:
        env.close()
        if was_training:
            model.train()

    returns = np.asarray([result["return"] for result in results], dtype=np.float32)
    lengths = np.asarray([result["length"] for result in results], dtype=np.float32)

    eval_row = {
        "eval/return_mean": returns.mean(),
        "eval/return_std": returns.std(),
        "eval/return_min": returns.min(),
        "eval/return_max": returns.max(),
        "eval/length_mean": lengths.mean(),
        "eval/length_max": lengths.max(),
        "eval/action_mean": np.mean([result["action_mean"] for result in results]),
        "eval/action_std": np.mean([result["action_std"] for result in results]),
    }
    if args.returns_condition:
        eval_row["eval/target_return"] = args.eval_return

    successes = [result["success"] for result in results if "success" in result]
    if successes:
        eval_row["eval/success_rate"] = np.mean(successes)

    if args.eval_video:
        logger.log_trajs_as_videos(
            [result["trajectory"] for result in results if "trajectory" in result],
            step,
            max_videos_to_save=args.eval_episodes,
            fps=args.video_fps,
            video_title="eval/rollout",
        )
    return eval_row
