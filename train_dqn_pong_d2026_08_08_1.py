#!/usr/bin/env python3
"""Train a simple DQN agent on ALE/Pong-v5 as a headless batch job.

Example:
    python train_dqn_pong.py --episodes 200 --save-dir /workspace/models/pong_dqn
"""

from __future__ import annotations

import argparse
import collections
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Optional

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from stable_baselines3.common import atari_wrappers


DEFAULT_ENV_NAME = "ALE/Pong-v5"


@dataclass
class Config:
    env_name: str = DEFAULT_ENV_NAME
    episodes: int = 200
    max_frames: int = 2_000_000
    gamma: float = 0.99
    batch_size: int = 32
    replay_size: int = 100_000
    replay_start_size: int = 10_000
    learning_rate: float = 1e-4
    sync_target_frames: int = 1_000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.01
    epsilon_decay_frames: int = 150_000
    frame_stack: int = 4
    seed: int = 42
    log_every_episodes: int = 10


class DQN(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            conv_out_size = self.conv(torch.zeros(1, *input_shape)).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return self.fc(self.conv(x))


class ImageToPyTorch(gym.ObservationWrapper):
    """Convert observations from (H, W, C) to (C, H, W)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs = self.observation_space
        assert isinstance(obs, spaces.Box)
        assert len(obs.shape) == 3
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        self.observation_space = spaces.Box(
            low=obs.low.min(),
            high=obs.high.max(),
            shape=new_shape,
            dtype=obs.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.moveaxis(observation, 2, 0)


class BufferWrapper(gym.ObservationWrapper):
    """Stack the most recent N frames along the channel dimension."""

    def __init__(self, env: gym.Env, n_steps: int):
        super().__init__(env)
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        self.observation_space = spaces.Box(
            obs.low.repeat(n_steps, axis=0),
            obs.high.repeat(n_steps, axis=0),
            dtype=obs.dtype,
        )
        self.buffer: Deque[np.ndarray] = collections.deque(maxlen=n_steps)

    def reset(self, *, seed=None, options=None):
        self.buffer.clear()
        for _ in range(self.buffer.maxlen):
            self.buffer.append(np.zeros_like(self.env.observation_space.low))
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(self.buffer)


def make_env(env_name: str, n_steps: int, seed: int) -> gym.Env:
    # Gymnasium 1.x may require explicit ALE environment registration.
    try:
        gym.register_envs(ale_py)
    except AttributeError:
        pass

    try:
        env = gym.make(env_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create {env_name}. Make sure ale-py and the Atari ROMs "
            "are installed in the F3i batch image."
        ) from exc

    # Same general preprocessing strategy as the source notebook:
    # Atari preprocessing -> channel-first -> 4-frame history.
    env = atari_wrappers.AtariWrapper(env, clip_reward=False, noop_max=0)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=n_steps)
    env.action_space.seed(seed)
    return env


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    done: bool
    next_state: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Experience] = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, exp: Experience) -> None:
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> list[Experience]:
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]


class Agent:
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer, seed: int):
        self.env = env
        self.replay_buffer = replay_buffer
        self.seed = seed
        self.state: Optional[np.ndarray] = None
        self.episode_reward = 0.0
        self.reset()

    def reset(self) -> None:
        self.state, _ = self.env.reset(seed=self.seed)
        self.seed += 1
        self.episode_reward = 0.0

    @torch.no_grad()
    def play_step(self, net: DQN, device: torch.device, epsilon: float) -> Optional[float]:
        assert self.state is not None

        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = torch.as_tensor(self.state, device=device).unsqueeze(0)
            q_vals = net(state_v)
            action = int(q_vals.argmax(dim=1).item())

        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.episode_reward += float(reward)

        self.replay_buffer.append(
            Experience(
                state=self.state,
                action=action,
                reward=float(reward),
                done=done,
                next_state=next_state,
            )
        )
        self.state = next_state

        if done:
            finished_reward = self.episode_reward
            self.reset()
            return finished_reward
        return None


def batch_to_tensors(batch: list[Experience], device: torch.device):
    states = torch.as_tensor(np.asarray([e.state for e in batch]), device=device)
    actions = torch.as_tensor([e.action for e in batch], dtype=torch.long, device=device)
    rewards = torch.as_tensor([e.reward for e in batch], dtype=torch.float32, device=device)
    dones = torch.as_tensor([e.done for e in batch], dtype=torch.bool, device=device)
    next_states = torch.as_tensor(np.asarray([e.next_state for e in batch]), device=device)
    return states, actions, rewards, dones, next_states


def calc_loss(
    batch: list[Experience],
    net: DQN,
    target_net: DQN,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    states, actions, rewards, dones, next_states = batch_to_tensors(batch, device)

    q_values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1).values
        next_q_values[dones] = 0.0
        expected_q_values = rewards + gamma * next_q_values

    return nn.MSELoss()(q_values, expected_q_values)


def save_checkpoint(
    path: Path,
    net: DQN,
    config: Config,
    obs_shape: tuple[int, ...],
    n_actions: int,
    frame_idx: int,
    episode_idx: int,
    mean_reward: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": net.state_dict(),
        "config": asdict(config),
        "obs_shape": tuple(obs_shape),
        "n_actions": int(n_actions),
        "frame_idx": int(frame_idx),
        "episode_idx": int(episode_idx),
        "mean_reward_100": float(mean_reward),
    }
    torch.save(checkpoint, path)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN on Pong")

    p.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of completed games to train"
    )

    p.add_argument(
        "--max-frames",
        type=int,
        default=2_000_000,
        help="Safety cap on environment frames"
    )

    p.add_argument(
        "--save-dir",
        type=Path,
        default=Path("models"),
        help="Directory where model checkpoints are saved"
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    p.add_argument(
        "--replay-size",
        type=int,
        default=100_000
    )

    p.add_argument(
        "--replay-start-size",
        type=int,
        default=10_000
    )

    p.add_argument(
        "--epsilon-decay-frames",
        type=int,
        default=150_000
    )

    p.add_argument(
        "--log-every-episodes",
        type=int,
        default=10
    )

    return p.parse_args()

def main() -> None:
    args = parse_args()

    config = Config(
        episodes=args.episodes,
        max_frames=args.max_frames,
        replay_size=args.replay_size,
        replay_start_size=args.replay_start_size,
        epsilon_decay_frames=args.epsilon_decay_frames,
        seed=args.seed,
        log_every_episodes=args.log_every_episodes,
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}", flush=True)

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    env = make_env(config.env_name, config.frame_stack, config.seed)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    assert obs_shape is not None

    print(f"Environment: {config.env_name}", flush=True)
    print(f"Observation shape: {obs_shape}; actions: {n_actions}", flush=True)
    print(f"Training for {config.episodes} completed episodes", flush=True)
    print(f"Saving checkpoints to: {args.save_dir}", flush=True)

    net = DQN(obs_shape, n_actions).to(device)

    target_net = DQN(obs_shape, n_actions).to(device)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
    )

    replay_buffer = ReplayBuffer(config.replay_size)
    agent = Agent(env, replay_buffer, config.seed)

    rewards: list[float] = []

    best_mean_reward = float("-inf")

    frame_idx = 0
    episode_idx = 0

    # ---------------------------------------------------------
    # Timing information
    # ---------------------------------------------------------

    start_time = time.perf_counter()

    episode_start_time = start_time
    episode_start_frame = 0

    training_start_time = None
    training_updates = 0

    try:

        while (
            episode_idx < config.episodes
            and frame_idx < config.max_frames
        ):

            frame_idx += 1

            epsilon = max(
                config.epsilon_final,
                config.epsilon_start
                - frame_idx / config.epsilon_decay_frames,
            )

            reward = agent.play_step(
                net,
                device,
                epsilon,
            )

            # -------------------------------------------------
            # Episode completed
            # -------------------------------------------------

            if reward is not None:

                episode_idx += 1
                rewards.append(reward)

                mean_reward = float(
                    np.mean(rewards[-100:])
                )

                now = time.perf_counter()

                # Time spent in this episode
                episode_seconds = (
                    now - episode_start_time
                )

                episode_frames = (
                    frame_idx - episode_start_frame
                )

                episode_fps = (
                    episode_frames / episode_seconds
                    if episode_seconds > 0
                    else 0.0
                )

                # Time for the entire run
                total_seconds = (
                    now - start_time
                )

                overall_fps = (
                    frame_idx / total_seconds
                    if total_seconds > 0
                    else 0.0
                )

                if (
                    episode_idx == 1
                    or episode_idx % config.log_every_episodes == 0
                    or episode_idx == config.episodes
                ):

                    print(
                        f"episode={episode_idx:5d}/{config.episodes} "
                        f"frame={frame_idx:9d} "
                        f"reward={reward:6.1f} "
                        f"mean100={mean_reward:7.3f} "
                        f"epsilon={epsilon:.3f} "
                        f"episode_time={episode_seconds:6.2f}s "
                        f"episode_fps={episode_fps:7.1f} "
                        f"overall_fps={overall_fps:7.1f} "
                        f"elapsed={total_seconds / 60.0:6.2f} min",
                        flush=True,
                    )

                # Reset timer for next episode
                episode_start_time = now
                episode_start_frame = frame_idx

                # Save best model
                if mean_reward > best_mean_reward:

                    best_mean_reward = mean_reward

                    save_checkpoint(
                        args.save_dir / "pong_dqn_best.pt",
                        net,
                        config,
                        obs_shape,
                        n_actions,
                        frame_idx,
                        episode_idx,
                        mean_reward,
                    )

            # -------------------------------------------------
            # Wait until replay buffer has enough experience
            # -------------------------------------------------

            if len(replay_buffer) < config.replay_start_size:
                continue

            # -------------------------------------------------
            # Actual neural-network training begins here
            # -------------------------------------------------

            if training_start_time is None:

                training_start_time = time.perf_counter()

                print(
                    f"\nReplay buffer reached "
                    f"{config.replay_start_size} frames."
                    f"\nGradient training begins "
                    f"at frame {frame_idx}.\n",
                    flush=True,
                )

            # Periodically synchronize target network
            if frame_idx % config.sync_target_frames == 0:

                target_net.load_state_dict(
                    net.state_dict()
                )

            # -------------------------------------------------
            # One gradient update
            # -------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            batch = replay_buffer.sample(
                config.batch_size
            )

            loss = calc_loss(
                batch,
                net,
                target_net,
                config.gamma,
                device,
            )

            loss.backward()

            optimizer.step()

            training_updates += 1

        # -----------------------------------------------------
        # Training loop completed
        # -----------------------------------------------------

        final_mean = (
            float(np.mean(rewards[-100:]))
            if rewards
            else float("nan")
        )

        save_checkpoint(
            args.save_dir / "pong_dqn_final.pt",
            net,
            config,
            obs_shape,
            n_actions,
            frame_idx,
            episode_idx,
            final_mean,
        )

        total_seconds = (
            time.perf_counter() - start_time
        )

        overall_fps = (
            frame_idx / total_seconds
            if total_seconds > 0
            else 0.0
        )

        print(
            "\nTraining finished.",
            flush=True,
        )

        print(
            f"Completed episodes: {episode_idx}",
            flush=True,
        )

        print(
            f"Frames: {frame_idx}",
            flush=True,
        )

        print(
            f"Gradient updates: {training_updates}",
            flush=True,
        )

        print(
            f"Total time: {total_seconds:.2f} sec "
            f"({total_seconds / 60.0:.2f} min)",
            flush=True,
        )

        print(
            f"Overall throughput: "
            f"{overall_fps:.1f} frames/sec",
            flush=True,
        )

        print(
            f"Final mean reward (last 100): "
            f"{final_mean:.3f}",
            flush=True,
        )

        print(
            f"Best model:  "
            f"{args.save_dir / 'pong_dqn_best.pt'}",
            flush=True,
        )

        print(
            f"Final model: "
            f"{args.save_dir / 'pong_dqn_final.pt'}",
            flush=True,
        )

    finally:

        env.close()

if __name__ == "__main__":
    main()
