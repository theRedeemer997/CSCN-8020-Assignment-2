#!/usr/bin/python3
"""
CSCN8020 – Assignment 2 helper with Q-Learning
------------------------------------------------
- Tabular Q-Learning agent for Taxi-v3
- Runs required experiments:
    * Baseline: alpha=0.1, gamma=0.9, epsilon=0.1
    * Learning rate alpha ∈ {0.01, 0.001, 0.2}
    * Discount factor gamma ∈ {0.2, 0.3}
- Adds a final best re-run (alpha=0.2, gamma=0.9)
- Saves ONLY the required plots (returns) by default; step plots optional
- Prints required metrics:
    1) Total episodes
    2) Total steps (sum across episodes)
    3) Average return per episode
"""

from dataclasses import dataclass
import os
import random
import time
from typing import List, Dict, Tuple

import gym
import numpy as np
import matplotlib.pyplot as plt

# -------- Report settings --------
PLOT_STEPS = False   # keep False to generate only the required plots (returns)

# -------- Reproducibility --------
random.seed(42)
np.random.seed(42)

# -------------------------------------------------------------------
# Gym 0.25 / 0.26+ compatibility helpers (reset/step signatures differ)
# -------------------------------------------------------------------
def compat_reset(env: gym.Env, *, seed=None) -> Tuple[int, dict]:
    """Return (obs, info) for both gym<0.26 and gym>=0.26."""
    try:
        # gym>=0.26
        return env.reset(seed=seed)
    except TypeError:
        # gym<0.26 returns just obs
        if seed is not None:
            try:
                env.reset(seed=seed)
            except TypeError:
                # very old gym
                env.seed(seed)
        obs = env.reset()
        return obs, {}

def compat_step(env: gym.Env, action) -> Tuple[int, float, bool, dict]:
    """Return (obs, reward, done, info) for both gym<0.26 and gym>=0.26."""
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        # gym>=0.26
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    else:
        # gym<=0.25
        obs, reward, done, info = out
        return obs, float(reward), bool(done), info

# ---------------------------
# Helper functions
# ---------------------------
def describe_env(env: gym.Env):
    """Describe the environment's spaces, actions, and rewards."""
    num_actions = env.action_space.n               # Number of discrete actions
    obs = env.observation_space                    # Observation space
    num_obs = env.observation_space.n              # Number of discrete states
    reward_range = env.reward_range                # Min and max reward values

    action_desc = {
        0: "Move south (down)",
        1: "Move north (up)",
        2: "Move east (right)",
        3: "Move west (left)",
        4: "Pickup passenger",
        5: "Drop off passenger"
    }
    print("Observation space:", obs)
    print("Observation space size:", num_obs)
    print("Reward Range:", reward_range)
    print("Number of actions:", num_actions)
    print("Action description:", action_desc)
    return num_obs, num_actions

def get_action_description(action: int) -> str:
    action_desc = {
        0: "Move south (down)",
        1: "Move north (up)",
        2: "Move east (right)",
        3: "Move west (left)",
        4: "Pickup passenger",
        5: "Drop off passenger"
    }
    return action_desc[action]

def breakdown_obs(obs: int) -> Dict[str, int]:
    """
    Takes an observation for the 'Taxi-v3' environment and returns components.
    ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
    """
    destination = obs % 4
    obs //= 4
    passenger_location = obs % 5
    obs //= 5
    taxi_col = obs % 5
    obs //= 5
    taxi_row = obs
    return {
        "destination": int(destination),
        "passenger_location": int(passenger_location),
        "taxi_row": int(taxi_row),
        "taxi_col": int(taxi_col)
    }

def describe_obs(obs: int) -> None:
    obs_desc = {0: "Red", 1: "Green", 2: "Yellow", 3: "Blue", 4: "In taxi"}
    d = breakdown_obs(obs)
    print(
        "Passenger is at: {0}, wants to go to {1}. Taxi currently at ({2}, {3})".format(
            obs_desc[d["passenger_location"]],
            obs_desc[d["destination"]],
            d["taxi_row"],
            d["taxi_col"]
        )
    )

def simulate_episodes(env: gym.Env, agent, num_episodes: int = 3) -> None:
    """Visualize with exploit-only actions so you see the learned policy."""
    for _ in range(num_episodes):
        done = False
        state, _ = compat_reset(env)
        describe_obs(state)
        env.render()
        while not done:
            action = agent.select_action(state, exploit_only=True)
            env.render()
            time.sleep(0.08)
            next_state, _, done, _ = compat_step(env, action)
            state = next_state
        time.sleep(0.6)

# =======================================
# Q-Learning Agent Implementation
# =======================================
@dataclass
class QConfig:
    alpha: float = 0.1        # learning rate
    gamma: float = 0.9        # discount factor
    epsilon: float = 0.1      # exploration rate (fixed per assignment)
    epsilon_decay: float = 1.0  # keep fixed at 0.1 => no decay
    epsilon_min: float = 0.1  # min epsilon (not used when decay==1.0)
    episodes: int = 3000      # training episodes (assignment runs use 3000)
    max_steps: int = 200      # safety cap per episode
    seed: int = 42            # random seed

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, cfg: QConfig):
        self.n_states = n_states
        self.n_actions = n_actions
        self.cfg = cfg
        # slightly random init to break ties early
        self.Q = np.random.uniform(low=-1e-3, high=1e-3, size=(n_states, n_actions))
        self.epsilon = cfg.epsilon
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    def select_action(self, state: int, exploit_only: bool = False) -> int:
        """ε-greedy action selection."""
        if not exploit_only and random.random() < self.epsilon:
            return random.randrange(self.n_actions)   # explore
        return int(np.argmax(self.Q[state]))          # exploit

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        """Q-learning update."""
        best_next = 0.0 if done else float(np.max(self.Q[s_next]))
        target = r + self.cfg.gamma * best_next
        self.Q[s, a] += self.cfg.alpha * (target - self.Q[s, a])

    def train(self, env: gym.Env) -> Dict[str, List[float]]:
        """Run training for cfg.episodes episodes; return per-episode logs."""
        episode_returns: List[float] = []
        episode_steps: List[int] = []

        for ep in range(self.cfg.episodes):
            s, _ = compat_reset(env, seed=self.cfg.seed + ep)
            done = False
            total_r, steps = 0.0, 0

            while not done and steps < self.cfg.max_steps:
                a = self.select_action(s)
                s_next, r, done, _ = compat_step(env, a)
                self.update(s, a, r, s_next, done)
                s = s_next
                total_r += r
                steps += 1

            # epsilon fixed per assignment (no decay)
            episode_returns.append(total_r)
            episode_steps.append(steps)

        return {"returns": episode_returns, "steps": episode_steps}

# =============================================================
# Experiment and Plotting Utilities
# =============================================================
def moving_average(x: List[float], k: int = 50) -> np.ndarray:
    """Smooth values for cleaner plots."""
    if not x:
        return np.array([])
    k = max(1, min(k, len(x)))
    cumsum = np.cumsum(np.insert(x, 0, 0))
    ma = (cumsum[k:] - cumsum[:-k]) / float(k)
    if len(ma) == 0:
        return np.array(x, dtype=float)
    pad = np.concatenate([np.full(k - 1, ma[0]), ma])
    return pad

def ensure_outdir(path: str = "outputs") -> str:
    os.makedirs(path, exist_ok=True)
    return path

def plot_series(series: Dict[str, List[float]], title: str, ylabel: str, out_path: str) -> None:
    """Generic plotter used for both returns and (optionally) steps."""
    plt.figure(figsize=(9, 5))
    x = np.arange(1, len(series["raw"]) + 1)
    plt.plot(x, series["raw"], alpha=0.3, label="per-episode")
    plt.plot(x, series["ma"], linewidth=2.0, label="moving avg")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def summarize_run(name: str, metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """Print summary with exact assignment metrics wording."""
    ret, steps = metrics["returns"], metrics["steps"]
    summary = {
        "name": name,
        "episodes": len(ret),
        "avg_return": float(np.mean(ret)) if len(ret) else 0.0,
        "median_return": float(np.median(ret)) if len(ret) else 0.0,
        "avg_steps": float(np.mean(steps)) if len(steps) else 0.0,
        "median_steps": float(np.median(steps)) if len(steps) else 0.0,
        "total_steps": int(np.sum(steps)),
        "best_return": float(np.max(ret)) if len(ret) else 0.0,
    }

    print(f"\n=== {name} Summary ===")
    print(f"{'episodes':15s}: {summary['episodes']:.3f}")
    print(f"{'avg_return':15s}: {summary['avg_return']:.3f}")
    print(f"{'median_return':15s}: {summary['median_return']:.3f}")
    print(f"{'avg_steps':15s}: {summary['avg_steps']:.3f}")
    print(f"{'median_steps':15s}: {summary['median_steps']:.3f}")
    print(f"{'total_steps':15s}: {summary['total_steps']:d}")
    print(f"{'best_return':15s}: {summary['best_return']:.3f}")
    return summary

def run_single_training(env: gym.Env, n_states: int, n_actions: int,
                        cfg: QConfig, tag: str, outdir: str) -> Dict[str, float]:
    """Run one training session and generate plots + printed summary."""
    print(f"\n[TRAIN] {tag} (alpha={cfg.alpha}, gamma={cfg.gamma}, epsilon={cfg.epsilon}, episodes={cfg.episodes})")
    agent = QLearningAgent(n_states, n_actions, cfg)
    metrics = agent.train(env)

    # Returns plot (required)
    ret_ma = moving_average(metrics["returns"], k=50)
    plot_series(
        {"raw": metrics["returns"], "ma": ret_ma},
        f"Returns per Episode – {tag}",
        "Episode Return",
        os.path.join(outdir, f"returns_{tag}.png"),
    )

    # Steps plot (optional; off by default)
    if PLOT_STEPS:
        steps_ma = moving_average(metrics["steps"], k=50)
        plot_series(
            {"raw": metrics["steps"], "ma": steps_ma},
            f"Steps per Episode – {tag}",
            "Steps",
            os.path.join(outdir, f"steps_{tag}.png"),
        )

    # Summary (includes: total episodes, total steps, average return)
    return summarize_run(tag, metrics)

def run_assignment_experiments(env: gym.Env, n_states: int, n_actions: int) -> None:
    """Run baseline and parameter-variation experiments, plus a final best re-run."""
    outdir = ensure_outdir()

    # Baseline (assignment default)
    base_cfg = QConfig(alpha=0.1, gamma=0.9, epsilon=0.1, episodes=3000)
    run_single_training(env, n_states, n_actions, base_cfg,
                        tag="baseline_a0.1_g0.9_e0.1", outdir=outdir)

    # Learning rate α ∈ {0.01, 0.001, 0.2}
    for a in [0.01, 0.001, 0.2]:
        cfg = QConfig(alpha=a, gamma=0.9, epsilon=0.1, episodes=3000)
        run_single_training(env, n_states, n_actions, cfg,
                            tag=f"alpha_{a}", outdir=outdir)

    # Discount factor γ ∈ {0.2, 0.3}
    for g in [0.2, 0.3]:
        cfg = QConfig(alpha=0.1, gamma=g, epsilon=0.1, episodes=3000)
        run_single_training(env, n_states, n_actions, cfg,
                            tag=f"gamma_{g}", outdir=outdir)

    # Final best re-run (based on findings)
    best_cfg = QConfig(alpha=0.2, gamma=0.9, epsilon=0.1, episodes=3000)
    run_single_training(env, n_states, n_actions, best_cfg,
                        tag="best_alpha0.2_gamma0.9", outdir=outdir)

    print("\nPlots saved to ./outputs (returns_* only unless PLOT_STEPS=True). Include them in your report.")

# ---------------------------
# Main
# ---------------------------
def main():
    env = gym.make('Taxi-v3')
    num_obs, num_actions = describe_env(env)
    run_assignment_experiments(env, num_obs, num_actions)

if __name__ == "__main__":
    main()
