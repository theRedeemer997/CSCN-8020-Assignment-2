from dataclasses import dataclass
import os, random, time, shutil
from typing import List, Dict, Tuple
import gym
import numpy as np
import matplotlib.pyplot as plt

# --- Settings ---
PLOT_STEPS = False
ENABLE_DECAY = False
EPISODES = 3000

random.seed(42)
np.random.seed(42)

# --- Auto-clean output directory ---
def ensure_outdir(path: str = "outputs") -> str:
    """Delete and recreate the outputs directory for each run."""
    if os.path.exists(path):
        shutil.rmtree(path)  # remove old files
    os.makedirs(path, exist_ok=True)
    return path

# --- Compatibility helpers ---
def compat_reset(env: gym.Env, *, seed=None) -> Tuple[int, dict]:
    try:
        return env.reset(seed=seed)
    except TypeError:
        if seed is not None:
            try:
                env.reset(seed=seed)
            except TypeError:
                env.seed(seed)
        obs = env.reset()
        return obs, {}

def compat_step(env: gym.Env, action) -> Tuple[int, float, bool, dict]:
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    obs, reward, done, info = out
    return obs, float(reward), bool(done), info


# --- Q-Learning Agent ---
@dataclass
class QConfig:
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.1
    episodes: int = EPISODES
    max_steps: int = 200
    seed: int = 42

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, cfg: QConfig):
        self.n_states = n_states
        self.n_actions = n_actions
        self.cfg = cfg
        self.Q = np.random.uniform(low=-1e-3, high=1e-3, size=(n_states, n_actions))
        self.epsilon = cfg.epsilon
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    def select_action(self, state: int, exploit_only: bool = False) -> int:
        if not exploit_only and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        best_next = 0.0 if done else float(np.max(self.Q[s_next]))
        target = r + self.cfg.gamma * best_next
        self.Q[s, a] += self.cfg.alpha * (target - self.Q[s, a])

    def train(self, env: gym.Env) -> Dict[str, List[float]]:
        episode_returns, episode_steps = [], []
        for ep in range(self.cfg.episodes):
            s, _ = compat_reset(env, seed=self.cfg.seed + ep)
            done, total_r, steps = False, 0.0, 0
            while not done and steps < self.cfg.max_steps:
                a = self.select_action(s)
                s_next, r, done, _ = compat_step(env, a)
                self.update(s, a, r, s_next, done)
                s, total_r, steps = s_next, total_r + r, steps + 1
            if ENABLE_DECAY:
                self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)
            episode_returns.append(total_r)
            episode_steps.append(steps)
        return {"returns": episode_returns, "steps": episode_steps}


# --- Utilities ---
def moving_average(x: List[float], k: int = 50) -> np.ndarray:
    if not x: return np.array([])
    k = max(1, min(k, len(x)))
    cumsum = np.cumsum(np.insert(x, 0, 0))
    ma = (cumsum[k:] - cumsum[:-k]) / float(k)
    if len(ma) == 0:
        return np.array(x, dtype=float)
    pad = np.concatenate([np.full(k - 1, ma[0]), ma])
    return pad

def plot_series(series: Dict[str, List[float]], title: str, ylabel: str, out_path: str):
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
    for k, v in summary.items():
        if k != "name":
            print(f"{k:15s}: {v:.3f}" if isinstance(v, float) else f"{k:15s}: {v}")
    return summary


# --- Automatically select best configuration ---
def pick_best_run(records: List[Dict[str, object]]) -> Dict[str, object]:
    def score(rec):
        s = rec["summary"]
        return (s["avg_return"], -s["avg_steps"], -s["total_steps"])
    return max(records, key=score)


# --- Run all experiments ---
def run_assignment_experiments(env: gym.Env, n_states: int, n_actions: int):
    outdir = ensure_outdir()
    all_runs = []

    # Baseline
    base_cfg = QConfig(alpha=0.1, gamma=0.9, epsilon=0.1, episodes=EPISODES)
    summary = run_single_training(env, n_states, n_actions, base_cfg, tag="baseline_a0.1_e0.1_g0.9", outdir=outdir)
    all_runs.append({"cfg": base_cfg, "tag": "baseline_a0.1_e0.1_g0.9", "summary": summary})

    # Learning rate α variations
    for a in [0.01, 0.001, 0.2]:
        cfg = QConfig(alpha=a, gamma=0.9, epsilon=0.1, episodes=EPISODES)
        tag = f"alpha_{a}"
        summary = run_single_training(env, n_states, n_actions, cfg, tag=tag, outdir=outdir)
        all_runs.append({"cfg": cfg, "tag": tag, "summary": summary})

    # Exploration factor ε variations
    for eps in [0.2, 0.3]:
        cfg = QConfig(alpha=0.1, gamma=0.9, epsilon=eps, episodes=EPISODES)
        tag = f"epsilon_{eps}"
        summary = run_single_training(env, n_states, n_actions, cfg, tag=tag, outdir=outdir)
        all_runs.append({"cfg": cfg, "tag": tag, "summary": summary})

    # Automatically select best
    best = pick_best_run(all_runs)
    best_cfg, best_tag = best["cfg"], best["tag"]
    print("\n>>> Auto-selected BEST configuration based on metrics:")
    print(f"    tag={best_tag}, α={best_cfg.alpha}, ε={best_cfg.epsilon}, γ={best_cfg.gamma}, episodes={best_cfg.episodes}")

    # Re-run best configuration
    final_tag = f"best_{best_tag}"
    run_single_training(env, n_states, n_actions, best_cfg, tag=final_tag, outdir=outdir)
    print("\nAll required experiments completed. Plots saved in ./outputs/")


# --- Wrapper for training ---
def run_single_training(env: gym.Env, n_states: int, n_actions: int, cfg: QConfig, tag: str, outdir: str):
    print(f"\n[TRAIN] {tag} (α={cfg.alpha}, γ={cfg.gamma}, ε={cfg.epsilon}, episodes={cfg.episodes})")
    agent = QLearningAgent(n_states, n_actions, cfg)
    metrics = agent.train(env)
    ret_ma = moving_average(metrics["returns"], k=50)
    plot_series({"raw": metrics["returns"], "ma": ret_ma},
                f"Returns per Episode – {tag}", "Episode Return",
                os.path.join(outdir, f"returns_{tag}.png"))
    if PLOT_STEPS:
        steps_ma = moving_average(metrics["steps"], k=50)
        plot_series({"raw": metrics["steps"], "ma": steps_ma},
                    f"Steps per Episode – {tag}", "Steps",
                    os.path.join(outdir, f"steps_{tag}.png"))
    return summarize_run(tag, metrics)


# --- Main ---
def main():
    env = gym.make("Taxi-v3")
    n_states, n_actions = env.observation_space.n, env.action_space.n
    print("Observation space size:", n_states)
    print("Number of actions:", n_actions)
    run_assignment_experiments(env, n_states, n_actions)

if __name__ == "__main__":
    main()
