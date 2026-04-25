"""Sanity-check random-policy script for BoardSimEnv.

NOTE: this is **not** the canonical baseline used in the headline
trained-vs-baseline comparison. The canonical baseline is
**base Qwen3-4B without LoRA**, computed inside `notebooks/train_grpo_v2.ipynb`
(and the mirrored `Training.py` script). A coin-flip is not a
competitive opponent for a 4 B language model choosing among 3
well-formed strings; we keep this script only as a quick env-health
check (it confirms the env is reachable and rewards stay in range).

Outputs:
  - assets/random_sanity.csv               per-episode final profitability
  - assets/random_sanity_distribution.png  histogram of final profitabilities
"""

from __future__ import annotations

import csv
import os
import random
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make `envs.board_sim_env...` importable.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "envs", "board_sim_env"))

from envs.board_sim_env.server.board_sim_env_environment import BoardSimEnvironment  # noqa: E402
from envs.board_sim_env.models import BoardSimAction  # noqa: E402


N_EPISODES = 200


def main() -> None:
    env = BoardSimEnvironment()
    final_profits: list[float] = []
    survival = 0
    total_reward_per_ep: list[float] = []

    for ep in range(N_EPISODES):
        obs = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        while not done:
            opts = obs.options
            if not opts:
                break
            obs = env.step(BoardSimAction(decision=random.choice(opts)))
            ep_reward += float(obs.reward or 0.0)
            done = obs.done
        final_profits.append(obs.state["profitability_score"])
        total_reward_per_ep.append(ep_reward)
        if obs.state.get("done_reason") != "runway_exhausted":
            survival += 1

    mean_p = statistics.mean(final_profits)
    std_p = statistics.stdev(final_profits)
    mean_r = statistics.mean(total_reward_per_ep)
    surv_rate = survival / N_EPISODES

    print(f"Random baseline over {N_EPISODES} episodes:")
    print(f"  mean final profitability = {mean_p:6.2f}  (std {std_p:.2f})")
    print(f"  mean total episode reward = {mean_r:6.2f}")
    print(f"  survival rate (no bankruptcy) = {surv_rate:.1%}")

    assets_dir = os.path.join(ROOT, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(assets_dir, "random_sanity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "final_profitability", "total_reward"])
        for i, (p, r) in enumerate(zip(final_profits, total_reward_per_ep)):
            w.writerow([i, f"{p:.4f}", f"{r:.4f}"])

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(final_profits, bins=20, color="#c44", edgecolor="white", alpha=0.85)
    plt.axvline(mean_p, color="black", linestyle="--", linewidth=2, label=f"mean = {mean_p:.1f}")
    plt.title(f"Random-policy baseline — final profitability ({N_EPISODES} episodes)")
    plt.xlabel("Final profitability score (0–100)")
    plt.ylabel("Episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "random_sanity_distribution.png"), dpi=120)
    plt.close()

    print(f"\nWrote {csv_path}")
    print(f"Wrote {os.path.join(assets_dir, 'random_sanity_distribution.png')}")


if __name__ == "__main__":
    main()
