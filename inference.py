"""
NeuralEdge AI Boardroom — Environment Testing Script
====================================================
A lightweight script to test the BoardSim environment and its reward function
without requiring an LLM. Runs predefined test cases or interactive mode to
demonstrate the environment dynamics.

Usage:
    python inference.py --mode interactive
    python inference.py --mode test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "envs"))
sys.path.insert(0, os.path.join(ROOT, "envs", "board_sim_env"))

DEFAULT_HF_SPACE = "https://stavankhobare-sst-metaxpytorch-hackathon.hf.space"

@dataclass
class EpisodeMetrics:
    seed: int
    total_reward: float
    final_profitability: float
    survived: bool
    votes_won: int
    votes_total: int
    pitches_written: int
    decisions: List[str] = field(default_factory=list)
    done_reason: Optional[str] = None
    policy: str = "unknown"

@contextmanager
def make_env_client(env_url: str):
    try:
        from board_sim_env.client import BoardSimEnv
    except Exception as e:
        raise RuntimeError(
            f"Cannot import BoardSimEnv client: {e}. "
            "Run from the repo root or `pip install -e envs/board_sim_env`."
        )
    if env_url.lower().startswith(("http://", "https://")):
        with BoardSimEnv(base_url=env_url).sync() as env:
            yield env
    else:
        from envs.board_sim_env.server.board_sim_env_environment import BoardSimEnvironment

        class _LocalEnv:
            def __init__(self):
                self._env = BoardSimEnvironment()

            def reset(self, seed: int = 0):
                obs = self._env.reset(seed=seed)
                return _Result(obs)

            def step(self, action):
                obs = self._env.step(action)
                return _Result(obs)

        @dataclass
        class _Result:
            observation: Any
            @property
            def reward(self): return float(self.observation.reward or 0.0)
            @property
            def done(self): return bool(self.observation.done)

        yield _LocalEnv()


class PredefinedPolicy:
    """A policy that uses predefined strategic responses to test the environment."""
    def __init__(self):
        # A dictionary mapping keywords in events to a specific decision and pitch
        self.strategies = [
            ("competitor", "double_down_on_quality", "Cutting prices destroys our margin. By investing in product quality, we protect runway through differentiation and keep engineering morale high."),
            ("talent", "partial_match", "We must retain our core engineering talent for operational excellence, but we'll do a partial match to preserve our cash runway."),
            ("regulatory", "full_compliance", "We cannot afford the risk of non-compliance. Regulatory safety ensures long-term consensus and investor trust."),
            ("acquisition", "reject_and_grow", "Selling now leaves money on the table. We have the runway and product readiness to grow and command a higher valuation later."),
            ("funding", "accept_terms", "We need this runway extension. The dilution is worth the safety and growth potential it unlocks."),
        ]

    def act(self, obs: Any) -> Tuple[str, str]:
        event_text = obs.event.lower()
        
        # Try to match a predefined strategy
        for keyword, pref_decision, pitch in self.strategies:
            if keyword in event_text:
                # Find the option that matches our preferred decision
                for opt in obs.options:
                    if pref_decision in opt.lower():
                        return opt, pitch
                        
        # Fallback if no specific strategy matches
        return obs.options[0], "This is the safest path forward to preserve runway and maintain stability."

class RandomPolicy:
    """A baseline policy that picks random actions."""
    def act(self, obs: Any) -> Tuple[str, str]:
        return random.choice(obs.options), ""


def run_episode(env: Any, policy: Any, seed: int, policy_name: str, verbose: bool = False) -> EpisodeMetrics:
    from board_sim_env.models import BoardSimAction
    result = env.reset(seed=seed)
    obs = result.observation

    metrics = EpisodeMetrics(
        seed=seed, total_reward=0.0, final_profitability=0.0,
        survived=True, votes_won=0, votes_total=0, pitches_written=0,
        policy=policy_name,
    )

    if verbose:
        print(f"\n--- Starting Episode (Seed: {seed}, Policy: {policy_name}) ---")

    while not result.done:
        decision, pitch = policy.act(obs)
        if pitch.strip():
            metrics.pitches_written += 1
            
        result = env.step(BoardSimAction(decision=decision, coalition_pitch=pitch))
        obs = result.observation
        
        step_reward = float(result.reward or 0.0)
        metrics.total_reward += step_reward
        metrics.votes_total += 1
        
        history = obs.state.get("history", [])
        won_vote = False
        if history and history[-1].get("agent_won_vote"):
            metrics.votes_won += 1
            won_vote = True
            
        metrics.decisions.append(decision)
        
        if verbose:
            print(f"Round {metrics.votes_total}: Decision: '{decision}'")
            if pitch:
                print(f"  Pitch: '{pitch}'")
            print(f"  Vote Won: {won_vote} | Step Reward: {step_reward:+.3f}")

    metrics.final_profitability = float(obs.state.get("profitability_score", 0.0))
    metrics.done_reason = obs.state.get("done_reason")
    metrics.survived = metrics.done_reason != "runway_exhausted"
    
    if verbose:
        print(f"--- Episode Finished ---")
        print(f"Final Profitability: {metrics.final_profitability:.2f}")
        print(f"Total Reward: {metrics.total_reward:+.3f} | Reason: {metrics.done_reason}\n")
        
    return metrics


def mode_test(args, env_url: str) -> None:
    print("\nNeuralEdge AI Boardroom — Environment Testing Mode")
    print("Running predefined test cases to demonstrate reward functionality...\n")
    
    policies = {
        "Predefined (Strategic)": PredefinedPolicy(),
        "Random (Baseline)": RandomPolicy()
    }
    
    with make_env_client(env_url) as env:
        for policy_name, policy in policies.items():
            print(f"\nEvaluating Policy: {policy_name}")
            print("=" * 60)
            
            # Run a couple of episodes with verbose output to demonstrate the working
            for i in range(args.episodes):
                seed = args.seed + i
                run_episode(env, policy, seed, policy_name, verbose=True)


def mode_interactive(args, env_url: str) -> None:
    from board_sim_env.models import BoardSimAction
    print("\nNeuralEdge AI Boardroom — interactive (human-play) mode")
    print("Type DECISION, then PITCH on a separate line. Empty input picks option[0].\n")
    with make_env_client(env_url) as env:
        result = env.reset(seed=args.seed)
        obs = result.observation
        ep_reward = 0.0
        while not result.done:
            print("=" * 70)
            print(f"Round {obs.round}/10 — score={obs.state.get('profitability_score', 0):.1f}  "
                  f"runway={obs.state.get('runway_months', 0):.1f}mo")
            print(f"Event: {obs.event}")
            for s in obs.npc_statements:
                print(f"  [{s['role']:13s}] votes {s['vote']:<28s} (conf {s.get('confidence', 0.5):.2f})")
                print(f"     {textwrap.fill(s['statement'], 90, subsequent_indent='     ')}")
            print(f"Options: {obs.options}")
            d_raw = input("DECISION: ").strip() or obs.options[0]
            decision = next((o for o in obs.options if o.lower() in d_raw.lower()), obs.options[0])
            pitch = input("PITCH:    ").strip()
            result = env.step(BoardSimAction(decision=decision, coalition_pitch=pitch))
            obs = result.observation
            ep_reward += float(result.reward or 0.0)
            print(f">>> reward {result.reward:+.3f}   cumulative {ep_reward:+.3f}")
        print(f"\nDONE. final profitability={obs.state.get('profitability_score', 0):.2f}  "
              f"reason={obs.state.get('done_reason')}  total_reward={ep_reward:+.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["interactive", "test"], default="test")
    p.add_argument("--env_url", default=os.environ.get("ENV_BASE_URL", "local"),
                   help="HF Space URL or 'local' for in-process env")
    p.add_argument("--episodes", type=int, default=2, help="Number of episodes to run per policy")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    print(f"NeuralEdge AI Boardroom — Environment Testing (mode={args.mode})")
    print(f"  env_url     = {args.env_url}")
    t0 = time.time()
    
    if args.mode == "interactive":
        mode_interactive(args, args.env_url)
    else:
        mode_test(args, args.env_url)
        
    print(f"\nelapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
