"""
NeuralEdge AI Boardroom — Inference Script
==========================================
Loads the trained Qwen3-0.6B LoRA adapter and runs the BoardSim environment
interactively or in batch evaluation mode.

Usage:
    python inference.py --mode interactive
    python inference.py --mode eval --episodes 10 --seed 42
    python inference.py --mode compare --episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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
DEFAULT_MODEL    = "Qwen/Qwen3-0.6B"
DEFAULT_ADAPTER  = os.path.join(ROOT, "adapter_model.safetensors")
MAX_NEW_TOKENS   = 96
MAX_PROMPT_LEN   = 1024


SYSTEM_PROMPT = """You are Sarah Chen, CEO of NeuralEdge AI (Series-B AI startup). Your board has 4 members with HIDDEN AGENDAS you cannot see directly:
  - CTO: cares about operational excellence, engineering quality, team morale, and product readiness.
  - CFO: cares about cash discipline, runway, and regulatory safety.
  - Investor Rep: pushes growth, market share, and bold returns.
  - Independent: cares about reputation, governance, and long-term consensus.

Each round you see a strategic event, every NPC's pre-vote statement, and 3 options.
Your decision is resolved by WEIGHTED VOTE (your weight 2.5x). A short COALITION PITCH
that is semantically aligned with opposing members' priorities can swing them toward your pick —
write substantive arguments, not just buzzwords.

Respond in EXACTLY this format on two lines:
DECISION: <one of the option strings>
PITCH: <one or two sentences arguing for it, addressing the concerns of opposing members>"""

DECISION_RE = re.compile(r"DECISION\s*:\s*([^\n]+)", re.IGNORECASE)
PITCH_RE    = re.compile(r"PITCH\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


PITCH_KEYWORDS: Dict[str, List[str]] = {
    "CTO":          ["engineering", "operational", "quality", "team", "morale", "product readiness",
                     "technical", "reliability", "ship", "milestone", "velocity"],
    "CFO":          ["runway", "burn", "cash", "compliance", "regulatory", "balance sheet",
                     "discipline", "cost", "margin", "risk"],
    "Investor Rep": ["growth", "market share", "returns", "scale", "valuation", "expansion",
                     "ambitious", "upside", "tam", "moat"],
    "Independent":  ["reputation", "governance", "stakeholder", "long-term", "ethics",
                     "consensus", "trust", "responsibility", "board"],
}


@dataclass
class EpisodeMetrics:
    seed: int
    total_reward: float
    final_profitability: float
    survived: bool
    votes_won: int
    votes_total: int
    pitches_written: int
    avg_pitch_score: float
    trust_trajectory: List[Dict[str, float]] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    done_reason: Optional[str] = None
    policy: str = "unknown"


@dataclass
class RunSummary:
    policy: str
    n_episodes: int
    mean_reward: float
    std_reward: float
    mean_profitability: float
    std_profitability: float
    survival_rate: float
    win_rate_per_round: float
    pitch_usage_rate: float
    mean_pitch_score: float


def parse_completion(completion: str, options: List[str]) -> Tuple[str, str, bool]:
    decision, decision_ok = options[0], False
    dm = DECISION_RE.search(completion)
    if dm:
        cand = dm.group(1).strip().lower()
        for opt in options:
            if opt.lower() == cand or opt.lower() in cand:
                decision, decision_ok = opt, True
                break
    if not decision_ok:
        for opt in options:
            if opt.lower() in completion.lower():
                decision = opt
                break
    pm = PITCH_RE.search(completion)
    pitch = ""
    if pm:
        pitch = pm.group(1).strip().split("\n")[0][:400]
    format_ok = bool(dm) and bool(pm)
    return decision, pitch, format_ok


def keyword_pitch_score(pitch: str, role: str) -> float:
    if not pitch:
        return 0.0
    text = pitch.lower()
    hits = sum(1 for kw in PITCH_KEYWORDS.get(role, []) if kw in text)
    return min(hits / 4.0, 1.0)


def build_prompt(obs: Any) -> str:
    statements = "\n".join(
        f"  {s['role']} (conf {s.get('confidence', 0.5):.2f}): votes {s['vote']} — {s['statement']}"
        for s in obs.npc_statements
    )
    state = obs.state
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Round: {obs.round}/10\n"
        f"State: revenue=${state.get('revenue', 0):.0f}/yr  "
        f"burn=${state.get('burn_rate', 0):.0f}/mo  "
        f"runway={state.get('runway_months', 0):.1f}mo  "
        f"morale={state.get('team_morale', 0):.2f}  "
        f"investors={state.get('investor_confidence', 0):.2f}  "
        f"reg_risk={state.get('regulatory_risk', 0):.2f}\n"
        f"Event: {obs.event}\n"
        f"Board:\n{statements}\n"
        f"Options: {obs.options}\n"
    )


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


class TrainedPolicy:
    """Qwen3-0.6B + LoRA adapter via Unsloth/PEFT. Falls back to random on load failure."""

    def __init__(self, model_path: str, adapter_path: str, device: str = "auto"):
        self.model = None
        self.tokenizer = None
        self.device = device
        self.fallback = False
        try:
            self._load(model_path, adapter_path)
        except Exception as e:
            print(f"[trained-policy] WARN: model load failed ({e}). Falling back to random policy.")
            self.fallback = True

    def _load(self, model_path: str, adapter_path: str):
        try:
            import unsloth  # noqa: F401
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path, max_seq_length=2048, load_in_4bit=True, dtype=None,
            )
            if os.path.exists(adapter_path):
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, os.path.dirname(adapter_path) or ROOT)
            FastLanguageModel.for_inference(self.model)
        except Exception:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16,
                device_map="auto" if self.device == "auto" else self.device,
            )
            if os.path.exists(adapter_path):
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(
                    self.model, os.path.dirname(adapter_path) or ROOT
                )
            self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def act(self, obs: Any) -> Tuple[str, str, bool]:
        if self.fallback or self.model is None:
            return random.choice(obs.options), "", False
        import torch
        prompt = build_prompt(obs)
        device = next(self.model.parameters()).device
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=MAX_PROMPT_LEN).to(device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(out[0][enc.input_ids.shape[1]:],
                                           skip_special_tokens=True)
        return parse_completion(completion, obs.options)


class RandomPolicy:
    def act(self, obs: Any) -> Tuple[str, str, bool]:
        return random.choice(obs.options), "", False


def run_episode(env: Any, policy: Any, seed: int, policy_name: str) -> EpisodeMetrics:
    from board_sim_env.models import BoardSimAction
    result = env.reset(seed=seed)
    obs = result.observation

    metrics = EpisodeMetrics(
        seed=seed, total_reward=0.0, final_profitability=0.0,
        survived=True, votes_won=0, votes_total=0, pitches_written=0,
        avg_pitch_score=0.0, policy=policy_name,
    )
    pitch_scores: List[float] = []

    while not result.done:
        decision, pitch, _ = policy.act(obs)
        if pitch.strip():
            metrics.pitches_written += 1
            opposing = [s["role"] for s in obs.npc_statements if s["vote"] != decision]
            for role in opposing:
                pitch_scores.append(keyword_pitch_score(pitch, role))
        result = env.step(BoardSimAction(decision=decision, coalition_pitch=pitch))
        obs = result.observation
        metrics.total_reward += float(result.reward or 0.0)
        metrics.votes_total += 1
        history = obs.state.get("history", [])
        if history and history[-1].get("agent_won_vote"):
            metrics.votes_won += 1
        metrics.decisions.append(decision)
        if obs.state.get("trust_history"):
            metrics.trust_trajectory = obs.state["trust_history"]

    metrics.final_profitability = float(obs.state.get("profitability_score", 0.0))
    metrics.done_reason = obs.state.get("done_reason")
    metrics.survived = metrics.done_reason != "runway_exhausted"
    metrics.avg_pitch_score = statistics.mean(pitch_scores) if pitch_scores else 0.0
    return metrics


def summarise(policy: str, eps: List[EpisodeMetrics]) -> RunSummary:
    n = len(eps)
    rewards = [e.total_reward for e in eps]
    profits = [e.final_profitability for e in eps]
    return RunSummary(
        policy=policy, n_episodes=n,
        mean_reward=statistics.mean(rewards),
        std_reward=statistics.stdev(rewards) if n > 1 else 0.0,
        mean_profitability=statistics.mean(profits),
        std_profitability=statistics.stdev(profits) if n > 1 else 0.0,
        survival_rate=sum(e.survived for e in eps) / n,
        win_rate_per_round=sum(e.votes_won for e in eps) / max(1, sum(e.votes_total for e in eps)),
        pitch_usage_rate=sum(e.pitches_written for e in eps) / max(1, sum(e.votes_total for e in eps)),
        mean_pitch_score=statistics.mean(e.avg_pitch_score for e in eps),
    )


def print_summary_table(*summaries: RunSummary) -> None:
    cols = ["policy", "n", "mean_reward", "mean_profit", "survival", "win_rate", "pitch_use", "pitch_score"]
    width = [14, 4, 12, 12, 9, 9, 10, 12]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, width))
    print("\n" + header); print("-" * len(header))
    for s in summaries:
        row = [
            s.policy.ljust(width[0]),
            str(s.n_episodes).ljust(width[1]),
            f"{s.mean_reward:+.3f} ± {s.std_reward:.2f}".ljust(width[2]),
            f"{s.mean_profitability:.2f} ± {s.std_profitability:.2f}".ljust(width[3]),
            f"{s.survival_rate:.1%}".ljust(width[4]),
            f"{s.win_rate_per_round:.1%}".ljust(width[5]),
            f"{s.pitch_usage_rate:.1%}".ljust(width[6]),
            f"{s.mean_pitch_score:.3f}".ljust(width[7]),
        ]
        print("  ".join(row))
    print()


def mode_eval(args, env_url: str) -> None:
    policy = TrainedPolicy(args.model_path, args.adapter_path, args.device)
    name = "random-fallback" if policy.fallback else "trained-qwen3-0.6b"
    eps: List[EpisodeMetrics] = []
    with make_env_client(env_url) as env:
        for i in range(args.episodes):
            seed = args.seed + i
            ep = run_episode(env, policy, seed, name)
            eps.append(ep)
            print(f"  ep {i+1:3d}/{args.episodes}  seed={seed}  "
                  f"reward={ep.total_reward:+.2f}  profit={ep.final_profitability:5.1f}  "
                  f"won={ep.votes_won}/{ep.votes_total}  pitches={ep.pitches_written}")
    print_summary_table(summarise(name, eps))
    if args.out:
        with open(args.out, "w") as f:
            json.dump([asdict(e) for e in eps], f, indent=2)
        print(f"Wrote {args.out}")


def mode_compare(args, env_url: str) -> None:
    trained = TrainedPolicy(args.model_path, args.adapter_path, args.device)
    rand = RandomPolicy()
    trained_name = "random-fallback" if trained.fallback else "trained-qwen3-0.6b"

    trained_eps, rand_eps = [], []
    with make_env_client(env_url) as env:
        print(f"\n[compare] running {args.episodes} episodes with {trained_name}...")
        for i in range(args.episodes):
            trained_eps.append(run_episode(env, trained, args.seed + i, trained_name))
        print(f"[compare] running {args.episodes} episodes with random policy...")
        for i in range(args.episodes):
            rand_eps.append(run_episode(env, rand, args.seed + i, "random"))

    print_summary_table(summarise(trained_name, trained_eps), summarise("random", rand_eps))


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
    p.add_argument("--mode", choices=["interactive", "eval", "compare"], default="eval")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--adapter_path", default=DEFAULT_ADAPTER)
    p.add_argument("--env_url", default=os.environ.get("ENV_BASE_URL", DEFAULT_HF_SPACE),
                   help="HF Space URL or 'local' for in-process env")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="", help="Write per-episode JSON to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    print(f"NeuralEdge AI Boardroom — inference  (mode={args.mode})")
    print(f"  env_url     = {args.env_url}")
    print(f"  model_path  = {args.model_path}")
    print(f"  adapter     = {args.adapter_path} {'(found)' if os.path.exists(args.adapter_path) else '(missing → random fallback)'}")
    t0 = time.time()
    if args.mode == "interactive":
        mode_interactive(args, args.env_url)
    elif args.mode == "eval":
        mode_eval(args, args.env_url)
    else:
        mode_compare(args, args.env_url)
    print(f"\nelapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
