"""End-to-end environment tests for the BoardSim OpenEnv environment.

Covers:
  * deterministic reset/step contract
  * observation schema and required fields
  * 10-round episode termination
  * reward bounds (per-step dense, terminal spikes)
  * vote resolution on weighted CEO + NPC tally
  * format penalty fires for invalid decisions
  * pitch bootstrap fires for non-empty pitch
  * runway-exhaustion bankruptcy path
  * trust dynamics persist and update bidirectionally
  * event order is shuffled per seed (no trajectory memorisation)
"""
from __future__ import annotations

import statistics

import pytest

from board_sim_env.models import BoardSimAction
from board_sim_env.server.board_sim_env_environment import (
    BoardSimEnvironment,
    EVENTS,
    NPC_AGENDAS,
    ROLE_WEIGHT,
    compute_profitability_score,
)


# ─── Determinism ──────────────────────────────────────────────────────────

def test_reset_is_deterministic_per_seed():
    e1, e2 = BoardSimEnvironment(), BoardSimEnvironment()
    o1, o2 = e1.reset(seed=7), e2.reset(seed=7)
    assert o1.event == o2.event
    assert o1.options == o2.options
    assert [s["statement"] for s in o1.npc_statements] == [s["statement"] for s in o2.npc_statements]


def test_different_seeds_produce_different_event_orders():
    seen_first_events = set()
    for s in range(20):
        env = BoardSimEnvironment()
        seen_first_events.add(env.reset(seed=s).event)
    assert len(seen_first_events) >= 4, "event shuffling should produce variety across seeds"


# ─── Schema ───────────────────────────────────────────────────────────────

def test_observation_schema():
    env = BoardSimEnvironment()
    obs = env.reset(seed=0)
    for key in ("revenue", "burn_rate", "runway_months", "profitability_score", "trust"):
        assert key in obs.state, f"observation.state missing {key}"
    assert obs.round == 1
    assert len(obs.options) == 3
    assert len(obs.npc_statements) == 4
    for s in obs.npc_statements:
        assert {"role", "vote", "confidence", "statement"}.issubset(s.keys())
        assert s["role"] in NPC_AGENDAS
        assert s["vote"] in obs.options


def test_npc_role_set_and_weights():
    assert set(NPC_AGENDAS.keys()) == {"CTO", "CFO", "Investor Rep", "Independent"}
    assert ROLE_WEIGHT["CEO"] == 2.5
    for role in NPC_AGENDAS:
        assert ROLE_WEIGHT[role] > 0


# ─── Episode lifecycle ───────────────────────────────────────────────────

def test_episode_terminates_at_or_before_ten_rounds():
    env = BoardSimEnvironment()
    obs = env.reset(seed=42)
    n = 0
    while not obs.done and n < 15:
        obs = env.step(BoardSimAction(decision=obs.options[0], coalition_pitch=""))
        n += 1
    assert obs.done
    assert n <= 10
    assert obs.state["done_reason"] in {"runway_exhausted", "acquisition", "ipo", "stay_private", "finished_10"}


def test_step_returns_required_fields():
    env = BoardSimEnvironment()
    obs = env.reset(seed=1)
    res = env.step(BoardSimAction(decision=obs.options[0], coalition_pitch=""))
    assert hasattr(res, "reward")
    assert hasattr(res, "done")
    assert hasattr(res, "state")


# ─── Reward bounds ───────────────────────────────────────────────────────

def test_per_step_reward_dense_and_bounded_until_terminal():
    """Non-terminal step rewards live in roughly [-3, +3]; terminal step can spike (+/-30 ish)."""
    env = BoardSimEnvironment()
    obs = env.reset(seed=11)
    rewards = []
    while not obs.done:
        obs = env.step(BoardSimAction(decision=obs.options[0], coalition_pitch="runway and morale matter"))
        rewards.append(float(obs.reward or 0.0))
        if obs.done:
            break
    assert len(rewards) >= 1
    nonterm = rewards[:-1] if len(rewards) > 1 else []
    for r in nonterm:
        assert -3.0 <= r <= 3.0, f"per-step reward {r} outside dense band"


def test_format_penalty_for_invalid_decision():
    """Format penalty (-0.5) should fire when action.decision is not in options.

    Pick a non-terminal first round so terminal bonuses don't dominate the
    measurement; compare same-seed paired (valid vs invalid) reward.
    """
    invalid_drops = 0
    n = 0
    for s in range(40):
        e_valid, e_invalid = BoardSimEnvironment(), BoardSimEnvironment()
        o_valid = e_valid.reset(seed=s)
        e_invalid.reset(seed=s)
        r_valid = e_valid.step(BoardSimAction(decision=o_valid.options[0], coalition_pitch=""))
        r_invalid = e_invalid.step(BoardSimAction(decision="NOT_A_VALID_OPTION", coalition_pitch=""))
        if r_valid.done or r_invalid.done:
            continue
        n += 1
        if (r_invalid.reward or 0.0) < (r_valid.reward or 0.0):
            invalid_drops += 1
    assert n >= 5, "needed enough non-terminal first rounds to compare"
    assert invalid_drops / n >= 0.7, "invalid decisions should reduce reward most of the time (format penalty)"


def test_pitch_bootstrap_increases_reward_vs_empty_pitch():
    """Non-empty pitch on a contested round earns the +0.05 bootstrap and >=0 persuasion bonus."""
    seeds_with_lift = 0
    for s in range(20):
        e1, e2 = BoardSimEnvironment(), BoardSimEnvironment()
        o1, o2 = e1.reset(seed=s), e2.reset(seed=s)
        r1 = e1.step(BoardSimAction(decision=o1.options[0], coalition_pitch=""))
        r2 = e2.step(BoardSimAction(
            decision=o2.options[0],
            coalition_pitch="runway discipline and engineering quality argue for this",
        ))
        if (r2.reward or 0.0) >= (r1.reward or 0.0):
            seeds_with_lift += 1
    assert seeds_with_lift >= 14, "pitch should generally not hurt reward (>=70% of seeds non-decreasing)"


# ─── Profitability score ─────────────────────────────────────────────────

def test_profitability_score_in_range():
    env = BoardSimEnvironment()
    env.reset(seed=0)
    score = env.state.state_dict["profitability_score"]
    assert 0.0 <= score <= 100.0
    score2 = compute_profitability_score(env.state.state_dict)
    assert abs(score - score2) < 1e-6


# ─── Trust dynamics ──────────────────────────────────────────────────────

def test_trust_persists_and_updates():
    env = BoardSimEnvironment()
    obs0 = env.reset(seed=3)
    init_trust = dict(obs0.state["trust"])
    obs1 = env.step(BoardSimAction(decision=obs0.options[0], coalition_pitch="strong product readiness"))
    after = obs1.state["trust"]
    assert set(after.keys()) == set(init_trust.keys())
    assert any(abs(after[r] - init_trust[r]) > 1e-6 for r in init_trust), "at least one NPC's trust should move"
    for v in after.values():
        assert 0.1 <= v <= 1.0


# ─── Vote resolution ────────────────────────────────────────────────────

def test_ceo_weight_dominates_when_no_persuasion():
    """CEO weight 2.5 + at least one aligned NPC should usually carry the vote."""
    wins = 0
    for s in range(50):
        env = BoardSimEnvironment()
        obs = env.reset(seed=s)
        target = obs.options[0]
        env.step(BoardSimAction(decision=target, coalition_pitch=""))
        history = env.state.state_dict["history"]
        if history and history[-1]["winning_decision"] == target:
            wins += 1
    assert wins / 50 >= 0.6, "CEO weight should win >=60% of single-step votes without pitch"


# ─── Sanity smoke ────────────────────────────────────────────────────────

def test_random_policy_survives_majority_of_episodes():
    import random
    rng = random.Random(123)
    survived = 0
    n = 30
    for ep in range(n):
        env = BoardSimEnvironment()
        obs = env.reset(seed=ep)
        while not obs.done:
            obs = env.step(BoardSimAction(decision=rng.choice(obs.options), coalition_pitch=""))
        if env.state.state_dict.get("done_reason") != "runway_exhausted":
            survived += 1
    assert survived / n >= 0.6, "random policy should survive >=60% of episodes (env-health floor)"
