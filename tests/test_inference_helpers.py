"""Unit tests for the parsing and scoring helpers in inference.py.

These guard the two-line action-format contract that the trained policy
emits and the reward function consumes.
"""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("inference", os.path.join(ROOT, "inference.py"))
inf = importlib.util.module_from_spec(spec)
sys.modules["inference"] = inf
spec.loader.exec_module(inf)

OPTIONS = ["full cooperation", "limited disclosure", "seek legal delay"]


def test_parses_well_formed_two_line_completion():
    completion = "DECISION: full cooperation\nPITCH: A clean record protects long-term reputation."
    decision, pitch, ok = inf.parse_completion(completion, OPTIONS)
    assert decision == "full cooperation"
    assert "clean record" in pitch
    assert ok is True


def test_falls_back_to_substring_match_when_format_partially_broken():
    completion = "I think we should go with limited disclosure to manage risk."
    decision, _, ok = inf.parse_completion(completion, OPTIONS)
    assert decision == "limited disclosure"
    assert ok is False


def test_returns_first_option_when_completion_unintelligible():
    completion = "blah blah blah"
    decision, pitch, ok = inf.parse_completion(completion, OPTIONS)
    assert decision == OPTIONS[0]
    assert pitch == ""
    assert ok is False


def test_pitch_clipped_to_first_line():
    completion = "DECISION: seek legal delay\nPITCH: buying time preserves runway.\nExtra trailing line."
    _, pitch, ok = inf.parse_completion(completion, OPTIONS)
    assert "buying time preserves runway." in pitch
    assert "Extra trailing line" not in pitch
    assert ok is True


def test_keyword_pitch_score_rewards_role_alignment():
    cfo_text = "runway discipline and burn control protect compliance"
    cto_text = "engineering quality and team morale make velocity sustainable"
    assert inf.keyword_pitch_score(cfo_text, "CFO") >= 0.5
    assert inf.keyword_pitch_score(cto_text, "CTO") >= 0.5
    assert inf.keyword_pitch_score("totally unrelated text about cats", "CFO") == 0.0


def test_keyword_pitch_score_zero_on_empty_pitch():
    assert inf.keyword_pitch_score("", "CTO") == 0.0
    assert inf.keyword_pitch_score("   ", "CFO") == 0.0


def test_random_policy_act_returns_valid_option():
    class FakeObs:
        options = OPTIONS
        npc_statements = []
    pol = inf.RandomPolicy()
    decision, pitch, ok = pol.act(FakeObs())
    assert decision in OPTIONS
    assert pitch == ""
    assert ok is False


def test_run_summary_aggregates_metrics_correctly():
    eps = [
        inf.EpisodeMetrics(seed=1, total_reward=2.0, final_profitability=50.0,
                           survived=True, votes_won=7, votes_total=10,
                           pitches_written=4, avg_pitch_score=0.3),
        inf.EpisodeMetrics(seed=2, total_reward=4.0, final_profitability=70.0,
                           survived=True, votes_won=9, votes_total=10,
                           pitches_written=8, avg_pitch_score=0.5),
    ]
    s = inf.summarise("trained-test", eps)
    assert s.policy == "trained-test"
    assert s.n_episodes == 2
    assert abs(s.mean_reward - 3.0) < 1e-6
    assert abs(s.mean_profitability - 60.0) < 1e-6
    assert s.survival_rate == 1.0
    assert abs(s.win_rate_per_round - 0.8) < 1e-6
    assert abs(s.pitch_usage_rate - 0.6) < 1e-6
    assert abs(s.mean_pitch_score - 0.4) < 1e-6
