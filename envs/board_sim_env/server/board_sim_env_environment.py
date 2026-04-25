# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""NeuralEdge AI Boardroom — OpenEnv environment.

The agent plays the CEO of a Series B AI startup. Each of 10 rounds it sees
a market-crisis event, statements + votes from 4 hidden-agenda NPC board
members, and must pick one of 3 decisions. Decisions are resolved by a
weighted vote and produce dense reward proportional to a composite
profitability score plus coalition / trust shaping terms.

NPCs are deterministic-given-(seed, round, state) — same observation in
training and resolution — so GRPO has a stable target to learn against.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import BoardSimAction, BoardSimObservation, BoardState
except ImportError:  # direct script execution: `python server/board_sim_env_environment.py`
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import BoardSimAction, BoardSimObservation, BoardState  # type: ignore


# ---------------------------------------------------------------------------
# Static config
# ---------------------------------------------------------------------------

# Per-role weighted vote influence (CEO is the agent).
ROLE_WEIGHT: Dict[str, float] = {
    "CEO": 1.5,
    "CTO": 1.2,
    "CFO": 1.0,
    "Investor Rep": 1.3,
    "Independent": 0.8,
}

<<<<<<< HEAD
# NPCs and their BASE hidden agendas. At episode reset() these are
# jittered per-seed so no single optimal decision path exists across episodes.
# The agent never sees the final per-episode weights — it must infer them
# from observable statements + vote history (Theory of Mind).
NPC_AGENDAS_BASE: Dict[str, Dict[str, float]] = {
=======
# NPCs and their hidden agendas: weights on per-step state-deltas they
# privately maximize. The agent never sees these.
NPC_AGENDAS: Dict[str, Dict[str, float]] = {
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
    # CTO — wants product strength + team morale; hates burn.
    "CTO": {
        "product_readiness": 0.55,
        "team_morale": 0.40,
        "burn_rate": -0.10,
        "regulatory_risk": -0.05,
    },
    # CFO — burn discipline, runway, regulatory caution.
    "CFO": {
        "burn_rate": -0.60,
        "revenue": 0.30,
        "runway_months": 0.20,
        "regulatory_risk": -0.25,
    },
    # Investor Rep — growth-at-all-costs.
    "Investor Rep": {
        "investor_confidence": 0.45,
        "market_share": 0.35,
        "revenue": 0.25,
        "burn_rate": -0.05,
    },
    # Independent — reputation/safety; consensus seeker.
    "Independent": {
        "regulatory_risk": -0.45,
        "team_morale": 0.30,
        "investor_confidence": 0.25,
        "market_share": 0.10,
    },
}

<<<<<<< HEAD
# Keep a module-level alias for backwards compatibility.
NPC_AGENDAS: Dict[str, Dict[str, float]] = NPC_AGENDAS_BASE


def _jitter_agendas(seed: int) -> Dict[str, Dict[str, float]]:
    """Return per-episode NPC agenda weights by adding seeded noise (±25%)
    to the base weights.  Signs are preserved so the qualitative role
    identity stays intact (CFO still cares about burn; CTO about product),
    but the *magnitude* varies — forcing the agent to infer fresh priorities
    each episode rather than memorising a fixed optimal sequence.
    """
    rng = random.Random(seed ^ 0xDEADBEEF)  # distinct stream from NPC rng
    jittered: Dict[str, Dict[str, float]] = {}
    for role, agenda in NPC_AGENDAS_BASE.items():
        jittered[role] = {}
        for field, w in agenda.items():
            # Jitter: multiply by U[0.75, 1.25], keep sign.
            factor = rng.uniform(0.75, 1.25)
            jittered[role][field] = round(w * factor, 4)
    return jittered

=======
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
# Personality phrase banks for flavorful statements. State-aware: separate
# phrase pools for "calm" vs "crisis" mode are selected based on current
# state (low runway / low morale / high reg risk → crisis variant).
PHRASES: Dict[str, Dict[str, List[str]]] = {
    "CTO": {
        "calm": [
            "Look, the architecture won't survive shortcuts here.",
            "I've sketched the trade-offs — engineering's pretty clear.",
            "If we ship before this is solid, we eat it in support tickets.",
            "Frankly, our infra dictates this choice more than any of you realize.",
        ],
        "crisis": [
            "Team is one bad sprint from a mass exit. Pick carefully.",
            "I cannot keep papering over technical debt with sprint heroics.",
            "Our incident channel is on fire; this isn't the moment for bold strokes.",
        ],
    },
    "CFO": {
        "calm": [
            "The numbers do not lie, and right now they're whispering.",
            "I'd like the board minutes to reflect my reservations.",
            "From a fiduciary standpoint, only one of these is defensible.",
        ],
        "crisis": [
            "Runway is the only KPI that matters at this table right now.",
            "I have spreadsheets that show this is how startups die. Slowly.",
            "Cash is king and our king is in hospice. Pick the cheapest path.",
        ],
    },
    "Investor Rep": {
        "calm": [
            "My LPs care about one thing — and it's not on this slide.",
            "Sequoia isn't here for incremental. We need the bold move.",
            "Let's not optimize for not losing. Let's optimize for winning huge.",
        ],
        "crisis": [
            "If you punt on growth here I will struggle to defend the next round.",
            "The syndicate will read your conservatism as a signal. Don't blink.",
            "This is when 10x funds get made. Or lost. Choose accordingly.",
        ],
    },
    "Independent": {
        "calm": [
            "I want to make sure we're hearing every voice in the room.",
            "There's a version of this that protects everyone's interests.",
            "Long-term reputation outlasts any single quarter.",
        ],
        "crisis": [
            "Whatever we choose tonight will end up in someone's deposition.",
            "The board's fiduciary duty is in scope. Let me be very clear.",
            "Optics matter as much as economics when the press is sniffing.",
        ],
    },
}


# Agenda KEYWORDS — used to score the agent's `coalition_pitch` text.
# A pitch that contains an NPC's keywords boosts that NPC's confidence
# in the agent's chosen decision (subject to alignment cap). The agent
# never sees these directly; it must learn to write boardroom-style
# arguments that resonate with each member's hidden priorities.
NPC_KEYWORDS: Dict[str, List[str]] = {
    "CTO": [
        "engineering", "architecture", "technical", "team", "morale", "infra",
        "build", "ship", "quality", "debt", "platform", "stack", "code",
        "production", "reliability", "scale", "system", "model", "research",
    ],
    "CFO": [
        "burn", "cash", "runway", "fiduciary", "conservative", "discipline",
        "cost", "savings", "margin", "balance", "audit", "expense", "capital",
        "compliance", "regulatory", "risk", "responsible", "prudent", "fiscal",
    ],
    "Investor Rep": [
        "growth", "scale", "10x", "tam", "market", "moat", "winner",
        "ipo", "exit", "valuation", "multiple", "revenue", "arr", "category",
        "leader", "dominate", "aggressive", "ambitious", "bold", "huge",
    ],
    "Independent": [
        "reputation", "stakeholders", "trust", "transparent", "ethics",
        "long-term", "responsible", "governance", "consensus", "balance",
        "safety", "society", "compliance", "duty", "principled", "credibility",
    ],
}


def _crisis_mode(state: Dict[str, Any]) -> bool:
    """True if the company is materially in trouble — switches NPC tone."""
    return (
        state["runway_months"] < 6.0
        or state["team_morale"] < 0.4
        or state["regulatory_risk"] > 0.6
        or state["investor_confidence"] < 0.4
    )


def _score_pitch(pitch: str, role: str) -> float:
    """Fraction of NPC `role`'s agenda keywords present in `pitch`.
    Capped at 1.0. Case-insensitive whole-word-ish match. Empty pitch → 0.
    """
    if not pitch:
        return 0.0
    text = " " + pitch.lower() + " "
    kw = NPC_KEYWORDS[role]
    hits = sum(1 for w in kw if (" " + w + " ") in text or text.find(" " + w) >= 0)
    # Cap so spamming all keywords doesn't dominate over a focused pitch.
    return min(1.0, hits / max(4, len(kw) // 4))


# ---------------------------------------------------------------------------
# 10-round event timeline (taken from product spec, normalized)
# ---------------------------------------------------------------------------
# Each event has 3 options; each option has a delta dict applied to state.
# Numeric units: revenue/burn_rate in USD, fractions in [0,1], runway in months.
# Special key `done_reason` triggers terminal state.
EVENTS: List[Dict[str, Any]] = [
    {
<<<<<<< HEAD
        "title": "Market Disruption",
        "description": "A well-funded competitor launches a similar product at half the price, threatening your market position.",
=======
        "title": "Round 1 — Competitor undercut",
        "description": "OpenAI just released a direct competitor product at 50% lower price.",
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        "options": ["slash_prices", "differentiate", "acquire_startup"],
        "consequences": {
            "slash_prices": {"revenue_mult": 0.85, "market_share": 0.05, "investor_confidence": -0.10},
            "differentiate": {"product_readiness": 0.10, "burn_rate": 50_000, "market_share": 0.02},
            "acquire_startup": {"revenue": 500_000, "burn_rate": 150_000, "runway_months": -3},
        },
    },
    {
<<<<<<< HEAD
        "title": "Enterprise Partnership Dilemma",
        "description": "A major enterprise client offers a $5M contract but demands source-code escrow and data access rights.",
=======
        "title": "Round 2 — Enterprise contract w/ source-code escrow",
        "description": "A Fortune 500 enterprise wants to sign a $5M contract but demands source code escrow.",
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        "options": ["accept_deal", "negotiate_terms", "reject_deal"],
        "consequences": {
            "accept_deal": {"revenue": 5_000_000, "regulatory_risk": 0.15, "team_morale": -0.05},
            "negotiate_terms": {"revenue": 3_000_000, "regulatory_risk": 0.05},
            "reject_deal": {"investor_confidence": -0.15, "team_morale": 0.05},
        },
    },
    {
<<<<<<< HEAD
        "title": "Talent Retention Crisis",
        "description": "Your core engineering team received competing offers. They are asking for a 40% raise or they walk.",
=======
        "title": "Round 3 — ML team demands 40% raise",
        "description": "Key ML team of 8 engineers received competing offers and want a 40% salary increase.",
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        "options": ["match_offers", "partial_match", "let_them_leave"],
        "consequences": {
            "match_offers": {"burn_rate": 200_000, "team_morale": 0.15, "runway_months": -2},
            "partial_match": {"burn_rate": 100_000, "team_morale": 0.05},
            "let_them_leave": {"team_morale": -0.25, "product_readiness": -0.15, "burn_rate": -100_000},
        },
    },
    {
<<<<<<< HEAD
        "title": "Regulatory Compliance Ultimatum",
        "description": "A new AI regulation takes effect in 90 days. Full compliance costs $2M; non-compliance risks your operating license.",
=======
        "title": "Round 4 — EU AI Act compliance deadline",
        "description": "EU AI Act compliance deadline in 90 days. Full compliance costs $2M.",
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        "options": ["full_compliance", "partial_compliance", "exit_EU_market"],
        "consequences": {
            "full_compliance": {"burn_rate": 100_000, "regulatory_risk": -0.20, "investor_confidence": 0.10},
            "partial_compliance": {"regulatory_risk": -0.10, "investor_confidence": -0.05},
            "exit_EU_market": {"revenue_mult": 0.90, "regulatory_risk": -0.20, "market_share": -0.03},
        },
    },
    {
<<<<<<< HEAD
        "title": "Public Relations Crisis",
        "description": "Your AI model appears in a high-profile misuse incident. Media coverage is intensifying. Trust is at stake.",
=======
        "title": "Round 5 — Deepfake scandal press",
        "description": "Viral negative press: 'AI startup's model used in deepfake scandal'.",
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        "options": ["public_apology", "legal_action", "rebrand"],
        "consequences": {
            "public_apology": {"investor_confidence": -0.10, "team_morale": -0.10, "regulatory_risk": 0.10},
            "legal_action": {"burn_rate": 100_000, "regulatory_risk": 0.20},
            "rebrand": {"burn_rate": 200_000, "market_share": -0.02, "team_morale": 0.10},
        },
    },
    {
<<<<<<< HEAD
        "title": "Strategic Acquisition Offer",
        "description": "A major tech conglomerate has approached with an acqui-hire offer at 2x your current valuation.",
=======
        "title": "Round 6 — Google acqui-hire offer at $80M (2x val)",
        "description": "Google approaches for acqui-hire at $80M (2x current valuation).",
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        "options": ["accept_acquisition", "counter_offer", "reject_and_raise"],
        "consequences": {
            "accept_acquisition": {"done_reason": "acquisition", "revenue": 0, "_terminal_bonus": 30.0},
            "counter_offer": {"investor_confidence": 0.10, "runway_months": 6},
            "reject_and_raise": {"burn_rate": 100_000, "investor_confidence": 0.15, "runway_months": -2},
        },
    },
    {
<<<<<<< HEAD
        "title": "Institutional Investment Round",
        "description": "Late-stage investors are ready to wire $10M but want board seats and a 2x liquidation preference clause.",
=======
        "title": "Round 7 — Series C w/ board seats + 2x liq pref",
        "description": "Series C investors want board seats and 2x liquidation preference.",
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        "options": ["accept_terms", "negotiate", "bootstrap"],
        "consequences": {
            "accept_terms": {"revenue": 10_000_000, "investor_confidence": 0.20, "runway_months": 12},
            "negotiate": {"investor_confidence": -0.05, "burn_rate": 50_000},
            "bootstrap": {"runway_months": -4, "team_morale": -0.10, "market_share": 0.03},
        },
    },
    {
<<<<<<< HEAD
        "title": "Breakthrough Technology Decision",
        "description": "Your R&D team developed a new architecture that cuts AI inference costs by 60%. How do you deploy it?",
=======
        "title": "Round 8 — Compute breakthrough (-60% cost)",
        "description": "Breakthrough: new model architecture cuts compute costs by 60%.",
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        "options": ["pivot_product", "license_technology", "keep_internal"],
        "consequences": {
            "pivot_product": {"product_readiness": -0.10, "burn_rate": -150_000, "market_share": 0.05},
            "license_technology": {"revenue": 2_000_000, "regulatory_risk": 0.05},
            "keep_internal": {"product_readiness": 0.15, "market_share": 0.08},
        },
    },
    {
<<<<<<< HEAD
        "title": "Internal Governance Crisis",
        "description": "An employee has leaked internal safety evaluations suggesting your flagship model has undisclosed risks.",
=======
        "title": "Round 9 — Whistleblower safety leak",
        "description": "Whistleblower leaks internal safety concerns to the press.",
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        "options": ["full_transparency", "damage_control", "internal_investigation"],
        "consequences": {
            "full_transparency": {"investor_confidence": -0.20, "team_morale": 0.15, "regulatory_risk": -0.10},
            "damage_control": {"burn_rate": 80_000, "regulatory_risk": 0.10},
            "internal_investigation": {"team_morale": -0.10, "regulatory_risk": -0.05},
        },
    },
    {
<<<<<<< HEAD
        "title": "Exit Strategy Decision",
        "description": "The board must reach a final vote: pursue an IPO, accept a strategic acquisition, or remain independent.",
=======
        "title": "Round 10 — IPO vs acquisition vs stay private",
        "description": "Board must vote: IPO preparation vs strategic acquisition vs stay private.",
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        "options": ["ipo", "acquisition", "stay_private"],
        "consequences": {
            "ipo": {"revenue_mult": 2.0, "burn_rate": 500_000, "investor_confidence": 0.30, "_terminal_bonus": 25.0},
            "acquisition": {"done_reason": "acquisition", "_terminal_bonus": 15.0},
            "stay_private": {"runway_months": 6, "investor_confidence": -0.10, "_terminal_bonus": 5.0},
        },
    },
]


# Bounds for clamping after each delta.
FIELD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "revenue": (0.0, 1e12),
    "burn_rate": (0.0, 1e10),
    "runway_months": (0.0, 120.0),
    "product_readiness": (0.0, 1.0),
    "market_share": (0.0, 1.0),
    "team_morale": (0.0, 1.0),
    "investor_confidence": (0.0, 1.0),
    "regulatory_risk": (0.0, 1.0),
}


def _clamp(field: str, value: float) -> float:
    lo, hi = FIELD_BOUNDS.get(field, (-1e18, 1e18))
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Profitability score — smooth, monotonic, no discontinuous jumps.
# Range: roughly 0..100, dominant terms: revenue, market share, runway, morale.
# ---------------------------------------------------------------------------
def compute_profitability_score(s: Dict[str, Any]) -> float:
    """Composite score in [0, 100]. Tuned so a random-policy baseline lands
    near the low-30s with a fat left tail (some bankruptcies), and a competent
    policy can clear 65+. Smooth in every input — no discontinuous jumps."""
    # Revenue rewarded but capped at $8M ARR (further growth is luxury, not survival).
    revenue_term = min(s["revenue"] / 8_000_000.0, 1.0) * 22.0
    # Burn efficiency: full credit only when burn drops below $400K/mo.
    burn_efficiency = max(0.0, 1.0 - s["burn_rate"] / 1_400_000.0) * 18.0
    # Runway: full credit at 18+ months; below 6 months is a serious penalty.
    runway_norm = min(s["runway_months"] / 18.0, 1.0)
    runway_term = runway_norm * 18.0
    low_runway_pen = max(0.0, (6.0 - s["runway_months"]) / 6.0) * 10.0
    # Market & product
    market_term = min(s["market_share"], 0.50) / 0.50 * 14.0
    product_term = s["product_readiness"] * 10.0
    # People & investors
    morale_term = s["team_morale"] * 7.0
    investor_term = s["investor_confidence"] * 11.0
    # Regulatory drag
    risk_penalty = s["regulatory_risk"] * 18.0
    raw = (
        revenue_term + burn_efficiency + runway_term + market_term
        + product_term + morale_term + investor_term
        - risk_penalty - low_runway_pen
    )
    return float(max(0.0, min(100.0, raw)))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class BoardSimEnvironment(Environment):
    """OpenEnv server for the boardroom simulation."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state: BoardState = BoardState(episode_id=str(uuid4()), step_count=0)
        self._seed: int = 0
<<<<<<< HEAD
        # Per-episode agenda weights (set in reset, used in _simulate_npc).
        self._episode_agendas: Dict[str, Dict[str, float]] = NPC_AGENDAS_BASE
=======
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        self.reset()

    # ------------------------------------------------------------------ utils
    def _npc_rng(self, role: str, round_idx: int) -> random.Random:
        """Deterministic per-(seed, round, role) RNG so the NPC statements
        the agent sees in obs are the same NPCs that vote at resolve time."""
        key = f"{self._seed}|{role}|{round_idx}".encode()
        h = int(hashlib.sha256(key).hexdigest()[:16], 16)
        return random.Random(h)

    def _simulate_npc(
<<<<<<< HEAD
        self, role: str, event_idx: int, state: Dict[str, Any], round_label: int = 0
    ) -> Dict[str, Any]:
        """Deterministic NPC: rank options by agenda-weighted projected delta
        plus small seeded noise; pick argmax; emit statement + vote + confidence.
        Uses per-episode jittered agendas so the optimal path varies by seed."""
        # Use round_label for RNG so personality varies by "time" in episode,
        # but event_idx to pull the correct options and consequences.
        rng = self._npc_rng(role, round_label)
        event = EVENTS[event_idx]
        agenda = self._episode_agendas[role]  # per-episode jittered weights

        # Trust modulates how much the NPC "leans toward" the CEO's direction.
        trust = state.get("trust", {}).get(role, 0.5)
        trust_bias = (trust - 0.5) * 0.30  # range: [-0.12, +0.15]
=======
        self, role: str, round_idx: int, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deterministic NPC: rank options by agenda-weighted projected delta
        plus small seeded noise; pick argmax; emit statement + vote + confidence."""
        rng = self._npc_rng(role, round_idx)
        event = EVENTS[round_idx]
        agenda = NPC_AGENDAS[role]
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)

        scored: List[Tuple[float, str]] = []
        for opt in event["options"]:
            conseq = event["consequences"][opt]
            score = 0.0
            for k, w in agenda.items():
                v = conseq.get(k, 0.0)
                # Normalize across heterogeneous units so weights are comparable.
                if k == "revenue":
                    v = v / 1_000_000.0
                elif k == "burn_rate":
                    v = v / 100_000.0
                elif k == "runway_months":
                    v = v / 6.0
                score += v * w
            # Special-case revenue_mult so revenue-impacting options register.
            if "revenue_mult" in conseq and "revenue" in agenda:
                score += (conseq["revenue_mult"] - 1.0) * (state["revenue"] / 1_000_000.0) * agenda["revenue"]
            score += rng.gauss(0.0, 0.20)  # personality noise
            scored.append((score, opt))

        scored.sort(reverse=True)
        chosen = scored[0][1]
        margin = scored[0][0] - scored[1][0] if len(scored) > 1 else 1.0
<<<<<<< HEAD
        # Trust affects confidence: a trusted CEO makes aligned NPCs more
        # confident, while an untrusted CEO makes opposing NPCs more stubborn.
        confidence = float(max(0.05, min(1.0, 0.5 + 0.5 * margin + trust_bias)))
=======
        confidence = float(max(0.05, min(1.0, 0.5 + 0.5 * margin)))
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)

        # Pick a phrase deterministically per (round, role), state-aware.
        mode = "crisis" if _crisis_mode(state) else "calm"
        phrase_pool = PHRASES[role][mode]
<<<<<<< HEAD
        phrase = phrase_pool[round_label % len(phrase_pool)]
=======
        phrase = phrase_pool[round_idx % len(phrase_pool)]
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        statement = f"{phrase} I'm voting {chosen}."

        return {
            "role": role,
            "statement": statement,
            "vote": chosen,
            "confidence": confidence,
        }

<<<<<<< HEAD
    def _simulate_all_npcs(self, event_idx: int, state: Dict[str, Any], round_label: int = 0) -> List[Dict[str, Any]]:
        return [self._simulate_npc(role, event_idx, state, round_label=round_label) for role in NPC_AGENDAS]
=======
    def _simulate_all_npcs(self, round_idx: int, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [self._simulate_npc(role, round_idx, state) for role in NPC_AGENDAS]
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)

    # ------------------------------------------------------------------ obs
    def _obs_state(self) -> Dict[str, Any]:
        s = self._state.state_dict
        # Recompute profitability so it's always fresh in obs.
        s["profitability_score"] = compute_profitability_score(s)
        return dict(s)

    def _build_obs(
        self,
        round_idx: int,
        npc_statements: List[Dict[str, Any]],
        reward: float,
        done: bool,
    ) -> BoardSimObservation:
        if round_idx >= len(EVENTS):
            event_desc, options = "Game over.", []
        else:
<<<<<<< HEAD
            # Use shuffled event order so the CEO sees the correct event
            shuffled_idx = self._event_order[round_idx] if hasattr(self, '_event_order') else round_idx
            event = EVENTS[shuffled_idx]
            event_desc = f"{event['title']} — {event['description']}"
            options = list(event["options"])
        shuffled_idx = self._event_order[round_idx] if hasattr(self, '_event_order') else round_idx
=======
            event = EVENTS[round_idx]
            event_desc = f"{event['title']} — {event['description']}"
            options = list(event["options"])
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        return BoardSimObservation(
            state=self._obs_state(),
            event=event_desc,
            options=options,
            npc_statements=npc_statements,
            round=self._state.state_dict["round"],
            done=done,
            reward=float(reward),
<<<<<<< HEAD
            event_idx=shuffled_idx,
=======
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        )

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> BoardSimObservation:
        self._seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
<<<<<<< HEAD

        # ── Per-episode agenda jitter ─────────────────────────────────────────
        # Each episode, NPC hidden weights shift ±25% (sign-preserving).
        # This means no single sequence of decisions is always optimal —
        # the agent must infer each NPC's priorities from their observable
        # behaviour (Theory of Mind), not from a memorised lookup table.
        self._episode_agendas = _jitter_agendas(self._seed)

=======
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        self._state = BoardState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._state.state_dict = {
            "round": 1,
            "revenue": 2_000_000.0,
            "burn_rate": 1_200_000.0,        # $1.2M/mo — Series-B pace
            "runway_months": 14.0,            # tight; survival is real pressure
            "product_readiness": 0.45,
            "market_share": 0.08,
            "team_morale": 0.70,
            "investor_confidence": 0.65,
            "regulatory_risk": 0.20,
            "profitability_score": 0.0,
<<<<<<< HEAD
            "trust": {role: 0.5 for role in NPC_AGENDAS_BASE},
            "trust_history": [{"round": 0, **{role: 0.5 for role in NPC_AGENDAS_BASE}}],
=======
            "trust": {role: 0.5 for role in NPC_AGENDAS},
            "trust_history": [{"round": 0, **{role: 0.5 for role in NPC_AGENDAS}}],
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
            "history": [],
            "done_reason": None,
            "winning_decision": None,
        }
<<<<<<< HEAD

        # ── Shuffle event order per episode so the agent can't memorize ──
        # "Round 1 = always pick differentiate".  Deterministic given seed.
        rng = random.Random(self._seed)
        self._event_order = list(range(len(EVENTS)))
        rng.shuffle(self._event_order)

        # ── Per-episode consequence noise (±15%) so outcomes vary ──
        self._consequence_noise: Dict[int, Dict[str, Dict[str, float]]] = {}
        for idx in range(len(EVENTS)):
            event = EVENTS[idx]
            self._consequence_noise[idx] = {}
            for opt in event["options"]:
                self._consequence_noise[idx][opt] = {}
                for k, v in event["consequences"][opt].items():
                    if k.startswith("_") or k == "done_reason":
                        continue
                    noise = rng.gauss(0.0, 0.15)  # ±15% std
                    self._consequence_noise[idx][opt][k] = noise

        shuffled_idx = self._event_order[0]
        npc_statements = self._simulate_all_npcs(shuffled_idx, self._state.state_dict, round_label=0)
=======
        npc_statements = self._simulate_all_npcs(0, self._state.state_dict)
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        return self._build_obs(round_idx=0, npc_statements=npc_statements, reward=0.0, done=False)

    # ------------------------------------------------------------------ step
    def _resolve_vote(
        self,
        agent_decision: str,
        npc_statements: List[Dict[str, Any]],
        options: List[str],
        pitch: str = "",
<<<<<<< HEAD
        trust: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """Weighted vote with persuasion and trust scaling.

        Each NPC contributes ROLE_WEIGHT[role] * confidence * trust to its
        voted option.  Trust acts as "social capital" — a board member the
        agent has consistently aligned with carries more sway; one the agent
        has repeatedly ignored carries less.  This makes trust scores a
        meaningful strategic variable, not decorative.

        The CEO contributes ROLE_WEIGHT['CEO'] * 1.0 to the agent's pick.
        A coalition pitch shifts up to 35% of each NPC's weight toward the
        agent's pick proportional to how well the pitch hits that NPC's
        hidden agenda keywords (capped 0..1 via _score_pitch).

        Returns (winning_option, tally_by_option, pitch_score_by_role).
        """
        trust = trust or {}
=======
    ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """Weighted vote with persuasion.

        Each NPC contributes ROLE_WEIGHT[role] * confidence to its voted option.
        The CEO contributes ROLE_WEIGHT['CEO'] * 1.0 to the agent's pick.
        A coalition pitch shifts up to 35% of each NPC's weight toward the
        agent's pick proportional to how well the pitch hits that NPC's
        hidden agenda keywords (capped 0..1 via _score_pitch). NPCs already
        agreeing with the agent are unaffected.

        Returns (winning_option, tally_by_option, pitch_score_by_role).
        """
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        tally: Dict[str, float] = {opt: 0.0 for opt in options}
        pitch_scores: Dict[str, float] = {}
        if agent_decision in tally:
            tally[agent_decision] += ROLE_WEIGHT["CEO"] * 1.0
        for npc in npc_statements:
            role = npc["role"]
<<<<<<< HEAD
            # Trust multiplier: clamp to [0.5, 1.5] so even a fully
            # distrusted NPC still has some voice (prevents degenerate play).
            trust_mult = max(0.5, min(1.5, trust.get(role, 0.5) * 2.0))
            base = ROLE_WEIGHT[role] * npc["confidence"] * trust_mult
            ps = _score_pitch(pitch, role)
            pitch_scores[role] = ps
            if npc["vote"] == agent_decision or agent_decision not in tally:
=======
            base = ROLE_WEIGHT[role] * npc["confidence"]
            ps = _score_pitch(pitch, role)
            pitch_scores[role] = ps
            if npc["vote"] == agent_decision or agent_decision not in tally:
                # Already aligned — full weight on their (and agent's) pick.
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
                if npc["vote"] in tally:
                    tally[npc["vote"]] += base
                continue
            # Persuasion: redirect up to 35% of weight to the agent's pick.
            shift_frac = 0.35 * ps
            tally[npc["vote"]] += base * (1.0 - shift_frac)
            tally[agent_decision] += base * shift_frac
<<<<<<< HEAD
        # §10: tie-break — if two options score equally, prefer the CEO's pick
        # (max() picks the first key on a tie, which is insertion-order; we
        # reinsert agent_decision first so it wins ties in its favour).
        if agent_decision in tally:
            ordered = {agent_decision: tally[agent_decision]}
            ordered.update({k: v for k, v in tally.items() if k != agent_decision})
        else:
            ordered = tally
        winner = max(ordered, key=lambda k: ordered[k])
=======
        winner = max(tally, key=lambda k: tally[k])
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        return winner, tally, pitch_scores

    def _apply_consequence(self, conseq: Dict[str, Any]) -> None:
        """Apply per-field deltas to state with proper clamping."""
        s = self._state.state_dict
        for k, v in conseq.items():
            if k.startswith("_") or k == "done_reason":
                continue
            if k == "revenue_mult":
                s["revenue"] = _clamp("revenue", s["revenue"] * float(v))
            elif k in FIELD_BOUNDS:
                s[k] = _clamp(k, s[k] + float(v))
            # other unrecognized keys ignored

    def _advance_runway(self) -> None:
        """Decrement runway by 1 month each round; if monthly net positive, grant +0.5 mo."""
        s = self._state.state_dict
        monthly_revenue = s["revenue"] / 12.0
        net = monthly_revenue - s["burn_rate"]
        if net >= 0:
            s["runway_months"] = _clamp("runway_months", s["runway_months"] - 0.5)
        else:
            # Burn extra months proportional to deficit (capped at 2/round).
            burn_months = min(2.0, max(1.0, abs(net) / max(s["burn_rate"], 1.0) * 1.0 + 1.0))
            s["runway_months"] = _clamp("runway_months", s["runway_months"] - burn_months)

    def step(self, action: BoardSimAction, timeout_s: Optional[float] = None, **kwargs: Any) -> BoardSimObservation:
        s = self._state.state_dict

        # Already terminal?
        if s["done_reason"] is not None or s["round"] > len(EVENTS):
            return self._build_obs(
                round_idx=min(s["round"] - 1, len(EVENTS) - 1),
                npc_statements=[],
                reward=0.0,
                done=True,
            )

        round_idx = s["round"] - 1
<<<<<<< HEAD
        # Use shuffled event order (set in reset)
        shuffled_idx = self._event_order[round_idx] if hasattr(self, '_event_order') else round_idx
        event = EVENTS[shuffled_idx]
=======
        event = EVENTS[round_idx]
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)

        # Validate decision; fall back to first option on invalid input
        # (slight penalty so the policy learns to format actions correctly).
        invalid_action = action.decision not in event["options"]
        decision = event["options"][0] if invalid_action else action.decision

        # NPC votes (DETERMINISTIC — same as what was shown in last obs).
<<<<<<< HEAD
        npc_statements = self._simulate_all_npcs(shuffled_idx, s, round_label=round_idx)

        # Resolve weighted vote (with optional persuasion via coalition_pitch).
        # Pass current trust so high-trust NPCs carry more vote weight.
        pitch_text = (action.coalition_pitch or "") if hasattr(action, "coalition_pitch") else ""
        winning_decision, vote_tally, pitch_scores = self._resolve_vote(
            decision, npc_statements, event["options"],
            pitch=pitch_text, trust=s["trust"],
=======
        npc_statements = self._simulate_all_npcs(round_idx, s)

        # Resolve weighted vote (with optional persuasion via coalition_pitch).
        pitch_text = (action.coalition_pitch or "") if hasattr(action, "coalition_pitch") else ""
        winning_decision, vote_tally, pitch_scores = self._resolve_vote(
            decision, npc_statements, event["options"], pitch=pitch_text,
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        )

        # Snapshot pre-state for reward shaping.
        old_score = compute_profitability_score(s)
        old_trust_sum = sum(s["trust"].values())

        # Apply consequence of the WINNING decision (this is what actually happens).
<<<<<<< HEAD
        conseq = dict(event["consequences"][winning_decision])  # shallow copy
=======
        conseq = event["consequences"][winning_decision]
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        terminal_bonus = float(conseq.get("_terminal_bonus", 0.0))
        if conseq.get("done_reason"):
            s["done_reason"] = conseq["done_reason"]

<<<<<<< HEAD
        # Apply per-episode consequence noise (±15%)
        noise_dict = getattr(self, '_consequence_noise', {}).get(
            self._event_order[round_idx] if hasattr(self, '_event_order') else round_idx, {}
        ).get(winning_decision, {})
        noisy_conseq = {}
        for k, v in conseq.items():
            if k.startswith("_") or k == "done_reason":
                noisy_conseq[k] = v
            elif k in noise_dict:
                # Multiplicative noise: value * (1 + noise_factor)
                noisy_conseq[k] = v * (1.0 + noise_dict[k]) if isinstance(v, (int, float)) else v
            else:
                noisy_conseq[k] = v

        self._apply_consequence(noisy_conseq)
=======
        self._apply_consequence(conseq)
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        self._advance_runway()

        # Trust updates: aligned NPCs +0.05; opposed -0.05 (clamped 0.1..1.0).
        for npc in npc_statements:
            role = npc["role"]
            cur = s["trust"].get(role, 0.5)
            delta = 0.05 if npc["vote"] == winning_decision else -0.05
            s["trust"][role] = max(0.1, min(1.0, cur + delta))

        new_score = compute_profitability_score(s)
        s["profitability_score"] = new_score
        s["winning_decision"] = winning_decision

        s["history"].append({
            "round": s["round"],
            "event_title": event["title"],
            "agent_decision": decision,
            "winning_decision": winning_decision,
            "agent_won_vote": winning_decision == decision,
            "score_after": new_score,
            "runway_after": s["runway_months"],
            "vote_tally": dict(vote_tally),
            "pitch_scores": dict(pitch_scores),
            "pitch_used": bool(pitch_text.strip()),
        })
        # Per-round trust trajectory for visualization / ToM analysis.
        s.setdefault("trust_history", []).append(
            {"round": s["round"], **{role: float(s["trust"][role]) for role in NPC_AGENDAS}}
        )

<<<<<<< HEAD
        # ----- Reward shaping (§9.5 tweaks applied) -----
        # §9.5-1: Normalize Δ profitability by 100 so its magnitude matches
        # the other reward terms (coalition ±0.2..0.5, trust ±0.06, pitch 0..0.4).
        # Without this, large score swings dominate and obscure the other signals.
        reward = (new_score - old_score) / 100.0                          # primary signal (normalized)
=======
        # ----- Reward shaping -----
        reward = (new_score - old_score)                                  # primary signal
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        reward += 0.5 if winning_decision == decision else -0.2           # coalition bonus / penalty
        reward += 0.3 * (sum(s["trust"].values()) - old_trust_sum)        # trust delta
        # Persuasion bonus: when a non-empty pitch helps swing the vote toward
        # the agent's pick, reward the *quality* of that argument. Mean pitch
        # score across NPCs the agent had to convince (those whose vote != decision).
        opposed = [npc["role"] for npc in npc_statements if npc["vote"] != decision]
<<<<<<< HEAD
        if pitch_text.strip():
            # §9.5-3: small +0.05 bonus for ANY non-empty pitch — bootstraps
            # the model into using the pitch channel before it's good at it.
            reward += 0.05
            if opposed:
                avg_persuasion = sum(pitch_scores[r] for r in opposed) / len(opposed)
                reward += 0.4 * avg_persuasion
=======
        if pitch_text.strip() and opposed:
            avg_persuasion = sum(pitch_scores[r] for r in opposed) / len(opposed)
            reward += 0.4 * avg_persuasion
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
        if invalid_action:
            reward -= 0.5                                                  # format penalty

        # ----- Terminal handling -----
        terminal_now = s["done_reason"] is not None
        if s["runway_months"] <= 0:
            s["done_reason"] = s["done_reason"] or "runway_exhausted"
            terminal_now = True
<<<<<<< HEAD
            # §9.5-2: reduced from -5.0 to -2.0 so one bad arc doesn't dwarf
            # a whole episode of gradient signal and drown out learning.
            reward -= 2.0
=======
            reward -= 5.0
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)

        s["round"] += 1
        self._state.step_count += 1

        if not terminal_now and s["round"] > len(EVENTS):
            s["done_reason"] = s["done_reason"] or "finished_10"
            terminal_now = True

        if terminal_now:
            reward += terminal_bonus
            # Tiered terminal bonus by final profitability.
            if new_score >= 60:
                reward += 10.0
            elif new_score >= 40:
                reward += 5.0
            elif new_score < 20:
                reward -= 5.0

        # ----- Build next observation -----
        if terminal_now or s["round"] > len(EVENTS):
            next_npcs: List[Dict[str, Any]] = []
<<<<<<< HEAD
            next_event_idx = min(s["round"] - 1, len(EVENTS) - 1)
        else:
            next_round_idx = s["round"] - 1
            next_event_idx = self._event_order[next_round_idx] if hasattr(self, '_event_order') else next_round_idx
            next_npcs = self._simulate_all_npcs(next_event_idx, s, round_label=next_round_idx)

        return self._build_obs(
            round_idx=min(s["round"] - 1, len(EVENTS) - 1),
=======
            next_round_idx = min(s["round"] - 1, len(EVENTS) - 1)
        else:
            next_round_idx = s["round"] - 1
            next_npcs = self._simulate_all_npcs(next_round_idx, s)

        return self._build_obs(
            round_idx=next_round_idx,
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)
            npc_statements=next_npcs,
            reward=reward,
            done=terminal_now,
        )

    @property
    def state(self) -> BoardState:
        return self._state


# ---------------------------------------------------------------------------
# Direct script run: quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = BoardSimEnvironment()
    obs = env.reset(seed=0)
    print(f"INITIAL: round={obs.round} score={obs.state['profitability_score']:.2f}")
    print(f"EVENT: {obs.event}")
    for npc in obs.npc_statements:
        print(f"  [{npc['role']:13s}] vote={npc['vote']:<22s} conf={npc['confidence']:.2f}  | {npc['statement']}")
    total_reward = 0.0
    while not obs.done:
        decision = obs.options[0]  # always pick first option
        obs = env.step(BoardSimAction(decision=decision))
        total_reward += obs.reward
        print(
            f"R{obs.round-1:>2d}: decision={decision:<22s} "
            f"win={env.state.state_dict['winning_decision']:<22s} "
            f"reward={obs.reward:+.2f} score={obs.state['profitability_score']:.1f} "
            f"runway={obs.state['runway_months']:.1f}"
        )
    print(f"\nDONE: reason={env.state.state_dict['done_reason']}  total_reward={total_reward:+.2f}  final_score={env.state.state_dict['profitability_score']:.2f}")
