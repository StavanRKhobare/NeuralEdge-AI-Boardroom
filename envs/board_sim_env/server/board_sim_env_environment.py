# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Boardroom — OpenEnv environment.

The agent plays the CEO of a generic mid-stage organization. Each of 10
rounds it sees a strategic event, statements + votes from 4 hidden-agenda
NPC board members, and must pick one of 3 decisions. Decisions are resolved
by a weighted vote and produce dense reward proportional to a composite
performance score plus coalition / trust shaping terms.

NPCs are deterministic-given-(seed, round, state) so GRPO has a stable
target to learn against. Events are intentionally generic (competition,
talent, regulation, PR, M&A, funding, governance, exit) so the simulation
applies to any organization, not a specific industry.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import BoardSimAction, BoardSimObservation, BoardState
except ImportError:  # direct script execution
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import BoardSimAction, BoardSimObservation, BoardState  # type: ignore


# ---------------------------------------------------------------------------
# Static config
# ---------------------------------------------------------------------------

# Per-role weighted vote influence. CEO weight bumped so a decisive CEO
# pick visibly moves outcomes (was 1.5; now 2.5).
ROLE_WEIGHT: Dict[str, float] = {
    "CEO": 2.5,
    "CTO": 1.2,
    "CFO": 1.0,
    "Investor Rep": 1.3,
    "Independent": 0.8,
}

# NPC base hidden agendas. Roles are intentionally generic (CTO, CFO,
# Investor Rep, Independent) — they map onto any organization with a
# leadership team and a board.
NPC_AGENDAS_BASE: Dict[str, Dict[str, float]] = {
    "CTO": {
        "product_readiness": 0.55,
        "team_morale": 0.40,
        "burn_rate": -0.10,
        "regulatory_risk": -0.05,
    },
    "CFO": {
        "burn_rate": -0.60,
        "revenue": 0.30,
        "runway_months": 0.20,
        "regulatory_risk": -0.25,
    },
    "Investor Rep": {
        "investor_confidence": 0.45,
        "market_share": 0.35,
        "revenue": 0.25,
        "burn_rate": -0.05,
    },
    "Independent": {
        "regulatory_risk": -0.45,
        "team_morale": 0.30,
        "investor_confidence": 0.25,
        "market_share": 0.10,
    },
}

NPC_AGENDAS: Dict[str, Dict[str, float]] = NPC_AGENDAS_BASE


# Plain-language manifestos used by the embedding-based pitch scorer.
# These describe the kind of argument each NPC finds persuasive without
# enumerating keyword lists. The scorer measures semantic similarity
# between the agent's pitch and each manifesto, so the agent has to write
# substantively aligned arguments rather than spray vocabulary.
NPC_MANIFESTOS: Dict[str, str] = {
    "CTO": (
        "Operational excellence and engineering quality come first. "
        "Protect the team. Reduce technical risk. Avoid shortcuts that "
        "create future failures. Invest in product reliability, infrastructure, "
        "and the people who build and maintain the system."
    ),
    "CFO": (
        "Capital discipline is the priority. Watch the burn, extend runway, "
        "and protect the balance sheet. Be cautious with regulatory exposure "
        "and prefer measured, defensible spending. The finance function "
        "exists to keep the company solvent and audit-ready."
    ),
    "Investor Rep": (
        "Growth, market share, and ambitious returns drive value. Move fast "
        "and play to win the category. Bold, scalable bets matter more than "
        "incremental optimization. Investors expect aggressive expansion and "
        "decisive moves on revenue and valuation."
    ),
    "Independent": (
        "Long-term reputation, governance, and stakeholder trust are decisive. "
        "Act with transparency and ethical responsibility. Avoid moves that "
        "compromise credibility or invite regulatory and societal backlash. "
        "Preserve consensus and the social license to operate."
    ),
}


def _jitter_agendas(seed: int) -> Dict[str, Dict[str, float]]:
    """Per-episode NPC agenda weights (sign-preserving ±25% jitter).
    Forces the agent to infer fresh priorities each episode rather than
    memorising a single optimal sequence."""
    rng = random.Random(seed ^ 0xDEADBEEF)
    jittered: Dict[str, Dict[str, float]] = {}
    for role, agenda in NPC_AGENDAS_BASE.items():
        jittered[role] = {}
        for field, w in agenda.items():
            factor = rng.uniform(0.75, 1.25)
            jittered[role][field] = round(w * factor, 4)
    return jittered


# Personality phrase banks (generic — no industry-specific jargon).
PHRASES: Dict[str, Dict[str, List[str]]] = {
    "CTO": {
        "calm": [
            "From an operational standpoint, the trade-offs here are clear.",
            "If we cut corners now we will pay for it in incidents later.",
            "I want to flag the implementation risk before we lock this in.",
            "The team can absorb one of these but not all three at once.",
        ],
        "crisis": [
            "Morale is fragile. Another bad call and we lose key people.",
            "I cannot keep the system stable while we keep adding scope.",
            "We are out of slack — pick the option my org can actually deliver.",
        ],
    },
    "CFO": {
        "calm": [
            "I would like the minutes to record my reservations on cost.",
            "From a fiduciary standpoint, only one of these is defensible.",
            "The numbers are tight and getting tighter; let's not pretend otherwise.",
        ],
        "crisis": [
            "Runway is the only metric that matters at this table right now.",
            "Cash is king and our king is in hospice. Pick the cheapest path.",
            "If we miss covenants this quarter, none of the other choices exist.",
        ],
    },
    "Investor Rep": {
        "calm": [
            "Our backers care about growth — that is not on the slide today.",
            "We were not funded to play it safe. Let's pick the bold lane.",
            "Optimize for winning the category, not for not losing.",
        ],
        "crisis": [
            "If you punt on growth here I will struggle to defend the next round.",
            "The syndicate will read your conservatism as a signal. Don't blink.",
            "This is when winners get made. Or unmade. Choose accordingly.",
        ],
    },
    "Independent": {
        "calm": [
            "I want to make sure every voice in the room is heard before we vote.",
            "There is a version of this that protects all stakeholders.",
            "Long-term reputation outlasts any single quarter's outcome.",
        ],
        "crisis": [
            "Whatever we choose tonight will end up in someone's deposition.",
            "The board's fiduciary duty is in scope — let me be very clear.",
            "Optics matter as much as economics when the press is paying attention.",
        ],
    },
}


def _crisis_mode(state: Dict[str, Any]) -> bool:
    return (
        state["runway_months"] < 6.0
        or state["team_morale"] < 0.4
        or state["regulatory_risk"] > 0.6
        or state["investor_confidence"] < 0.4
    )


# ---------------------------------------------------------------------------
# Pitch scoring — semantic similarity, NOT keyword counting.
# Primary path: sentence-transformers (genuine sentence embeddings).
# Fallback: TF-IDF cosine over (1,2)-grams with English stopwords removed,
# which is still token-based but with proper IDF weighting and stop-word
# handling — vastly better than literal keyword presence checks.
# ---------------------------------------------------------------------------
class _PitchScorer:
    def __init__(self) -> None:
        self._mode: Optional[str] = None  # "st" | "tfidf"
        self._st_model = None
        self._role_emb: Dict[str, Any] = {}
        self._tfidf = None
        self._tfidf_role_vecs: Dict[str, Any] = {}
        self._init_backend()

    def _init_backend(self) -> None:
        # Honour an env var to skip the heavyweight embedding model
        # (useful in CI / unit tests).
        force_tfidf = os.environ.get("BOARDSIM_PITCH_BACKEND", "").lower() == "tfidf"
        if not force_tfidf:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                import numpy as np  # noqa: F401
                self._st_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                for role, manifesto in NPC_MANIFESTOS.items():
                    emb = self._st_model.encode(
                        manifesto, normalize_embeddings=True, convert_to_numpy=True
                    )
                    self._role_emb[role] = emb
                self._mode = "st"
                return
            except Exception:
                pass
        # TF-IDF fallback (always available with scikit-learn).
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            self._tfidf = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), min_df=1
            )
            corpus = list(NPC_MANIFESTOS.values())
            self._tfidf.fit(corpus)
            for role, manifesto in NPC_MANIFESTOS.items():
                self._tfidf_role_vecs[role] = self._tfidf.transform([manifesto])
            self._mode = "tfidf"
        except Exception:
            self._mode = None  # last-resort: zero score

    def score(self, pitch: str, role: str) -> float:
        if not pitch or not pitch.strip():
            return 0.0
        if self._mode == "st" and self._st_model is not None:
            try:
                import numpy as np
                emb = self._st_model.encode(
                    pitch, normalize_embeddings=True, convert_to_numpy=True
                )
                sim = float(np.dot(emb, self._role_emb[role]))
                # Map cosine [-1, 1] → [0, 1] with a soft floor so neutral
                # text scores ~0.0 and topical text scores ~0.4-0.8.
                return max(0.0, min(1.0, (sim + 0.05) * 1.2))
            except Exception:
                pass
        if self._mode == "tfidf" and self._tfidf is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
                v = self._tfidf.transform([pitch])
                sim = float(cosine_similarity(v, self._tfidf_role_vecs[role])[0, 0])
                # TF-IDF cosine in (0,1); a touch of gain so a clearly aligned
                # pitch saturates above 0.7.
                return max(0.0, min(1.0, sim * 1.4))
            except Exception:
                pass
        return 0.0


_PITCH_SCORER: Optional[_PitchScorer] = None


def _get_pitch_scorer() -> _PitchScorer:
    global _PITCH_SCORER
    if _PITCH_SCORER is None:
        _PITCH_SCORER = _PitchScorer()
    return _PITCH_SCORER


def _score_pitch(pitch: str, role: str) -> float:
    """Semantic similarity between `pitch` and `role`'s manifesto. [0, 1]."""
    return _get_pitch_scorer().score(pitch, role)


# ---------------------------------------------------------------------------
# 10-round event timeline — generic across organizations.
# Numeric units: revenue/burn_rate in USD, fractions in [0,1], runway in months.
# Magnitudes raised so a CEO decision visibly moves the state on every round.
# ---------------------------------------------------------------------------
EVENTS: List[Dict[str, Any]] = [
    {
        "title": "New Competitor Entry",
        "description": "A larger competitor enters your core market with aggressive pricing and threatens your customer base.",
        "options": ["cut_prices", "double_down_on_quality", "form_strategic_partnership"],
        "consequences": {
            "cut_prices":              {"revenue_mult": 0.80, "market_share": 0.08, "investor_confidence": -0.18, "team_morale": -0.05},
            "double_down_on_quality":  {"product_readiness": 0.20, "burn_rate": 80_000, "market_share": 0.04, "team_morale": 0.08},
            "form_strategic_partnership": {"revenue": 800_000, "burn_rate": 120_000, "runway_months": -2, "investor_confidence": 0.08},
        },
    },
    {
        "title": "Major Client Contract Demand",
        "description": "A flagship enterprise client offers a $5M annual contract but demands exclusivity, audit rights, and tighter SLAs.",
        "options": ["accept_deal", "negotiate_terms", "decline_deal"],
        "consequences": {
            "accept_deal":     {"revenue": 5_000_000, "regulatory_risk": 0.20, "team_morale": -0.10, "investor_confidence": 0.10},
            "negotiate_terms": {"revenue": 3_000_000, "regulatory_risk": 0.08, "team_morale": 0.02},
            "decline_deal":    {"investor_confidence": -0.20, "team_morale": 0.10, "market_share": -0.02},
        },
    },
    {
        "title": "Talent Retention Crisis",
        "description": "Your highest performers received external offers and are asking for a 40% raise or they walk.",
        "options": ["match_offers", "partial_match", "accept_attrition"],
        "consequences": {
            "match_offers":     {"burn_rate": 250_000, "team_morale": 0.25, "runway_months": -2, "investor_confidence": -0.05},
            "partial_match":    {"burn_rate": 120_000, "team_morale": 0.10, "runway_months": -1},
            "accept_attrition": {"team_morale": -0.30, "product_readiness": -0.20, "burn_rate": -100_000},
        },
    },
    {
        "title": "Regulatory Compliance Ultimatum",
        "description": "A new industry regulation takes effect in 90 days. Full compliance costs $2M; non-compliance risks your operating license in a key market.",
        "options": ["full_compliance", "minimum_compliance", "exit_market"],
        "consequences": {
            "full_compliance":    {"burn_rate": 150_000, "regulatory_risk": -0.30, "investor_confidence": 0.15, "team_morale": 0.05},
            "minimum_compliance": {"burn_rate": 50_000, "regulatory_risk": -0.10, "investor_confidence": -0.08},
            "exit_market":        {"revenue_mult": 0.85, "regulatory_risk": -0.25, "market_share": -0.05, "investor_confidence": -0.05},
        },
    },
    {
        "title": "Public Relations Incident",
        "description": "A product issue is going viral and the press is preparing a critical story. Customer trust is at stake.",
        "options": ["public_apology", "legal_pushback", "rebrand_campaign"],
        "consequences": {
            "public_apology":   {"investor_confidence": -0.12, "team_morale": -0.05, "regulatory_risk": -0.10, "market_share": -0.02},
            "legal_pushback":   {"burn_rate": 150_000, "regulatory_risk": 0.25, "investor_confidence": -0.05},
            "rebrand_campaign": {"burn_rate": 250_000, "market_share": -0.04, "team_morale": 0.10, "investor_confidence": 0.05},
        },
    },
    {
        "title": "Strategic Acquisition Offer",
        "description": "A larger firm has approached with an acqui-hire offer at 2x your current valuation.",
        "options": ["accept_acquisition", "counter_offer", "reject_and_grow"],
        "consequences": {
            "accept_acquisition": {"done_reason": "acquisition", "revenue": 0, "_terminal_bonus": 30.0},
            "counter_offer":      {"investor_confidence": 0.15, "runway_months": 6, "burn_rate": 30_000},
            "reject_and_grow":    {"burn_rate": 120_000, "investor_confidence": 0.20, "runway_months": -2, "team_morale": 0.05},
        },
    },
    {
        "title": "Institutional Funding Round",
        "description": "Late-stage investors are ready to wire $10M but want board seats and a 2x liquidation preference.",
        "options": ["accept_terms", "negotiate_terms", "bootstrap"],
        "consequences": {
            "accept_terms":   {"revenue": 10_000_000, "investor_confidence": 0.25, "runway_months": 12, "team_morale": -0.05},
            "negotiate_terms": {"investor_confidence": -0.05, "burn_rate": 50_000, "runway_months": 4},
            "bootstrap":      {"runway_months": -4, "team_morale": -0.15, "market_share": 0.05, "investor_confidence": -0.10},
        },
    },
    {
        "title": "Operational Innovation Decision",
        "description": "Your operations team developed a process that cuts unit costs by 60%. How do you deploy it?",
        "options": ["reinvest_in_growth", "license_externally", "keep_competitive_advantage"],
        "consequences": {
            "reinvest_in_growth":         {"product_readiness": -0.05, "burn_rate": -200_000, "market_share": 0.10, "team_morale": 0.05},
            "license_externally":         {"revenue": 2_500_000, "regulatory_risk": 0.05, "investor_confidence": 0.05},
            "keep_competitive_advantage": {"product_readiness": 0.20, "market_share": 0.12, "burn_rate": 50_000},
        },
    },
    {
        "title": "Internal Whistleblower Report",
        "description": "An employee leaked an internal audit suggesting your flagship product has undisclosed quality issues.",
        "options": ["full_disclosure", "contained_response", "internal_review"],
        "consequences": {
            "full_disclosure":     {"investor_confidence": -0.25, "team_morale": 0.20, "regulatory_risk": -0.20, "market_share": -0.03},
            "contained_response":  {"burn_rate": 120_000, "regulatory_risk": 0.20, "team_morale": -0.10},
            "internal_review":     {"team_morale": -0.05, "regulatory_risk": -0.10, "burn_rate": 50_000},
        },
    },
    {
        "title": "Strategic Exit Decision",
        "description": "The board must reach a final vote on the long-term path: pursue an IPO, accept a strategic acquisition, or stay independent.",
        "options": ["ipo", "sell_to_strategic", "stay_independent"],
        "consequences": {
            "ipo":               {"revenue_mult": 2.0, "burn_rate": 500_000, "investor_confidence": 0.30, "_terminal_bonus": 25.0},
            "sell_to_strategic": {"done_reason": "acquisition", "_terminal_bonus": 18.0},
            "stay_independent":  {"runway_months": 6, "investor_confidence": -0.10, "team_morale": 0.10, "_terminal_bonus": 5.0},
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
# Performance score — smooth, monotonic, no discontinuous jumps.
# Range: roughly 0..100. Tuned so a uniformly-random policy lands in the
# low-30s and a competent policy clears 60+.
# ---------------------------------------------------------------------------
def compute_profitability_score(s: Dict[str, Any]) -> float:
    revenue_term = min(s["revenue"] / 8_000_000.0, 1.0) * 22.0
    burn_efficiency = max(0.0, 1.0 - s["burn_rate"] / 1_400_000.0) * 18.0
    runway_norm = min(s["runway_months"] / 18.0, 1.0)
    runway_term = runway_norm * 18.0
    low_runway_pen = max(0.0, (6.0 - s["runway_months"]) / 6.0) * 10.0
    market_term = min(s["market_share"], 0.50) / 0.50 * 14.0
    product_term = s["product_readiness"] * 10.0
    morale_term = s["team_morale"] * 7.0
    investor_term = s["investor_confidence"] * 11.0
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
        self._episode_agendas: Dict[str, Dict[str, float]] = NPC_AGENDAS_BASE
        self.reset()

    def _npc_rng(self, role: str, round_idx: int) -> random.Random:
        key = f"{self._seed}|{role}|{round_idx}".encode()
        h = int(hashlib.sha256(key).hexdigest()[:16], 16)
        return random.Random(h)

    def _simulate_npc(
        self, role: str, event_idx: int, state: Dict[str, Any], round_label: int = 0
    ) -> Dict[str, Any]:
        rng = self._npc_rng(role, round_label)
        event = EVENTS[event_idx]
        agenda = self._episode_agendas[role]

        trust = state.get("trust", {}).get(role, 0.5)
        trust_bias = (trust - 0.5) * 0.30

        scored: List[Tuple[float, str]] = []
        for opt in event["options"]:
            conseq = event["consequences"][opt]
            score = 0.0
            for k, w in agenda.items():
                v = conseq.get(k, 0.0)
                if k == "revenue":
                    v = v / 1_000_000.0
                elif k == "burn_rate":
                    v = v / 100_000.0
                elif k == "runway_months":
                    v = v / 6.0
                score += v * w
            if "revenue_mult" in conseq and "revenue" in agenda:
                score += (conseq["revenue_mult"] - 1.0) * (state["revenue"] / 1_000_000.0) * agenda["revenue"]
            score += rng.gauss(0.0, 0.20)
            scored.append((score, opt))

        scored.sort(reverse=True)
        chosen = scored[0][1]
        margin = scored[0][0] - scored[1][0] if len(scored) > 1 else 1.0
        confidence = float(max(0.05, min(1.0, 0.5 + 0.5 * margin + trust_bias)))

        mode = "crisis" if _crisis_mode(state) else "calm"
        phrase_pool = PHRASES[role][mode]
        phrase = phrase_pool[round_label % len(phrase_pool)]
        statement = f"{phrase} I'm voting {chosen}."

        return {
            "role": role,
            "statement": statement,
            "vote": chosen,
            "confidence": confidence,
        }

    def _simulate_all_npcs(self, event_idx: int, state: Dict[str, Any], round_label: int = 0) -> List[Dict[str, Any]]:
        return [self._simulate_npc(role, event_idx, state, round_label=round_label) for role in NPC_AGENDAS]

    def _obs_state(self) -> Dict[str, Any]:
        s = self._state.state_dict
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
            shuffled_idx = self._event_order[round_idx] if hasattr(self, '_event_order') else round_idx
            event = EVENTS[shuffled_idx]
            event_desc = f"{event['title']} — {event['description']}"
            options = list(event["options"])
        shuffled_idx = self._event_order[round_idx] if hasattr(self, '_event_order') else round_idx
        return BoardSimObservation(
            state=self._obs_state(),
            event=event_desc,
            options=options,
            npc_statements=npc_statements,
            round=self._state.state_dict["round"],
            done=done,
            reward=float(reward),
            event_idx=shuffled_idx,
        )

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> BoardSimObservation:
        self._seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        self._episode_agendas = _jitter_agendas(self._seed)

        self._state = BoardState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._state.state_dict = {
            "round": 1,
            "revenue": 2_000_000.0,
            "burn_rate": 1_200_000.0,
            "runway_months": 14.0,
            "product_readiness": 0.45,
            "market_share": 0.08,
            "team_morale": 0.70,
            "investor_confidence": 0.65,
            "regulatory_risk": 0.20,
            "profitability_score": 0.0,
            "trust": {role: 0.5 for role in NPC_AGENDAS_BASE},
            "trust_history": [{"round": 0, **{role: 0.5 for role in NPC_AGENDAS_BASE}}],
            "history": [],
            "done_reason": None,
            "winning_decision": None,
        }

        rng = random.Random(self._seed)
        self._event_order = list(range(len(EVENTS)))
        rng.shuffle(self._event_order)

        # Per-episode consequence noise (±15%) so outcomes vary slightly.
        self._consequence_noise: Dict[int, Dict[str, Dict[str, float]]] = {}
        for idx in range(len(EVENTS)):
            event = EVENTS[idx]
            self._consequence_noise[idx] = {}
            for opt in event["options"]:
                self._consequence_noise[idx][opt] = {}
                for k, v in event["consequences"][opt].items():
                    if k.startswith("_") or k == "done_reason":
                        continue
                    noise = rng.gauss(0.0, 0.15)
                    self._consequence_noise[idx][opt][k] = noise

        shuffled_idx = self._event_order[0]
        npc_statements = self._simulate_all_npcs(shuffled_idx, self._state.state_dict, round_label=0)
        return self._build_obs(round_idx=0, npc_statements=npc_statements, reward=0.0, done=False)

    def _resolve_vote(
        self,
        agent_decision: str,
        npc_statements: List[Dict[str, Any]],
        options: List[str],
        pitch: str = "",
        trust: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """Weighted vote with persuasion and trust scaling.

        CEO contributes ROLE_WEIGHT['CEO'] (= 2.5) to the agent's pick — a
        deliberate buff so the CEO's call usually wins, making the impact
        of CEO decision-making visible round-to-round. NPCs still matter
        through coalition shaping (trust) and persuasion shifts.
        """
        trust = trust or {}
        tally: Dict[str, float] = {opt: 0.0 for opt in options}
        pitch_scores: Dict[str, float] = {}
        if agent_decision in tally:
            tally[agent_decision] += ROLE_WEIGHT["CEO"] * 1.0
        for npc in npc_statements:
            role = npc["role"]
            trust_mult = max(0.5, min(1.5, trust.get(role, 0.5) * 2.0))
            base = ROLE_WEIGHT[role] * npc["confidence"] * trust_mult
            ps = _score_pitch(pitch, role)
            pitch_scores[role] = ps
            if npc["vote"] == agent_decision or agent_decision not in tally:
                if npc["vote"] in tally:
                    tally[npc["vote"]] += base
                continue
            # Persuasion: redirect up to 55% of an NPC's weight toward the
            # agent's pick proportional to semantic alignment of the pitch.
            shift_frac = 0.55 * ps
            tally[npc["vote"]] += base * (1.0 - shift_frac)
            tally[agent_decision] += base * shift_frac
        if agent_decision in tally:
            ordered = {agent_decision: tally[agent_decision]}
            ordered.update({k: v for k, v in tally.items() if k != agent_decision})
        else:
            ordered = tally
        winner = max(ordered, key=lambda k: ordered[k])
        return winner, tally, pitch_scores

    def _apply_consequence(self, conseq: Dict[str, Any]) -> None:
        s = self._state.state_dict
        for k, v in conseq.items():
            if k.startswith("_") or k == "done_reason":
                continue
            if k == "revenue_mult":
                s["revenue"] = _clamp("revenue", s["revenue"] * float(v))
            elif k in FIELD_BOUNDS:
                s[k] = _clamp(k, s[k] + float(v))

    def _advance_runway(self) -> None:
        s = self._state.state_dict
        monthly_revenue = s["revenue"] / 12.0
        net = monthly_revenue - s["burn_rate"]
        if net >= 0:
            s["runway_months"] = _clamp("runway_months", s["runway_months"] - 0.5)
        else:
            burn_months = min(2.0, max(1.0, abs(net) / max(s["burn_rate"], 1.0) * 1.0 + 1.0))
            s["runway_months"] = _clamp("runway_months", s["runway_months"] - burn_months)

    def step(self, action: BoardSimAction, timeout_s: Optional[float] = None, **kwargs: Any) -> BoardSimObservation:
        s = self._state.state_dict

        if s["done_reason"] is not None or s["round"] > len(EVENTS):
            return self._build_obs(
                round_idx=min(s["round"] - 1, len(EVENTS) - 1),
                npc_statements=[],
                reward=0.0,
                done=True,
            )

        round_idx = s["round"] - 1
        shuffled_idx = self._event_order[round_idx] if hasattr(self, '_event_order') else round_idx
        event = EVENTS[shuffled_idx]

        invalid_action = action.decision not in event["options"]
        decision = event["options"][0] if invalid_action else action.decision

        npc_statements = self._simulate_all_npcs(shuffled_idx, s, round_label=round_idx)

        pitch_text = (action.coalition_pitch or "") if hasattr(action, "coalition_pitch") else ""
        winning_decision, vote_tally, pitch_scores = self._resolve_vote(
            decision, npc_statements, event["options"],
            pitch=pitch_text, trust=s["trust"],
        )

        old_score = compute_profitability_score(s)
        old_trust_sum = sum(s["trust"].values())

        conseq = dict(event["consequences"][winning_decision])
        terminal_bonus = float(conseq.get("_terminal_bonus", 0.0))
        if conseq.get("done_reason"):
            s["done_reason"] = conseq["done_reason"]

        noise_dict = getattr(self, '_consequence_noise', {}).get(
            self._event_order[round_idx] if hasattr(self, '_event_order') else round_idx, {}
        ).get(winning_decision, {})
        noisy_conseq = {}
        for k, v in conseq.items():
            if k.startswith("_") or k == "done_reason":
                noisy_conseq[k] = v
            elif k in noise_dict:
                noisy_conseq[k] = v * (1.0 + noise_dict[k]) if isinstance(v, (int, float)) else v
            else:
                noisy_conseq[k] = v

        self._apply_consequence(noisy_conseq)
        self._advance_runway()

        # Trust deltas widened ±0.05 → ±0.08 so CEO-driven trust shifts are
        # visible across a 10-round episode.
        for npc in npc_statements:
            role = npc["role"]
            cur = s["trust"].get(role, 0.5)
            delta = 0.08 if npc["vote"] == winning_decision else -0.08
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
        s.setdefault("trust_history", []).append(
            {"round": s["round"], **{role: float(s["trust"][role]) for role in NPC_AGENDAS}}
        )

        # Reward shaping (magnitudes raised so CEO impact is visible).
        reward = (new_score - old_score) / 100.0
        reward += 1.0 if winning_decision == decision else -0.4
        reward += 0.5 * (sum(s["trust"].values()) - old_trust_sum)
        opposed = [npc["role"] for npc in npc_statements if npc["vote"] != decision]
        if pitch_text.strip():
            reward += 0.05  # bootstrap bonus for using the pitch channel
            if opposed:
                avg_persuasion = sum(pitch_scores[r] for r in opposed) / len(opposed)
                reward += 0.6 * avg_persuasion
        if invalid_action:
            reward -= 0.5

        terminal_now = s["done_reason"] is not None
        if s["runway_months"] <= 0:
            s["done_reason"] = s["done_reason"] or "runway_exhausted"
            terminal_now = True
            reward -= 2.0

        s["round"] += 1
        self._state.step_count += 1

        if not terminal_now and s["round"] > len(EVENTS):
            s["done_reason"] = s["done_reason"] or "finished_10"
            terminal_now = True

        if terminal_now:
            reward += terminal_bonus
            if new_score >= 60:
                reward += 10.0
            elif new_score >= 40:
                reward += 5.0
            elif new_score < 20:
                reward -= 5.0

        if terminal_now or s["round"] > len(EVENTS):
            next_npcs: List[Dict[str, Any]] = []
            next_event_idx = min(s["round"] - 1, len(EVENTS) - 1)
        else:
            next_round_idx = s["round"] - 1
            next_event_idx = self._event_order[next_round_idx] if hasattr(self, '_event_order') else next_round_idx
            next_npcs = self._simulate_all_npcs(next_event_idx, s, round_label=next_round_idx)

        return self._build_obs(
            round_idx=min(s["round"] - 1, len(EVENTS) - 1),
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
        print(f"  [{npc['role']:13s}] vote={npc['vote']:<28s} conf={npc['confidence']:.2f}  | {npc['statement']}")
    total_reward = 0.0
    while not obs.done:
        decision = obs.options[0]
        obs = env.step(BoardSimAction(decision=decision))
        total_reward += obs.reward
        print(
            f"R{obs.round-1:>2d}: decision={decision:<28s} "
            f"win={env.state.state_dict['winning_decision']:<28s} "
            f"reward={obs.reward:+.2f} score={obs.state['profitability_score']:.1f} "
            f"runway={obs.state['runway_months']:.1f}"
        )
    print(f"\nDONE: reason={env.state.state_dict['done_reason']}  total_reward={total_reward:+.2f}  final_score={env.state.state_dict['profitability_score']:.2f}")
