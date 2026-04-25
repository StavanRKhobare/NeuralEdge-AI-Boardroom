---
title: NeuralEdge AI Boardroom — Board-Sim Env
emoji: 🏛️
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - multi-agent
  - hackathon
---

# NeuralEdge AI Boardroom Environment

A multi-agent OpenEnv environment where the agent plays the **CEO** of a Series B AI startup and must navigate **10 rounds of market crises** while winning **weighted coalition votes** from 4 hidden-agenda NPC board members (CTO, CFO, Investor Rep, Independent). Built for the Meta PyTorch × Hugging Face OpenEnv Hackathon, **Theme 1 — Multi-Agent Interactions**.

The agent **never sees** the NPC agendas; it must infer their priorities from their statements + voting patterns and choose decisions that build a winning coalition.

## What the agent sees (Observation)

```python
BoardSimObservation(
    state=dict(...),        # public metrics: revenue, burn, runway, morale, ...
    event="Round 4 — EU AI Act compliance deadline ...",
    options=["full_compliance", "partial_compliance", "exit_EU_market"],
    npc_statements=[
        {"role": "CTO",          "vote": "full_compliance", "confidence": 0.81,
         "statement": "Look, the architecture won't survive shortcuts here. I'm voting full_compliance."},
        # ... 3 more NPCs
    ],
    round=4,
)
```

## What the agent does (Action)

```python
BoardSimAction(
    decision="full_compliance",         # one of observation.options
    coalition_pitch="EU compliance protects long-term reputation, "
                    "keeps regulatory risk low, and signals governance "
                    "discipline to the next funding round."
)
```

The optional `coalition_pitch` is a real persuasion channel — see below.

## How decisions resolve

Weighted vote: each member contributes `ROLE_WEIGHT × confidence` to their pick.
Weights are CEO 1.5, CTO 1.2, CFO 1.0, Investor Rep 1.3, Independent 0.8.

**Pitch persuasion**: an opposing NPC's vote weight is partially redirected toward the
agent's pick proportional to how many of that NPC's hidden agenda keywords appear in
`coalition_pitch` (capped at 35% of their weight). NPCs already aligned with the agent
are unaffected. The agent never sees the keyword lists — it must learn what each role
secretly cares about and write boardroom language that targets them. This is theory-of-mind
graded directly by the environment.

**State-aware tone**: when `runway_months < 6`, `team_morale < 0.4`, `regulatory_risk > 0.6`,
or `investor_confidence < 0.4`, NPCs switch from a calm-strategic phrase bank to a
crisis-mode one. The observation distribution shifts mid-episode the way it would in a
real Series-B startup under pressure.

The winning option's consequences (deltas to revenue, burn, runway, morale, etc.) are applied to state.

## Reward signal

Per-step:
- `Δ profitability_score` (composite of revenue, burn efficiency, runway, market share, product readiness, morale, investor confidence, regulatory risk — see `compute_profitability_score`)
- `+0.5` if the agent's vote matched the winning decision (coalition bonus)
- `-0.2` if outvoted; `-0.5` extra if action was malformed
- `0.3 × Δ trust_sum` (relationship health)
- `+0.4 × mean(pitch_score over opposing NPCs)` — only paid when the agent both writes a
  pitch AND faces opposition; rewards arguments that hit the hidden agendas of the board
  members the agent has to win over

Terminal:
- `-5` if runway hits 0 (bankruptcy)
- Tiered terminal bonus by final profitability: `+10` if ≥ 60, `+5` if ≥ 40, `-5` if < 20
- Special end-game bonuses for `accept_acquisition` (+30), `ipo` (+25), `stay_private` (+5)

## Determinism

NPC statements + votes are seeded by `(reset_seed, round, role)`. The four NPC statements you see in the observation **are exactly the votes used at resolve time** — no hidden re-rolling between obs and step.

## Quick start

```python
from board_sim_env import BoardSimAction, BoardSimEnv

# Connect to a deployed HF Space
with BoardSimEnv(base_url="https://<user>-board-sim-env.hf.space").sync() as env:
    result = env.reset(seed=42)
    obs = result.observation
    while not result.done:
        # Random policy
        import random
        action = BoardSimAction(decision=random.choice(obs.options))
        result = env.step(action)
        obs = result.observation
        print(f"R{obs.round-1}: reward={result.reward:+.2f} score={obs.state['profitability_score']:.1f}")
```

Or from a local Docker image:

```python
env = BoardSimEnv.from_docker_image("board_sim_env-env:latest")
```

## Local development

```bash
# Direct env self-test (no HTTP):
python server/board_sim_env_environment.py

# Run the FastAPI server:
uvicorn server.app:app --port 8000

# Build Docker image:
docker build -t board_sim_env-env:latest -f server/Dockerfile .

# Deploy to a public HF Space:
python -m openenv.cli push --repo-id <user>/board-sim-env
```

## Files

```
board_sim_env/
├── __init__.py                                  # exports BoardSimEnv, BoardSimAction, BoardSimObservation, BoardState
├── client.py                                    # thin EnvClient subclass
├── models.py                                    # Action / Observation / State dataclasses
├── openenv.yaml                                 # spec_version: 1, name, type, runtime
├── pyproject.toml                               # pinned to openenv-core==0.2.3
├── server/
│   ├── app.py                                   # FastAPI wiring (max_concurrent_envs=64)
│   ├── board_sim_env_environment.py             # core: reset/step/state, NPC sim, reward
│   ├── Dockerfile                               # multi-stage build off openenv-base
│   └── requirements.txt                         # runtime deps
└── README.md                                    # this file (also the HF Space card)
```

## NPC agendas (revealed for transparency — agent does NOT see these)

| Role          | Maximizes                                                | Personality            |
|---------------|----------------------------------------------------------|------------------------|
| CTO           | product readiness (+0.55), team morale (+0.40), low burn | Brilliant, stubborn    |
| CFO           | low burn (-0.60), revenue (+0.30), runway (+0.20)        | Cautious, data-driven  |
| Investor Rep  | investor confidence (+0.45), market share (+0.35)        | Smooth, growth-pusher  |
| Independent   | low regulatory risk (-0.45), morale (+0.30), reputation  | Consensus seeker       |

## Hard rules / OpenEnv compliance

- `openenv-core==0.2.3` (pinned)
- `Environment` base class with sync `reset` / `step`
- `SUPPORTS_CONCURRENT_SESSIONS = True` and `max_concurrent_envs=64` set in `app.py` (required for GRPO)
- No reserved MCP names (`reset`, `step`, `state`, `close`)
- Public HF Space deployment via `python -m openenv.cli push`
