---
title: NeuralEdge AI Boardroom — OpenEnv environment
emoji: 🏛️
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - multi-agent
  - reinforcement-learning
  - theory-of-mind
  - hackathon
---

# NeuralEdge AI Boardroom — Environment package

OpenEnv environment in which a CEO RL agent (Sarah Chen, Series-B AI startup) plays 10 rounds of strategic decisions against 4 NPC board members (CTO, CFO, Investor Rep, Independent Director) with **hidden agendas**. Built for the Meta × PyTorch × HuggingFace OpenEnv Hackathon, Theme 1 — Multi-Agent Interactions.

The agent **never sees** NPC agendas — it must infer them from per-round NPC `statement` text and the voting record in `history`, then articulate decisions in a `(decision, coalition_pitch)` action.

## Observation

```python
BoardSimObservation(
    state=dict(...),                # public metrics: revenue, burn, runway,
                                    #   morale, investor_confidence, regulatory_risk,
                                    #   profitability_score, trust[role], history,
                                    #   trust_history
    event="Regulatory compliance ultimatum — EU AI Act 90-day deadline ...",
    options=["full cooperation", "limited disclosure", "seek legal delay"],
    npc_statements=[
        {"role": "CTO", "vote": "limited disclosure", "confidence": 0.61,
         "statement": "Engineering can implement a partial compliance layer in 6 weeks ..."},
        # ... 3 more NPCs (CFO, Investor Rep, Independent)
    ],
    round=4,
)
```

## Action

```python
BoardSimAction(
    decision="full cooperation",          # one of observation.options
    coalition_pitch="A clean AI Act record protects the Series-C close ..."
)
```

`coalition_pitch` is a **graded action channel** — see persuasion below.

## Vote resolution

Weighted vote per round:

| Role | Weight |
|---|---|
| CEO | 2.5 |
| CTO | 1.2 |
| CFO | 1.0 |
| Investor Rep | 1.3 |
| Independent | 0.8 |

Each NPC scores the 3 options against its **hidden agenda** (per-round Gaussian personality noise), votes for `argmax`, and contributes `ROLE_WEIGHT × confidence × clamp(trust × 2, 0.5, 1.5)` to its pick's tally.

**Pitch persuasion.** The pitch is scored against each opposing NPC's hidden manifesto via sentence-transformer cosine (`all-MiniLM-L6-v2`); TF-IDF (1,2)-gram fallback when the embedding model is unavailable. Up to **55% of an opposing NPC's weight** is redirected to the CEO's pick proportional to the pitch score. NPCs already aligned with the CEO are unaffected. The agent never sees the manifestos — it must learn what each role secretly cares about and write language that targets them. **This is theory-of-mind graded directly by the reward function.**

The winning option's consequences (deltas to revenue, burn, runway, morale, regulatory_risk, …) are applied with ±15% Gaussian noise, sampled once at `reset()` and fixed for the episode.

## Reward function

Source of truth: `server/board_sim_env_environment.py:646` (full docstring on the `step()` method).

```
# Per-step (dense, bounded ≈ [-0.7, +0.65])
reward  = (new_profit_score - old_profit_score) / 100.0
reward += +1.0 if winning_decision == agent_decision else -0.4
reward += 0.5 * (Σ trust_after - Σ trust_before)
if pitch is non-empty:
    reward += 0.05                                          # bootstrap
    if any NPC opposed CEO's pick:
        reward += 0.6 * mean(pitch_score over opposing NPCs)  # ToM persuasion
if action.decision not in current_round.options:
    reward -= 0.5                                           # format penalty

# Terminal (episodic spikes by design)
if runway_months <= 0:
    reward -= 2.0                                           # bankruptcy
if terminal:
    reward += event._terminal_bonus       # acquisition +30, IPO +25, stay-private +5
    reward += {+10 if final_score >= 60, +5 if >= 40, -5 if < 20}
```

## Determinism + variability

- **Seeded `reset()`**: NPC statements, votes, consequence noise, event order, agenda jitter — all fixed at episode reset.
- **Event order shuffled per episode** (same 10 events, different sequence per seed).
- **Consequence magnitudes ±15% Gaussian noise** sampled at `reset()`.
- **NPC agenda weights ±25% sign-preserving jitter** per episode.

## NPC manifestos (revealed for transparency — agent does NOT see these at runtime)

| Role | Hidden manifesto (paraphrased) |
|---|---|
| CTO | Operational excellence, engineering quality, team morale, technical risk reduction |
| CFO | Capital discipline, runway, balance-sheet protection, regulatory caution |
| Investor Rep | Growth, market share, ambitious returns, decisive bold bets |
| Independent | Long-term reputation, governance, stakeholder trust, ethical responsibility |

## Quick start

```python
from board_sim_env import BoardSimAction, BoardSimEnv

with BoardSimEnv(base_url="https://stavankhobare-sst-metaxpytorch-hackathon.hf.space").sync() as env:
    result = env.reset(seed=42)
    obs = result.observation
    while not result.done:
        result = env.step(BoardSimAction(
            decision=obs.options[0],
            coalition_pitch="Margin protection and runway discipline argue for the conservative path.",
        ))
        obs = result.observation
        print(f"R{obs.round-1}: reward={result.reward:+.2f}  "
              f"score={obs.state['profitability_score']:.1f}")
```

Or against a local Docker image:

```python
env = BoardSimEnv.from_docker_image("board_sim_env-env:latest")
```

## Local development

```bash
# Direct env self-test (no HTTP, in-process)
python server/board_sim_env_environment.py

# Run the FastAPI server
uvicorn server.app:app --port 8000   # Swagger: http://localhost:8000/docs

# Build Docker image
docker build -t board_sim_env-env:latest -f server/Dockerfile .

# Deploy to a public HF Space
python -m openenv.cli push --repo-id <user>/board-sim-env
```

## Files

```
board_sim_env/
├── __init__.py                       # exports BoardSimEnv, BoardSimAction, BoardSimObservation, BoardState
├── client.py                         # EnvClient subclass
├── models.py                         # Action / Observation / State types
├── openenv.yaml                      # spec_version: 1, name, runtime: docker
├── pyproject.toml                    # pinned openenv-core==0.2.3
└── server/
    ├── app.py                        # FastAPI wiring (max_concurrent_envs=64)
    ├── board_sim_env_environment.py  # reset/step/state, NPC simulation, semantic pitch scorer, reward
    ├── Dockerfile                    # multi-stage build off openenv-base
    └── requirements.txt              # runtime deps incl. scikit-learn + sentence-transformers
```

## OpenEnv compliance

- `openenv-core==0.2.3` (pinned in `pyproject.toml`)
- Synchronous `reset()` / `step()` `Environment` API
- `SUPPORTS_CONCURRENT_SESSIONS = True`, `max_concurrent_envs=64` in `app.py` (required for GRPO group rollouts)
- No reserved MCP names
- Public HF Space deployment via `python -m openenv.cli push`
