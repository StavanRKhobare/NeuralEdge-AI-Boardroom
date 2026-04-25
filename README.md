---
title: BoardSim — Multi-Agent Boardroom
emoji: 🏛️
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - multi-agent
  - reinforcement-learning
  - hackathon
---

# BoardSim — Multi-Agent Boardroom (OpenEnv submission)

**Theme**: Theme 1 — Multi-Agent Interactions  
**Framework**: OpenEnv `v0.2.3` · Qwen3-4B · Unsloth LoRA · TRL `GRPOTrainer` (group-relative policy optimisation)  
**Event**: Meta PyTorch × Hugging Face OpenEnv Hackathon — India finale, Scaler Bangalore, **Apr 25–26 2026**

> A CEO agent learns to build winning board coalitions across 10 rounds of organisational events — against 4 NPCs with hidden agendas — by writing semantically aligned pitches that target what each board member privately cares about.

---

## What's new in this revision

- **Events are organisation-agnostic**: competition, talent, regulation, PR, M&A, funding, governance, exit. The simulation maps to *any* mid-stage company, not one industry.
- **Semantic pitch scoring**: pitches are scored by sentence-transformer cosine similarity (`all-MiniLM-L6-v2`) against per-role manifestos, with a TF-IDF (1,2)-gram fallback. The agent can no longer game the scorer by spraying keywords.
- **Baseline is the same Qwen3-4B model with LoRA disabled** — not a random policy. A coin flip is not a meaningful opponent for a 4 B language model picking among 3 well-formed strings; the apples-to-apples reference is the *same model* without the fine-tuning delta. Recovered cheaply via peft's `model.disable_adapter()` (no second model load).
- **CEO vote weight 2.5×** and **persuasion shift cap 55%** so a CEO decision visibly moves outcomes round-to-round.
- **Per-event boardroom win-rate plot** added — the most direct picture of *where* fine-tuning helps.

---

## Submission links

| # | Required | Link |
|---|---|---|
| 1 | **HF Space** (live env) | https://huggingface.co/spaces/StavanKhobare/SST-MetaxPyTorch-Hackathon |
| 2 | **Colab notebook** (training) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StavanRKhobare/SST-MetaxPyTorch-Hackathon/blob/master/notebooks/train_grpo_v2.ipynb) |
| 3 | **Code repository** | https://github.com/StavanRKhobare/SST-MetaxPyTorch-Hackathon |
| 4 | **Writeup** | TBD — record after training run |
| 5 | **W&B run** | TBD — populate after Colab run |

---

## What the agent does

```
You are the CEO of a mid-stage organisation. The board has 4 members with HIDDEN AGENDAS:
  - CTO: operational excellence, engineering quality, team morale, product readiness.
  - CFO: cash discipline, runway, regulatory safety.
  - Investor Rep: growth, market share, ambitious returns.
  - Independent: long-term reputation, governance, stakeholder consensus.

Each round you see a strategic event, every NPC's pre-vote statement, and 3 options.
Your decision is resolved by WEIGHTED VOTE (CEO weight 2.5x). A short COALITION PITCH
that is SEMANTICALLY ALIGNED with opposing members' priorities can swing them toward
your pick — write substantive arguments, not buzzword spray.

Respond on EXACTLY two lines:
  DECISION: <one of the option strings>
  PITCH: <one or two sentences arguing for it, addressing opposing members' concerns>
```

The agent **never sees** NPC agendas — it must infer them from statements + voting history and write boardroom rhetoric semantically aligned with each role's private manifesto. Trust persists across all 10 rounds.

---

## The 10 events (organisation-agnostic)

| # | Event | Decisions |
|---|---|---|
| 1 | New competitor entry | undercut price · double down on quality · pivot upmarket |
| 2 | Major client contract demand | accept full demands · counter-offer · walk away |
| 3 | Talent retention crisis | match offers · promote internally · accept attrition |
| 4 | Regulatory compliance ultimatum | full cooperation · limited disclosure · seek legal delay |
| 5 | Public relations incident | public apology · counter-narrative · stay silent |
| 6 | Strategic acquisition offer | accept · negotiate · reject |
| 7 | Institutional funding round | accept terms · counter-offer · seek alternatives |
| 8 | Operational innovation decision | aggressive rollout · phased rollout · defer |
| 9 | Internal whistleblower report | open investigation · internal HR review · dismiss claim |
| 10 | Strategic exit decision | acquisition · IPO · stay private |

Per-episode jitter ±25% on agenda weights and ±15% on consequence magnitudes prevents trajectory memorisation.

---

## Why this is novel

Multi-agent envs in this space are usually symmetric games. **BoardSim is asymmetric, partially observable, adversarially noisy, and graded on natural-language quality**. Three design properties push it past a "pick-an-action" toy:

1. **Coalition pitch is a graded action channel.** Each step the agent emits `(decision, coalition_pitch)`. The pitch is **scored by sentence-transformer cosine similarity** against each opposing NPC's hidden manifesto, and a high-similarity pitch can swing up to 55% of that NPC's vote weight. The agent must learn what each role secretly cares about and articulate it — implicit theory-of-mind, graded semantically.

2. **Trust persists and feeds back into NPC behaviour.** NPCs that repeatedly lose votes lower their confidence in the CEO (`Δtrust ≈ ±0.08/round`), reducing their effective vote weight. Building early trust makes the endgame easier; burning it makes NPCs increasingly adversarial.

3. **Events are shuffled and consequence-noised per episode.** Different event order and ±15% magnitude variance per seed forces genuine policy generalisation.

---

## The baseline — same model, LoRA disabled

The trained-vs-baseline comparison runs **the same Qwen3-4B**, on the **same paired seeds**, with the LoRA adapter context-managed off:

```python
# Fine-tuned (with LoRA active)
trained_finals = run_episodes(model, seeds=HELDOUT)

# Same model, LoRA disabled — apples-to-apples base reference
with model.disable_adapter():
    base_finals = run_episodes(model, seeds=HELDOUT)
```

This isolates the *fine-tuning delta* from the *language-model prior*. A coin flip is not a competitive opponent for a 4 B model selecting among 3 well-formed strings; the only honest question is "did training move the same model in the right direction?"

Statistical comparison: paired t-test, Wilcoxon signed-rank, Cohen's d, bootstrap 95% CI — all on the per-seed paired delta `trained − base`.

---

## Reward design (high level — full math in `MECHANICS.md`)

```
Per step:
  reward  = 1.0  if CEO won the weighted vote, else −0.4
          + 0.5 × (Δ profitability normalised)
          + 0.5 × (Σtrust_after − Σtrust_before)
          + 0.6 × mean(pitch_semantic_score[opposing])
          + 0.05 if pitch non-empty (bootstrap)
          − 0.5  if action malformed

Terminal:
  − 2.0  if runway exhausted
  + bonus from final profitability score (0–100)
```

Pitch score is `cosine(SBERT(pitch), SBERT(role_manifesto)) ∈ [0,1]`, clipped. The keyword-match scorer used in earlier revisions is gone.

---

## Results

The headline comparison is **fine-tuned Qwen3-4B vs base Qwen3-4B** on a held-out paired-seed eval inside `notebooks/train_grpo_v2.ipynb`. Numbers below are populated after the Colab run; the notebook saves all artefacts to `assets/`.

| Metric | Base Qwen3-4B | Fine-tuned (Qwen3-4B + LoRA) |
|---|---|---|
| Final profitability (mean ± std) | TBD | TBD |
| Win-rate (paired delta > 0) | n/a | TBD |
| Mean episode reward | TBD | TBD |
| ToM probe (predict opposing NPC) | TBD | TBD — chance ≈ 25% |
| Format-compliance rate | TBD | TBD |
| Pitch usage rate | TBD | TBD |

**Training reward / loss / format-compliance / pitch-rate curves**:
![Training curves](assets/reward_curve.png)

**Profitability distribution — base vs fine-tuned on same seeds**:
![Before/after profitability](assets/before_after.png)

**Per-event boardroom win-rate** *(the most diagnostic plot — shows which event types fine-tuning helps with)*:
![Per-event win rate](assets/per_event_winrate.png)

**Trust trajectory across rounds — fine-tuned vs base**:
![Trust trajectory](assets/trust_trajectory.png)

The `scripts/random_baseline.py` script is retained only as an environment-health smoke test (it confirms reachability and that rewards stay in range). It is **not** the canonical baseline.

---

## Quickstart — run the env locally

```bash
# 1. install env deps
cd envs/board_sim_env && pip install -e .

# 2. self-test (no HTTP, in-process)
python server/board_sim_env_environment.py

# 3. spin up the FastAPI server
uvicorn server.app:app --port 8000
# Swagger: http://localhost:8000/docs
```

```python
# 4. drive it from a Python client
from board_sim_env import BoardSimEnv
from board_sim_env.models import BoardSimAction
import random

with BoardSimEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(seed=42)
    obs = result.observation
    while not result.done:
        result = env.step(BoardSimAction(
            decision=random.choice(obs.options),
            coalition_pitch="",
        ))
        obs = result.observation
```

## Quickstart — train

Open `notebooks/train_grpo_v2.ipynb` in Colab. Add `HF_TOKEN` and `WANDB_API_KEY` to Colab Secrets. Run all cells. The notebook (a) loads base Qwen3-4B, (b) runs the base-model baseline on held-out seeds, (c) wraps with LoRA, (d) trains with GRPO, and (e) runs paired same-seed fine-tuned-vs-base evaluation with full statistics.

---

## Repository layout

```
.
├── envs/board_sim_env/                   # OpenEnv environment (deploys to HF Space)
│   ├── client.py                         # EnvClient subclass
│   ├── models.py                         # BoardSimAction / BoardSimObservation / BoardState
│   ├── openenv.yaml                      # spec_version: 1, name, runtime: docker
│   ├── pyproject.toml                    # pinned openenv-core==0.2.3
│   └── server/
│       ├── app.py                        # FastAPI wiring
│       ├── board_sim_env_environment.py  # reset/step, NPC sim, semantic pitch scorer, reward
│       ├── requirements.txt              # incl. scikit-learn + sentence-transformers
│       └── Dockerfile
├── notebooks/
│   ├── train_grpo_v2.ipynb               # canonical Colab notebook
│   └── train_grpo.ipynb                  # mirror
├── Training.py                           # canonical script — notebooks are generated from this
├── boardsim_local.py                     # local dev script (no HF / no Docker)
├── scripts/
│   ├── random_baseline.py                # env-health smoke test only
│   ├── test_server.py                    # in-process FastAPI test
│   └── test_client.py                    # client ↔ server round-trip
├── assets/                               # reward_curve · before_after · per_event_winrate · trust_trajectory
├── MECHANICS.md                          # full math reference
└── README.md                             # ← you are here
```

---

## License

Apache-2.0
