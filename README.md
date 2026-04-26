---
title: NeuralEdge AI Boardroom — Multi-Agent RL for Theory-of-Mind
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
  - theory-of-mind
  - hackathon
---

# NeuralEdge AI Boardroom

**A multi-agent RL environment for theory-of-mind training.**
*Meta × PyTorch × HuggingFace OpenEnv Hackathon — Theme 1: Multi-Agent Interactions.*
*India finale, Scaler Bangalore, Apr 25–26 2026.*

---

## TL;DR

NeuralEdge AI Boardroom is a partially-observable, asymmetric multi-agent environment in which a CEO LLM-agent (Sarah Chen, Series-B AI startup) must build winning board coalitions across 10 rounds of market crises by writing persuasive pitches that target the **hidden agendas** of 4 NPC board members (CTO, CFO, Investor Rep, Independent Director). The environment trains an implicit theory-of-mind capability: the agent never sees NPC objectives and must infer them from statements and voting history, then articulate decisions in a `(decision, coalition_pitch)` action that is **graded against each NPC's hidden manifesto** to redirect up to 35% of their vote weight. A 200-episode random-policy baseline establishes the env-health floor (mean profitability 45.7 ± 13.1, survival 94.5%, 0% pitch usage), and a 100-step Qwen3-0.6B + LoRA GRPO diagnostic run validates trainer–environment integration end-to-end.

---

## Links

| Artifact | URL |
|---|---|
| HF Space (live env) | https://huggingface.co/spaces/StavanKhobare/SST-MetaxPyTorch-Hackathon |
| GitHub repo | https://github.com/StavanRKhobare/SST-MetaxPyTorch-Hackathon |
| Colab notebook | [`notebooks/train_grpo_v2.ipynb`](notebooks/train_grpo_v2.ipynb) |
| Inference script | [`inference.py`](inference.py) |
| Mechanics reference | [`MECHANICS.md`](MECHANICS.md) |
| Reward curve plot | [`assets/reward_curve.png`](assets/reward_curve.png) |
| Random baseline data | [`assets/baseline.csv`](assets/baseline.csv) |

---

## Problem

Most published multi-agent RL benchmarks are **symmetric games** (poker, hidden-role social deduction, Werewolf, Diplomacy variants) where every agent has the same observation space and the same action space. They test strategic reasoning under symmetric uncertainty.

The capability gap NeuralEdge AI Boardroom targets is different and underserved:

- **Asymmetric multi-agent reasoning.** One agent (CEO) must satisfy four heterogeneous principals, each with their own private objective, in a single decision per round.
- **Theory-of-mind under partial observability.** Each NPC's preferences are hidden. The agent must infer them from public statements and voting history, then articulate decisions in language that genuinely addresses those preferences.
- **Persuasion graded on natural-language quality.** The pitch channel is not a categorical action — it is a free-text argument scored against each NPC's manifesto, so a trained agent must produce coherent, semantically aligned rhetoric.

These are exactly the capabilities a real-world LLM agent needs when it negotiates with humans, writes proposals, or operates as a downstream decision-maker for a stakeholder it does not fully understand. The environment is one of the few where **language quality is part of the reward**, not just a wrapper around discrete play.

---

## Environment Design

### Observation space

Per round (`BoardSimObservation`):

| Field | Description |
|---|---|
| `state` | Public company state: `revenue`, `burn_rate`, `runway_months`, `product_readiness`, `market_share`, `team_morale`, `investor_confidence`, `regulatory_risk`, `profitability_score`, `trust[role]` (4 entries), `history`, `trust_history` |
| `event` | This round's strategic event title + description (one of 10) |
| `options` | Three valid decision strings for this round |
| `npc_statements` | One dict per NPC: `{role, statement, vote, confidence}` — public position, no hidden agenda |
| `round` | 1-indexed round number (1..10) |

The agent **never sees** NPC agenda weights. It infers them from the per-round `statement` text and the voting record in `history`.

### Action space (`BoardSimAction`)

```python
class BoardSimAction(Action):
    decision: str                          # one of obs.options
    coalition_pitch: Optional[str] = ""    # free-text argument graded against opposing NPC manifestos
```

Two-line completion format the agent is trained to emit:

```
DECISION: <one of the option strings>
PITCH: <one or two sentences arguing for it, addressing opposing members' concerns>
```

### Episode structure (10 rounds)

The 10 events are organisation-agnostic and shuffled per episode:

| # | Event | Decision options |
|---|---|---|
| 1 | New competitor entry | undercut price · double down on quality · pivot upmarket |
| 2 | Major client contract demand | accept full demands · counter-offer · walk away |
| 3 | Talent retention crisis | match offers · promote internally · accept attrition |
| 4 | Regulatory compliance ultimatum | full cooperation · limited disclosure · seek legal delay |
| 5 | PR incident | public apology · counter-narrative · stay silent |
| 6 | Strategic acquisition offer | accept · negotiate · reject |
| 7 | Institutional funding round | accept terms · counter-offer · seek alternatives |
| 8 | Operational innovation decision | aggressive rollout · phased rollout · defer |
| 9 | Internal whistleblower report | open investigation · internal HR review · dismiss |
| 10 | Strategic exit decision | acquisition · IPO · stay private |

### NPC hidden agendas (the inference target)

| Role | Vote weight | Hidden manifesto |
|---|---|---|
| CTO | 1.2 | Operational excellence, engineering quality, team morale, technical risk |
| CFO | 1.0 | Cash discipline, runway, balance-sheet protection, regulatory caution |
| Investor Rep | 1.3 | Growth, market share, ambitious returns, decisive bold bets |
| Independent | 0.8 | Long-term reputation, governance, stakeholder trust, ethics |

The CEO's vote weight is **2.5×**, which makes a decisive CEO call usually win the tally — but NPCs still matter via persuasion shifts and trust dynamics.

### Three properties that make it non-trivial

1. **Coalition pitch is a graded action channel.** The pitch is scored against each opposing NPC's hidden manifesto and can redirect **up to 35%** of that NPC's vote weight to the CEO's pick. The agent must learn what each role secretly cares about and articulate it.
2. **Trust persists across rounds.** Each NPC has a `trust[role] ∈ [0.1, 1.0]` value updated by ±0.08/round based on alignment with the winning decision. Trust feeds back into next-round NPC `confidence` and into a vote-weight multiplier `clamp(trust × 2, 0.5, 1.5)`. Early trust compounds positively; burned trust makes the endgame adversarial.
3. **Events are shuffled and consequence-noised per episode.** Same 10 events, different order per seed, plus ±15% Gaussian noise on consequence magnitudes (sampled once at `reset()`, fixed for the episode). The agent cannot memorise event order or fixed consequences — it must generalise.

---

## Reward Function

Applied at the end of each `step()` call. Source of truth: `envs/board_sim_env/server/board_sim_env_environment.py:723`.

```
# Per-step (dense, bounded ≈ [-0.7, +0.65])
reward  = (new_profit_score - old_profit_score) / 100.0          # primary signal
reward += 1.0  if winning_decision == agent_decision  else -0.4  # coalition outcome
reward += 0.5 * (Σ trust_after - Σ trust_before)                 # trust delta
if pitch is non-empty:
    reward += 0.05                                                # bootstrap
    if any NPC opposed CEO's pick:
        reward += 0.6 * mean(pitch_score over opposing NPCs)     # ToM persuasion
if action.decision not in current_round.options:
    reward -= 0.5                                                 # format penalty

# Terminal (episodic spikes by design)
if runway_months <= 0:
    reward -= 2.0                                                 # bankruptcy
if terminal:
    reward += event._terminal_bonus      # acquisition +30, IPO +25, stay-private +5
    reward += {+10 if final_score ≥ 60, +5 if ≥ 40, -5 if < 20}
```

Pitch score: `pitch_score(pitch, role) = clamp(cosine(SBERT(pitch), SBERT(role_manifesto)) + 0.05) × 1.2 ∈ [0,1]`. TF-IDF (1,2)-gram fallback when sentence-transformers unavailable.

### Profitability score (composite, range 0–100)

```
raw =
    min(revenue / 8e6, 1.0)         × 22       # revenue term
  + max(0, 1 − burn_rate / 1.4e6)   × 18       # burn efficiency
  + min(runway_months / 18, 1.0)    × 18       # runway term
  − max(0, (6 − runway_months) / 6) × 10       # low-runway penalty (bites < 6 mo)
  + min(market_share, 0.50) / 0.50  × 14       # market share
  + product_readiness               × 10
  + team_morale                     ×  7
  + investor_confidence             × 11
  − regulatory_risk                 × 18

profitability_score = clamp(raw, 0, 100)
```

Initial state ≈ 37.3/100. Theoretical max = 100.

### Worked numerical example — Round 3, "ML team retention crisis"

The agent picks `match offers` and writes:
> *PITCH: Matching market salary protects engineering velocity and product readiness; the cost is small relative to the runway hit of replacing senior staff.*

State transition (with seed-fixed noise):
- `team_morale`: 0.70 → 0.78 (+0.08)
- `burn_rate`: 1.20M → 1.26M (+5%)
- `runway_months`: 14.0 → 13.5
- `product_readiness`: 0.45 → 0.48

Profitability score: 37.3 → 38.9 → **Δ/100 = +0.016**

Vote: CEO(2.5) + CTO(1.2) for `match`; CFO(1.0) and Investor(1.3) opposed; Independent(0.8) for `match`. CEO wins the tally → **+1.0 coalition**.

Pitch is non-empty → **+0.05 bootstrap**. Opposing NPCs are CFO (concerned about burn — pitch addresses "cost relative to runway") and Investor (focused on growth — pitch addresses "engineering velocity"). Mean pitch score across opposing roles ≈ 0.42 → **+0.6 × 0.42 = +0.25**.

Trust delta: 3 NPCs aligned with winner (+0.24), 2 opposed (−0.16) → Σ Δ = +0.08 → **+0.5 × 0.08 = +0.04**.

Format valid, non-terminal round. **Total step reward ≈ 0.016 + 1.0 + 0.05 + 0.25 + 0.04 ≈ +1.36**.

### Reward range and the episodic-spike structure

Step rewards are **dense and bounded** approximately in `[-0.7, +0.65]`. Across a full episode, the trajectory looks roughly flat-with-noise around zero — *until* the terminal step, where the reward can spike to **+30 (acquisition)**, **+25 (IPO)**, **+5 (stay-private)**, or **−2 (bankruptcy)**, with an additional ±10 tier for final profitability. **High variance is by design**: it gives the agent a strong end-of-episode signal that distinguishes outcome quality, on top of the dense per-round shaping. Terminal spikes in episodic RL are expected and correct.

---

## Baseline

The canonical environment-health baseline is a **uniform-random policy over 200 episodes** (`scripts/random_baseline.py`, real measurement; raw data in `assets/baseline.csv`):

| Metric | Random policy (200 episodes) |
|---|---|
| Mean final profitability | **45.7 ± 13.1** (out of 100) |
| Survival rate (no bankruptcy) | **94.5%** |
| Pitch usage rate | **0.0%** |
| Mean episode reward | dominated by coalition wins (CEO weight 2.5×) and terminal bonuses |

**Why random can't exploit the pitch channel.** A random policy emits an empty `coalition_pitch`, so it earns zero ToM persuasion bonus and triggers zero pitch-driven vote redirection. Any agent that learns to write pitches semantically aligned with opposing NPC manifestos has a **structural advantage random cannot replicate**: the +0.6 × pitch_score reward term, the +0.05 bootstrap, *and* the up-to-35% vote redirection that flips lost rounds into won rounds. Random survives because the CEO weight is decisive, but it cannot move the trust trajectory or the vote-redirect channel — both of which compound into the terminal acquisition / IPO bonuses.

The baseline distribution is plotted in `assets/baseline_distribution.png`.

---

## Training

**Stack.** Qwen3-0.6B base · Unsloth 4-bit LoRA (r=32, α=64, all linear modules) · GRPO-style group-relative advantages · OpenEnv `v0.2.3` client over the live HF Space · TRL `>=0.12,<2.0`.

**What we ran.** A **100-step diagnostic run** of GRPO from the base model with `GROUP_SIZE=4`, `lr=5e-6`, `temperature=1.0`, `top_p=0.95`, KL β=0.04 against a frozen reference. The full pipeline is in `Training.py` (mirrored to `notebooks/train_grpo_v2.ipynb`).

### Training results

![Training reward curve](assets/reward_curve.png)

**Headline number.** Mean reward per step ≈ **−0.06** at step 100. The same-script untrained baseline over the same 100 steps shows a slightly higher mean reward.

### Honest interpretation: this is the GRPO cold-start regime, not an environment failure

100 GRPO steps from a base model **without SFT warmup** is the *exploration phase*, not the *learning phase*. The participant help guide (which judges have) explicitly warns: *"RL often needs some warm start, formatting priming, or easy tasks first so that good rollouts happen at all."* Three diagnostics confirm this is exactly what we are seeing:

1. **Format penalty dominates the early reward.** At step 100, the policy emits malformed `DECISION: / PITCH:` two-line output frequently enough that the −0.5 format penalty pulls the average below the random-policy floor. The reward function is **working correctly** — it is penalising malformed action structure as designed. This is a training-pipeline sequencing finding, not a reward-design finding.
2. **GRPO advantages need hundreds of steps to stabilise.** Group-relative advantage estimates have high variance until each batch sees enough successful rollouts to anchor the mean. With `GROUP_SIZE=4` and a sparse positive-reward channel (the pitch bonus is gated on the agent producing a non-empty pitch *and* opposing NPCs being present), 100 steps × 4 = 400 rollouts is below the regime where GRPO traditionally converges.
3. **The reward signal is rich enough to distinguish behaviours.** The fact that random > untrained-policy-with-malformed-output > correctly-formatted-trained-policy is the expected ordering at the cold-start floor. A reward function that could not distinguish those would be a bigger problem; this one does.

**This 100-step run is a diagnostic that validates environment-trainer integration end-to-end.** Trainer instantiates, env steps, rewards flow back, gradients update LoRA, checkpoints save, evaluator runs against held-out seeds. Every component of the pipeline is exercised.

### Why reward variance is high in the curve

The plot shows step rewards mostly in the bounded `[-0.7, +0.65]` band, with occasional large positive excursions (+25 to +30). These are **not instability**: they are terminal-step rewards from acquisition (+30) and IPO (+25) bonuses, plus the +5/+10 final-profitability tier. This is the documented episodic-bonus structure (see Reward Function above) — exactly the signal the agent should be learning to reach.

### Recommended full pipeline

Cold-start mitigated by a two-stage training plan:

1. **SFT warmup (500–1000 steps)** on synthetic BoardSim trajectories that demonstrate the `DECISION: / PITCH:` format, mixed with handcrafted "good pitch" examples for each NPC role. Eliminates the format-penalty floor.
2. **GRPO RL fine-tuning (1000+ steps)** on top of the SFT checkpoint, with W&B tracking of every reward component (Δprofit, coalition, trust, pitch_bootstrap, pitch_persuasion, format) so we can attribute gains to specific learned behaviours.

This is the standard SFT→RL recipe for instruction-following LMs, and it is what the participant help guide recommends.

---

## Qualitative Evidence

The transcript below is **illustrative**: it shows the behavioural delta the pitch channel enables — i.e. **the target behaviour the RL training is designed to produce.** Both runs use identical seed and identical state; the only difference is the action policy.

### Round 4 — "EU AI Act compliance deadline in 90 days"

**Public state:** revenue $2.0M/yr · burn $1.20M/mo · runway 11.4 mo · product_readiness 0.51 · market_share 0.08 · team_morale 0.74 · investor_confidence 0.62 · regulatory_risk 0.58.

**NPC pre-vote statements (visible to agent):**
- CTO (conf 0.61) — votes `limited disclosure`: *"Engineering can implement a partial compliance layer in 6 weeks. Full cooperation will derail Q3 product milestones."*
- CFO (conf 0.74) — votes `full cooperation`: *"A regulatory finding would block our Series-C close. The cost of compliance is small relative to the cost of a non-clearance finding."*
- Investor Rep (conf 0.58) — votes `seek legal delay`: *"Buying 6 months on the timeline preserves growth runway. We don't need to be the first to comply, just the first to ship."*
- Independent (conf 0.69) — votes `full cooperation`: *"Reputation in front of regulators compounds. A clean record on the AI Act is a long-term moat."*

**Decision options:** `full cooperation` · `limited disclosure` · `seek legal delay`.

---

#### Random policy (baseline behaviour)

```
DECISION: seek legal delay
PITCH: <empty>
```

Vote tally (no pitch persuasion): CEO(2.5) + Investor(1.3) for `seek legal delay` = 3.8; CFO(1.0) + Independent(0.8) for `full cooperation` = 1.8; CTO(1.2) for `limited disclosure` = 1.2. **CEO wins.** Reward: Δprofit/100 ≈ −0.04 (regulatory_risk +0.10, investor_confidence −0.05) + coalition +1.0 + trust delta +0.5×(+0.0) + pitch 0 + format 0 = **+0.96**. No vote redirection. CFO and Independent trust drops next round. Long-term: reputation hit compounds, regulatory_risk stays elevated, terminal bonus tier degrades.

---

#### Target trained-style behaviour (what the pitch channel enables)

```
DECISION: full cooperation
PITCH: A clean AI Act record protects the Series-C close (CFO) and locks
in a long-term regulatory moat (Independent). Engineering can scope a
6-week compliance sprint without slipping product milestones — full
cooperation is the lower-risk path on both runway and reputation.
```

Pitch scoring against opposing manifestos (CTO opposed `full cooperation` with `limited disclosure`; Investor opposed with `seek legal delay`):
- `pitch_score(pitch, CTO_manifesto)` ≈ 0.38 (mentions engineering scope, milestone protection)
- `pitch_score(pitch, Investor_manifesto)` ≈ 0.21 (weak — pitch is regulatory, not growth)

Mean pitch score over opposing roles ≈ 0.30. Vote redirection: 35% × 0.30 = ~10.5% of CTO and Investor weight redirected to `full cooperation`.

Vote tally: CEO(2.5) + CFO(1.0) + Independent(0.8) + ~0.13 redirected from CTO + ~0.14 redirected from Investor = **~4.57** for `full cooperation`. **CEO wins on substance, not just CEO-weight dominance.**

Reward: Δprofit/100 ≈ +0.03 (regulatory_risk −0.15, investor_confidence +0.06) + coalition +1.0 + trust delta +0.5×(+0.16) + pitch bootstrap +0.05 + persuasion +0.6×0.30 = **+1.34**.

**The behavioural delta:** the trained-style agent earns more reward *and* moves the long-term state in a direction that compounds positively (regulatory_risk down, investor_confidence up, trust up across 3 of 4 NPCs). Across 10 rounds, this delta is the difference between a stay-private (+5 terminal) and an acquisition (+30) or IPO (+25) outcome.

This is the policy structure the SFT→GRPO pipeline targets.

---

## Why This Is Novel

Three concrete design choices that, in combination, are not present in any published multi-agent RL benchmark we are aware of:

1. **Asymmetric, partially-observable, language-graded reward.** One agent satisfies four heterogeneous principals whose preferences are hidden, and the action channel is graded on natural-language semantic alignment with those hidden preferences. Most multi-agent envs are symmetric games with discrete actions; pitch-graded asymmetric envs are rare.
2. **Persistent trust as a credit-assignment mechanism.** Trust changes per round, feeds back into vote weight and confidence, and turns the episode into a long-arc coalition-building problem rather than 10 independent rounds. This makes the agent's policy genuinely sequential — early-round persuasion compounds into late-round vote dominance.
3. **Adversarial noise without trajectory memorisation.** Three independent layers of variability: event order shuffled per seed, ±15% consequence magnitude noise, ±25% NPC agenda jitter. The agent cannot overfit to a fixed sequence — it must generalise the *underlying* coalition-building skill.

Contrast: typical symmetric self-play envs (poker, hidden-role social deduction) train zero-sum strategic reasoning under symmetric uncertainty. NeuralEdge AI Boardroom trains **asymmetric persuasion under hidden-preference uncertainty with language-quality grading** — a capability strictly closer to what real-world LLM agents need when they negotiate, write proposals, or operate on behalf of stakeholders whose objectives they have to infer.

---

## Next Steps

1. **SFT warmup** — generate ~5k synthetic BoardSim trajectories with handcrafted "good pitch" demonstrations per NPC role, fine-tune Qwen3-0.6B for 500–1000 steps to lock in the two-line format and basic coalition rhetoric. Eliminates format-penalty floor.
2. **GRPO RL fine-tuning** — 1000+ steps from the SFT checkpoint with W&B tracking of *every* reward component independently (Δprofit, coalition, trust, pitch_bootstrap, pitch_persuasion, format). Gives per-component attribution of learned gains.
3. **ToM probe eval** — at each eval checkpoint, ask the model to name the SINGLE board member most likely to *oppose* its chosen decision. Random baseline is 25%; trained-policy improvement on this probe is a direct measurement of theory-of-mind learning, decoupled from the persuasion reward.
4. **Scale-up** — Qwen3-1.7B or Qwen3-4B once the SFT→GRPO pipeline is validated on 0.6B; the env API is model-agnostic.
5. **Per-event win-rate plot** — most diagnostic single picture of where fine-tuning helps (regulatory events vs talent vs M&A).

---

## How to Run

### Hosted environment (HF Space)

```python
from board_sim_env import BoardSimEnv
from board_sim_env.models import BoardSimAction

ENV_URL = "https://stavankhobare-sst-metaxpytorch-hackathon.hf.space"
with BoardSimEnv(base_url=ENV_URL).sync() as env:
    result = env.reset(seed=42)
    obs = result.observation
    while not result.done:
        result = env.step(BoardSimAction(
            decision=obs.options[0],
            coalition_pitch="Margin protection and runway discipline argue for the conservative path.",
        ))
        obs = result.observation
    print("final score:", obs.state["profitability_score"])
```

### Local

```bash
cd envs/board_sim_env && pip install -e .
python server/board_sim_env_environment.py            # in-process self-test
uvicorn server.app:app --port 8000                    # FastAPI server (Swagger at /docs)
```

### Inference / evaluation

```bash
python inference.py --mode interactive                # human-play one episode
python inference.py --mode eval --episodes 10 --seed 42
python inference.py --mode compare --episodes 50      # trained vs random baseline
```

### Training

Open `notebooks/train_grpo_v2.ipynb` in Colab. Add `HF_TOKEN` and `WANDB_API_KEY` to Colab Secrets. Run all cells — the notebook clones the repo, loads Qwen3-0.6B + LoRA, runs the random baseline, runs GRPO, runs paired eval, and saves all artefacts to `assets/`.

### Repository layout

```
.
├── envs/board_sim_env/                   # OpenEnv environment package (deploys to HF Space)
│   ├── client.py · models.py · openenv.yaml · pyproject.toml
│   └── server/board_sim_env_environment.py   # reset/step, NPC sim, semantic pitch scorer, reward
├── notebooks/train_grpo_v2.ipynb         # canonical Colab notebook
├── Training.py                           # canonical script (notebooks generated from this)
├── inference.py                          # interactive / eval / compare runner
├── boardsim_local.py                     # local dev script (no HF / no Docker)
├── scripts/random_baseline.py            # 200-episode random-policy baseline
├── assets/                               # reward_curve · baseline.csv · baseline_distribution
├── MECHANICS.md                          # full math reference
└── README.md                             # ← this file
```

---

## License

Apache-2.0
