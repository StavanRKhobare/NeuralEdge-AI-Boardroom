# BoardSim — Full Mechanics Reference

> This document is the authoritative math and design reference for the NeuralEdge AI Boardroom environment.  
> Target audience: hackathon judges who want the internals, and future contributors.  
> See `README.md` for the submission overview.

---

## 1. State variables — every field, every formula

State lives in `BoardState.state_dict`, initialized in `reset()` at `board_sim_env_environment.py:471`.

### Core company state (mutated by consequences each round)

| Field | Initial value | Range | Unit | Meaning |
|---|---|---|---|---|
| `revenue` | 2,000,000 | [0, 1e12] | USD/year | Annual recurring revenue |
| `burn_rate` | 1,200,000 | [0, 1e10] | USD/month | Monthly cash expenditure |
| `runway_months` | 14.0 | [0, 120] | months | Time until cash = 0 |
| `product_readiness` | 0.45 | [0, 1] | fraction | Shippability of the product |
| `market_share` | 0.08 | [0, 1] | fraction | % of total addressable market |
| `team_morale` | 0.70 | [0, 1] | fraction | Engineering team happiness/retention |
| `investor_confidence` | 0.65 | [0, 1] | fraction | Board investors' belief in success |
| `regulatory_risk` | 0.20 | [0, 1] | fraction | Legal/compliance exposure |

### Coalition state

| Field | Initial | Range | Update rule |
|---|---|---|---|
| `trust[CTO]` | 0.5 | [0.1, 1.0] | ±0.05 per round depending on vote alignment |
| `trust[CFO]` | 0.5 | [0.1, 1.0] | same |
| `trust[Investor Rep]` | 0.5 | [0.1, 1.0] | same |
| `trust[Independent]` | 0.5 | [0.1, 1.0] | same |

Trust update (applied after every vote):
```
for each NPC:
    if NPC voted for the winning decision:
        trust[NPC] = clamp(trust[NPC] + 0.05, 0.1, 1.0)
    else:
        trust[NPC] = clamp(trust[NPC] - 0.05, 0.1, 1.0)
```
Trust influences NPC confidence from the *next* round onward:  
`trust_bias = (trust[role] - 0.5) × 0.30`  → added to that NPC's option-scoring, range `[-0.15, +0.15]`.

### Bookkeeping fields

| Field | Purpose |
|---|---|
| `round` | 1..10, increments each step |
| `profitability_score` | Recomputed composite at end of each step |
| `history` | Per-round decision log (agent_decision, winning_decision, vote_tally, pitch_scores, …) |
| `trust_history` | Per-round snapshot of all 4 trust values |
| `done_reason` | `"runway_exhausted"` / `"acquisition"` / `"finished_10"` / `None` |
| `winning_decision` | Last round's vote winner |

---

## 2. Profitability score — the composite health metric

```
profitability_score = clamp(raw, 0, 100)

raw =
  min(revenue / 8_000_000, 1.0) × 22        # revenue term       (max 22)
  + max(0, 1 − burn_rate / 1_400_000) × 18  # burn efficiency    (max 18)
  + min(runway_months / 18.0, 1.0) × 18     # runway term        (max 18)
  − max(0, (6 − runway_months) / 6) × 10    # low-runway penalty (bites below 6mo)
  + min(market_share, 0.50) / 0.50 × 14     # market share       (max 14)
  + product_readiness × 10                  # product readiness  (max 10)
  + team_morale × 7                         # team morale        (max  7)
  + investor_confidence × 11               # investor confidence (max 11)
  − regulatory_risk × 18                    # regulatory drag    (max −18)
```

**Initial state score** (with default init values) ≈ 37.3/100.  
**Theoretical maximum** = 22 + 18 + 18 + 0 + 14 + 10 + 7 + 11 − 0 = **100**.  
**Random policy** lands near 30–55 with mean ≈ 45.7 (measured over 200 episodes after §9.5 reward tweaks).

---

## 3. Next-state computation — how the simulation physics work

**Answer: yes, consequence deltas are hardcoded.** The transition is:

```
next_state = current_state + consequences[winning_decision] × (1 + ε)
    where ε ~ N(0, 0.15) per consequence value, fixed at episode reset (seeded)

runway_months -= _advance_runway()     # depends on current revenue/burn, not the action
trust[role] += ±0.05 per NPC           # based on vote alignment with winning_decision
profitability_score = compute_profitability_score(next_state)   # derived
```

### Runway decrement formula

```python
monthly_revenue = revenue / 12.0
net = monthly_revenue - burn_rate

if net >= 0:
    runway_months -= 0.5                        # profitable: slow burn
else:
    burn_months = min(2.0, max(1.0, abs(net) / burn_rate + 1.0))
    runway_months -= burn_months                # unprofitable: faster bleed
```

### Three layers of variability (the agent cannot memorize the optimal path)

1. **Event order shuffled per episode** — same 10 events, different sequence each seed.
2. **Consequence magnitudes ±15% Gaussian noise** — computed once at `reset()`, fixed for the episode.
3. **NPC vote positions depend on accumulated trust** — same option in round 5 produces different vote weights if you've built (or burned) coalitions in rounds 1–4.

---

## 4. NPC vote resolution

### Vote weight configuration

```
CEO: 1.5   CTO: 1.2   CFO: 1.0   Investor Rep: 1.3   Independent: 0.8
```

### NPC option scoring (per NPC, per round)

Each NPC has a hidden agenda dict (e.g. CFO: `{burn_rate: -0.60, revenue: 0.30, runway_months: 0.20, regulatory_risk: -0.25}`).

```
for each option opt:
    score[opt] = 0
    for each (metric, weight) in NPC_agenda:
        v = consequences[opt][metric]  (with unit normalization)
        score[opt] += v × weight
    score[opt] += N(0, 0.20)          # personality noise, seeded per (role, round)

NPC votes for argmax(score)
confidence = clamp(0.5 + 0.5 × margin_between_top_two, 0.05, 1.0)
           + trust_bias                # trust influences confidence
```

### Pitch persuasion mechanism

```python
pitch_score[role] = min(1.0, keyword_hits / max(4, len(agenda_keywords) // 4))
# where keyword_hits = count of role's agenda keywords present in pitch text

# Persuasion shifts up to 35% of NPC's vote weight toward CEO's pick:
shift_fraction = 0.35 × pitch_score[role]
tally[NPC's_vote]    += base_weight × (1 - shift_fraction)
tally[CEO's_decision] += base_weight × shift_fraction
```

NPC keyword lists (the hidden information the CEO must infer via ToM):

| Role | Keywords |
|---|---|
| CTO | engineering, architecture, technical, quality, morale, product, team, scalable, reliable, robust |
| CFO | burn, cash, runway, fiduciary, conservative, discipline, cost, savings, margin, compliance, prudent, fiscal |
| Investor Rep | growth, scale, 10x, tam, market, moat, ipo, exit, valuation, revenue, arr, dominate, aggressive, ambitious |
| Independent | reputation, stakeholders, trust, transparent, ethics, long-term, governance, consensus, safety, credibility |

### Tie-breaking

If two options score equally in the tally, the CEO's pick wins. This is implemented by inserting `agent_decision` first in the `ordered` dict before calling `max()`, so Python's stable `max()` breaks ties in the CEO's favour.

---

## 5. The full reward formula

Applied at the end of each `step()` call:

```
# Primary signal — normalized (§9.5)
reward  = (new_profitability_score - old_profitability_score) / 100.0

# Coalition bonus / penalty
reward += 0.5   if winning_decision == agent_decision
       else -0.2

# Trust delta (range ≈ ±0.06 per round)
reward += 0.3 × (Σtrust_after - Σtrust_before)

# Pitch bootstrap (§9.5) — fires for any non-empty pitch
if pitch_text is non-empty:
    reward += 0.05
    if any NPC opposed the CEO's pick:
        reward += 0.4 × mean(pitch_score over opposing NPCs)

# Format penalty
if agent's decision string not in round's options:
    reward -= 0.5

# Terminal penalties / bonuses (only at episode end)
if runway_months <= 0:
    reward -= 2.0               # bankruptcy (§9.5: reduced from -5)
if terminal:
    reward += event._terminal_bonus        # acquisition +30, IPO +25, stay_private +5
    reward += {+10 if final≥60, +5 if ≥40, -5 if <20}
```

### Why each term exists

| Term | Purpose |
|---|---|
| Δ score / 100 | Primary learning signal: profitability improvement per decision |
| Coalition ±0.5/−0.2 | Teaches the agent to actually win votes, not just pick good-looking options |
| Trust delta × 0.3 | Rewards long-arc coalition building across rounds |
| Pitch bootstrap +0.05 | Bootstraps the pitch channel before the model is good enough to earn keyword bonuses |
| Pitch persuasion × 0.4 | Rewards pitches that specifically target opposing NPC keywords (ToM signal) |
| Invalid −0.5 | Teaches correct output format (DECISION: / PITCH: two-line structure) |
| Bankruptcy −2.0 | Episode-ending failure signal, reduced to avoid drowning gradient |
| Terminal tiered | Long-horizon incentive toward high profitability, acquisition, or IPO |

---

## 6. When profitability is computed relative to the decision

The exact sequence inside `step()`:

```
1. old_score = compute_profitability_score(state)      ← snapshot BEFORE
2. NPC votes computed from current state + trust
3. CEO's decision + pitch → _resolve_vote() → winning_decision
4. consequences[winning_decision] × noise → applied to state
5. _advance_runway() → runway decrements
6. trust updated per NPC (±0.05)
7. new_score = compute_profitability_score(state)      ← AFTER consequences
8. reward = (new_score - old_score) / 100 + ...
9. next observation returned with new_score in obs.state
```

The CEO **never consults profitability to make its decision** — it sees last round's score in the observation, emits a decision, and then the score updates. Profitability is the *outcome metric*, not a planning input. The policy learns to predict which decisions increase profitability by observing the correlation across training episodes.

---

## 7. Training pipeline — key design decisions

### §9a: Per-round gradient flow (Option A)

The current training loop samples 1 completion from the model for **every round** of **every group member's episode**. This gives the model gradient signal for all 10 decisions per trajectory, not just the opening decision.

```
For each training step:
    Create GROUP_SIZE independent envs (different seeds → divergent trajectories)
    For each round r in 0..9:
        For each group member g:
            prompt = build_prompt(obs_g)
            completion = model.generate(prompt, do_sample=True)   ← gradient-connected
            obs_g = env_g.step(parse(completion))
            ep_reward[g] += obs_g.reward
    advantages = GRPO(ep_rewards)    # group-relative normalization
    For each (g, r) completion:
        loss = advantage[g] × NLL(completion) / (GROUP_SIZE × n_rounds)
        + β_KL × KL(π_θ || π_ref)
    optimizer.step()
```

Total forward passes per training step: 10 rounds × 4 group members × 2 (policy + ref) = **80 forward passes**.

### §9c: KL penalty

A frozen copy of the initial model (`ref_model`) computes reference log-probs. KL ≈ `current_loss - ref_loss` per completion, clamped at 0. Coefficient β = 0.04.  
Purpose: prevents the policy from drifting into degenerate text patterns (always emitting the same decision, empty pitches) that lock in low-reward equilibria.

### §9.5: Reward normalization

Three changes to the reward function to improve gradient quality:
1. **Δscore ÷ 100** — brings profitability delta (typically −5 to +10) to the same scale as the coalition term (±0.5)
2. **Bankruptcy penalty −2 (was −5)** — one bad arc was drowning 9 rounds of positive signal
3. **Pitch bootstrap +0.05** — needed to push a 0.6B model into using the pitch channel before it's good enough to earn keyword bonuses

---

## 8. Theory-of-Mind — what's actually measured

"ToM" in this environment has a specific, narrow meaning: **can the agent infer what vocabulary each NPC uses when reasoning**, given only observation of statements and votes?

The grading mechanism is keyword overlap: `pitch_score[role] = hits / threshold`. This is coarse but measurable without human annotation.

A stronger ToM measurement (planned, not yet implemented): after each episode, ask the model "Given round 3's event and the CFO's statement, predict the CFO's vote." Compare predicted vs actual. Random baseline = 25% (1 in 4 options). Exceeding 50% indicates the model has learned the CFO's agenda.

The trust trajectory is a secondary ToM diagnostic: if trust rises across rounds, the model is consistently picking decisions that align with NPC preferences, which requires some implicit modeling of their objectives.
