# BoardSim — Full Mechanics Reference

> Authoritative math and design reference for the BoardSim environment
> (organisation-agnostic boardroom simulation).
> Target audience: hackathon judges who want internals, and future contributors.
> See `README.md` for the submission overview.

---

## 1. State variables

State lives in `BoardState.state_dict`, initialised in `BoardSimEnvironment.reset()` at `envs/board_sim_env/server/board_sim_env_environment.py:536`.

### Core company state (mutated each round by event consequences)

| Field | Initial | Range | Unit | Meaning |
|---|---|---|---|---|
| `revenue` | 2,000,000 | [0, 1e12] | USD/year | Annual recurring revenue |
| `burn_rate` | 1,200,000 | [0, 1e10] | USD/month | Monthly cash expenditure |
| `runway_months` | 14.0 | [0, 120] | months | Time until cash = 0 |
| `product_readiness` | 0.45 | [0, 1] | fraction | Shippability / quality of the product |
| `market_share` | 0.08 | [0, 1] | fraction | % of total addressable market |
| `team_morale` | 0.70 | [0, 1] | fraction | Team retention / engagement signal |
| `investor_confidence` | 0.65 | [0, 1] | fraction | Board investors' belief in success |
| `regulatory_risk` | 0.20 | [0, 1] | fraction | Legal / compliance exposure |

### Coalition state

| Field | Initial | Range | Update rule |
|---|---|---|---|
| `trust[CTO]` | 0.5 | [0.1, 1.0] | ±0.08 per round depending on alignment with the *winning* decision |
| `trust[CFO]` | 0.5 | [0.1, 1.0] | same |
| `trust[Investor Rep]` | 0.5 | [0.1, 1.0] | same |
| `trust[Independent]` | 0.5 | [0.1, 1.0] | same |

Trust feeds back via two channels:
- **NPC confidence**: `confidence += (trust − 0.5) × 0.30`, clipped.
- **Vote weight multiplier**: `trust_mult = clamp(trust × 2.0, 0.5, 1.5)` applied to that NPC's tally contribution next round.

### Bookkeeping

| Field | Purpose |
|---|---|
| `round` | 1..10, increments each step |
| `profitability_score` | Composite recomputed at end of each step |
| `history` | Per-round log: agent_decision, winning_decision, vote_tally, pitch_scores, pitch_used |
| `trust_history` | Per-round snapshot of all 4 trust values |
| `done_reason` | `"runway_exhausted"` / `"acquisition"` / `"finished_10"` / `None` |
| `winning_decision` | Last round's vote winner |

---

## 2. Profitability score

```
profitability_score = clamp(raw, 0, 100)

raw =
  min(revenue / 8_000_000, 1.0) × 22         # revenue term       (max 22)
  + max(0, 1 − burn_rate / 1_400_000) × 18   # burn efficiency    (max 18)
  + min(runway_months / 18.0, 1.0) × 18      # runway term        (max 18)
  − max(0, (6 − runway_months) / 6) × 10     # low-runway penalty (bites < 6 mo)
  + min(market_share, 0.50) / 0.50 × 14      # market share       (max 14)
  + product_readiness × 10                   # product readiness  (max 10)
  + team_morale × 7                          # team morale        (max  7)
  + investor_confidence × 11                 # investor confidence (max 11)
  − regulatory_risk × 18                     # regulatory drag    (max −18)
```

Initial state ≈ 37.3/100. Theoretical max = 100.

---

## 3. Transition

```
next_state = current_state + consequences[winning_decision] × (1 + ε)
    where ε ~ N(0, 0.15) per consequence value, fixed at episode reset (seeded)

runway_months -= _advance_runway()                # depends on net cash flow
trust[role]   ±= 0.08 per NPC                     # based on alignment with winning_decision
profitability_score = compute_profitability_score(next_state)
```

### Runway decrement

```python
monthly_revenue = revenue / 12.0
net = monthly_revenue - burn_rate
if net >= 0:
    runway_months -= 0.5                       # profitable: slow burn
else:
    burn_months = min(2.0, max(1.0, abs(net) / burn_rate + 1.0))
    runway_months -= burn_months               # unprofitable: faster bleed
```

### Three layers of variability (no trajectory memorisation)

1. **Event order shuffled per episode** — same 10 events, different sequence per seed.
2. **Consequence magnitudes ±15% Gaussian noise** — sampled at `reset()`, fixed for the episode.
3. **NPC agendas ±25% sign-preserving jitter** — `_jitter_agendas(seed)` perturbs base NPC priorities each episode.

---

## 4. Vote resolution

### Vote weights

```
CEO: 2.5    CTO: 1.2    CFO: 1.0    Investor Rep: 1.3    Independent: 0.8
```

CEO weight 2.5 ensures a decisive CEO call usually wins — the agent's actions visibly move outcomes round-to-round. NPCs still matter via persuasion shifts and trust dynamics.

### NPC option scoring (per NPC, per round)

```
for each option opt:
    score[opt] = Σ over (metric, weight) in NPC_agenda:
                    consequences[opt][metric] × weight        (with unit normalisation)
    score[opt] += N(0, 0.20)                                  # personality noise

NPC votes for argmax(score)
margin     = top_two_score_difference
confidence = clamp(0.5 + 0.5 × margin + (trust − 0.5)×0.30, 0.05, 1.0)
```

Unit normalisation in scoring: `revenue /= 1e6`, `burn_rate /= 1e5`, `runway_months /= 6`. `revenue_mult` consequences are scored against the current `revenue` × the agenda weight on `revenue`.

### Pitch persuasion — semantic similarity, not keyword matching

```
ps_role = pitch_score(pitch, role) ∈ [0, 1]

# Persuasion redirects up to 55% of the NPC's vote weight to CEO's pick:
shift_frac        = 0.55 × ps_role
tally[NPC_vote]    += base_weight × (1 − shift_frac)
tally[CEO_decision] += base_weight × shift_frac
```

Where `base_weight = ROLE_WEIGHT[role] × confidence × clamp(trust[role] × 2, 0.5, 1.5)`.

The pitch scorer (`_PitchScorer` in `board_sim_env_environment.py`) has two backends:

1. **Sentence-transformer (primary)**: `all-MiniLM-L6-v2`, normalised cosine. `score = clamp((cosine + 0.05) × 1.2, 0, 1)`. Genuine sentence embeddings — semantically aligned arguments score high even with no shared tokens.
2. **TF-IDF fallback**: `(1,2)`-grams, English stop-words removed, IDF-weighted bag-of-bigrams cosine vs the role's manifesto. `score = clamp(cosine × 1.4, 0, 1)`. Token-based but properly stop-worded and IDF-weighted — already much more robust than a literal keyword count.

Set `BOARDSIM_PITCH_BACKEND=tfidf` to force the fallback (e.g. for CI without the embedding model).

### NPC manifestos (the hidden objective the CEO must infer)

| Role | Manifesto (paraphrased) |
|---|---|
| CTO | Operational excellence, engineering quality, team morale, technical risk reduction. |
| CFO | Capital discipline, runway, balance-sheet protection, regulatory caution. |
| Investor Rep | Growth, market share, ambitious returns, decisive bold bets. |
| Independent | Long-term reputation, governance, stakeholder trust, ethical responsibility. |

The full text lives in `NPC_MANIFESTOS` in the environment file.

### Tie-breaking

If two options tie in the tally, the CEO's pick wins (implementation: insert `agent_decision` first into the ordered tally before `max()`).

---

## 5. Reward formula

Applied at the end of each `step()` call:

```
# Primary signal — normalised profitability delta
reward  = (new_score − old_score) / 100.0

# Coalition bonus / penalty (magnitudes raised so CEO impact is visible)
reward += 1.0   if winning_decision == agent_decision  else −0.4

# Trust delta term
reward += 0.5 × (Σ trust_after − Σ trust_before)

# Pitch bootstrap + semantic persuasion
if pitch is non-empty:
    reward += 0.05
    if any NPC opposed the CEO's pick:
        reward += 0.6 × mean(pitch_score over opposing NPCs)

# Format penalty
if action.decision not in current_round.options:
    reward −= 0.5

# Terminal
if runway_months <= 0:
    reward −= 2.0                          # bankruptcy
if terminal:
    reward += event._terminal_bonus        # acquisition +30, IPO +25, stay-private +5, etc.
    reward += {+10 if final ≥ 60, +5 if ≥ 40, −5 if < 20}
```

| Term | Purpose |
|---|---|
| Δ score / 100 | Primary learning signal: profitability improvement per decision |
| Coalition ±1.0 / −0.4 | Teaches the agent to actually win votes, not pick "good-looking" options |
| Trust × 0.5 | Rewards long-arc coalition building across rounds |
| Pitch bootstrap +0.05 | Ensures the pitch channel is exercised before the model is good enough to earn semantic bonuses |
| Pitch persuasion × 0.6 | Rewards pitches semantically aligned with opposing NPC manifestos (ToM signal) |
| Invalid −0.5 | Format-compliance signal (DECISION: / PITCH: two-line structure) |
| Bankruptcy −2.0 | Episode-ending failure signal |
| Terminal tiered | Long-horizon incentive toward high profitability, acquisition, or IPO |

---

## 6. Step ordering

```
1. old_score      = compute_profitability_score(state)            # snapshot BEFORE
2. NPC votes computed from current state + trust
3. CEO decision + pitch → _resolve_vote() → winning_decision
4. consequences[winning_decision] × noise → applied to state
5. _advance_runway()
6. trust updated per NPC (±0.08)
7. new_score      = compute_profitability_score(state)            # AFTER consequences
8. reward = (new_score − old_score)/100 + coalition + trust + pitch + ...
9. next observation returned with new_score in obs.state
```

The CEO **never consults profitability to make its decision** — it sees the previous round's score in the observation, emits a decision, and then the score updates. Profitability is the *outcome metric*, not a planning input.

---

## 7. Training pipeline

### Per-round gradient flow

The training loop samples one completion per round, per group member. Every one of the 10 decisions in a trajectory contributes gradient signal — not just the opening decision.

```
For each training step:
    Create GROUP_SIZE independent envs (different seeds)
    For each round r in 0..9:
        For each group member g:
            prompt = build_prompt(obs_g)
            completion = model.generate(prompt, do_sample=True)   # gradient-connected
            obs_g = env_g.step(parse(completion))
            ep_reward[g] += obs_g.reward
    advantages = GRPO(ep_rewards)            # group-relative normalisation
    For each (g, r) completion:
        loss = advantage[g] × NLL(completion) / (GROUP_SIZE × n_rounds)
             + β_KL × KL(π_θ || π_ref)
    optimizer.step()
```

### KL penalty

A frozen reference model computes reference log-probs. KL ≈ `current_loss − ref_loss` per completion, clamped at 0. Coefficient β = 0.04. Prevents drift into degenerate text patterns (always emitting the same decision, empty pitches).

### Reward normalisation

Three normalisations in the reward function so terms are commensurate:
1. **Δ score ÷ 100** — brings profitability delta into the same scale as the coalition term.
2. **Bankruptcy penalty −2** (was −5) — one bad arc no longer drowns 9 rounds of positive signal.
3. **Pitch bootstrap +0.05** — kickstarts the pitch channel before the model is good enough to earn semantic bonuses.

---

## 8. The baseline — same Qwen3-0.6B with LoRA disabled

Earlier revisions compared the trained policy against a uniform-random policy. A coin flip is not a meaningful opponent for a 4 B language model picking among 3 well-formed strings — it can only highlight that the LM ≠ noise, which is not the relevant question.

The current baseline runs **the same Qwen3-0.6B**, on the **same paired seeds**, with the LoRA adapter context-managed off. Implementation (see `Training.py` / `notebooks/train_grpo_v2.ipynb`):

```python
# Fine-tuned (LoRA active)
trained_finals = run_episodes(model, seeds=HELDOUT)

# Same model, LoRA disabled — apples-to-apples base reference.
with model.disable_adapter():
    base_finals = run_episodes(model, seeds=HELDOUT)
```

Statistical comparison on the per-seed paired delta `trained − base`:

- Paired t-test
- Wilcoxon signed-rank
- Cohen's d
- Bootstrap 95% CI on the mean delta
- Win-rate (fraction of seeds where trained > base)

---

## 9. Theory-of-Mind — what's actually measured

ToM in this environment has a specific, narrow meaning: **can the agent infer what each NPC privately values**, given only their statements and prior votes?

It is graded two ways:

1. **Pitch persuasion score**: `cosine(SBERT(pitch), SBERT(role_manifesto))`. A pitch that genuinely articulates the role's priorities scores above ~0.4; a pitch that is merely topically adjacent scores ~0.1; off-topic pitches score ~0.0. This replaces the earlier keyword-overlap metric, which the agent could trivially game.
2. **ToM probe**: ask the model to name the SINGLE board member most likely to *oppose* its chosen decision. Random baseline = 25% (1 of 4). The probe is run for both the fine-tuned policy and the disable-adapter base — the delta isolates what fine-tuning taught the model about its boardroom.

Trust trajectory across 10 rounds is a secondary diagnostic: rising trust for 3+ NPCs indicates the agent is consistently picking decisions aligned with their private preferences, which requires implicit modelling of those preferences.
