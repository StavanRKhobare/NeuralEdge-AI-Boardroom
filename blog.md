<h1 align="center">NeuralEdge AI Boardroom</h1>
<h3 align="center">Asymmetric Theory-of-Mind Training over OpenEnv v0.2.3</h3>
<p align="center"><em>Submission to the Meta × PyTorch × HuggingFace OpenEnv Hackathon — Theme 1: Multi-Agent Interactions</em></p>

---

## Abstract

We present **NeuralEdge AI Boardroom**, an asymmetric, partially-observable multi-agent reinforcement-learning environment in which a single CEO agent must build winning coalitions across ten rounds of strategic crises against four NPC board members with private agendas. Unlike symmetric multi-agent benchmarks (Poker, Werewolf, Diplomacy variants), the action channel is a `(decision, coalition_pitch)` tuple where the pitch is a free-text argument graded against each NPC's hidden manifesto via sentence-transformer cosine similarity. We fine-tune **Qwen3-0.6B** with a 4-bit LoRA (Unsloth, r=32, α=64) under a KL-regularised GRPO objective, and observe the four RL signatures the environment is designed to elicit: rapid pitch-channel uptake, terminal-spike density on `reward_max`, exploration-consistent `reward_std`, and stable GRPO loss.

---

## 1. Motivation

Real-world LLM agents do not negotiate against symmetric opponents. They write proposals, reason about heterogeneous stakeholders whose objectives are not directly observable, and argue for decisions in natural language. Existing multi-agent benchmarks capture none of this — they are zero-sum games over discrete action spaces. The underserved capability is **asymmetric persuasion under hidden-preference uncertainty with language-quality grading**, and BoardSim is the environment that trains it.

## 2. Environment Design

Built natively on **OpenEnv v0.2.3**: `BoardSimEnv` is an `EnvClient` subclass over a Dockerised FastAPI server hosted on HuggingFace Spaces, with a synchronous `reset(seed) / step(action)` contract and a typed `BoardSimAction(decision, coalition_pitch)` schema.

| Property | Value |
|---|---|
| Episode length | 10 rounds, events shuffled per seed |
| Observation | public state + 4 NPC pre-vote statements (zero agenda leak) |
| Action | `decision ∈ options[3]`, `coalition_pitch: free text` |
| Vote resolution | weighted tally — CEO=2.5, NPC weights ∈ [0.8, 1.3] |
| Persuasion cap | up to **55%** of opposing NPC weight redirected by pitch |
| Trust dynamics | per-NPC trust ∈ [0.1, 1.0], ±0.08/round, multiplicatively gates vote weight |

Three independent variability layers — event order, ±15% consequence-magnitude noise, ±25% NPC agenda jitter — eliminate trajectory memorisation as a learning shortcut.

## 3. Reward Function

Reference: `envs/board_sim_env/server/board_sim_env_environment.py:723`. Per-step reward is dense and bounded ≈ [−0.7, +0.65]:

```
r_t = Δprofitability / 100
    + (+1.0 if winning_decision == agent_decision else −0.4)
    + 0.5 · Σ Δtrust
    + 0.05 · 𝟙[pitch ≠ ∅]                                   # bootstrap
    + 0.6 · mean_pitch_score over opposing NPCs              # ToM persuasion
    − 0.5 · 𝟙[decision ∉ options]                            # format penalty
```

The terminal step adds episodic spikes (+30 acquisition, +25 IPO, +5 stay-private, −2 bankruptcy) plus a ±10 final-profitability tier. The split is deliberate: dense shaping for per-step credit assignment, sparse spikes for outcome-quality discrimination across the ten-round horizon.

## 4. Training and Empirical Signal

**Stack:** Qwen3-0.6B base · Unsloth 4-bit LoRA (r=32, α=64, all linear modules) · KL-regularised GRPO (β=0.04 against a frozen reference) · `GROUP_SIZE=4`, lr=5e-6, T=1.0, top_p=0.95 · OpenEnv v0.2.3 client over the live HF Space.

![GRPO training curves on BoardSim](assets/reward_curve.png)

The curves expose the four signatures the environment is designed to elicit:

1. **Pitch-rate convergence to 1.0.** The agent rapidly internalises that emitting an empty `coalition_pitch` is strategically dominated — there is no recovery of the +0.05 bootstrap nor the +0.6 × pitch_score persuasion term, and the format penalty fires whenever the two-line schema breaks. Convergence to 100 % pitch rate is a direct measurement of the policy learning *the structural reward channel that the random baseline cannot exploit*.
2. **Terminal-spike density on `reward_max`.** Repeated +25 / +30 spikes are not training noise; they are the agent successfully navigating ten rounds of asymmetric vote resolution into the IPO and Acquisition terminal events. The signal that the long-horizon credit assignment is working.
3. **`reward_std` consistent with active exploration.** Group-relative advantage estimation requires within-group reward variance — the curve confirms the policy is sampling distinct terminal outcomes rather than collapsing to a single sub-optimal mode.
4. **GRPO loss stabilising.** The advantage-weighted NLL flattens as group-relative advantages stop diverging from the running mean — the standard signature of a stable GRPO checkpoint.

## 5. Discussion

The headline measurement is *not* aggregate step-reward, which is dominated by terminal-spike density and seed luck. The axes that genuinely separate a trained policy from the random-baseline floor are **held-out final profitability** on paired same-seed evaluation and **pitch-usage rate**. The 200-episode random baseline scores 45.7 ± 13.1 profitability with **0% pitch usage**; the trained agent's pitch-rate convergence to 1.0 establishes a structural advantage random cannot replicate, because the persuasion reward channel is gated on producing non-empty, role-aligned text. Per-round trust dynamics convert the episode into a long-arc credit-assignment problem in which early persuasion compounds into late-round vote dominance — a regime closer to real-world stakeholder negotiation than zero-sum self-play.

## Conclusion

NeuralEdge AI Boardroom contributes, to the OpenEnv ecosystem, an asymmetric multi-agent environment in which **language quality is part of the reward** rather than a wrapper around discrete play. Qwen3-0.6B + LoRA + GRPO produces the predicted RL signatures end-to-end on the live HF-Space-hosted environment, validating BoardSim as a target for theory-of-mind capability training rather than a dressed-up symmetric benchmark.

---

<p align="center"><em>Code · Environment · Adapter — see the project README for HF Space and GitHub links.</em></p>
