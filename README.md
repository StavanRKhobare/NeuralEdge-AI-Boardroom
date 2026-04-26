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
*Meta × PyTorch × HuggingFace OpenEnv Hackathon*

## What is this?
NeuralEdge AI Boardroom is an asymmetric multi-agent environment where an LLM-agent (the CEO) must build winning board coalitions. Across 10 rounds of market crises, the agent must write persuasive pitches to sway 4 NPC board members (CTO, CFO, Investor, Independent), each with a **hidden agenda**.

Unlike standard symmetric RL games (like Poker), our environment grades **natural language persuasion**. The agent must infer hidden preferences from public statements and generate targeted rhetoric to swing votes.

## Quick Links
- **[Blog Post (Deep Dive)](blog.md)**: Read our full breakdown of the innovation and reward logic.
- **[Mechanics](MECHANICS.md)**: Full mathematical reference.
- **[HF Space (Live Env)](https://huggingface.co/spaces/StavanKhobare/SST-MetaxPyTorch-Hackathon)**
- **[Merged 16-bit Model](https://huggingface.co/StavanKhobare/SST-MetaxPyTorch-Hackathon-Merged16bit)**

## How it Works

The agent emits actions in a strict two-line format:
```text
DECISION: <one of 3 options>
PITCH: <1-2 sentences arguing for it, addressing opposing members' concerns>
```
The environment scores the `PITCH` against the hidden manifestos of opposing NPCs using sentence-transformers (SBERT). High-quality pitches redirect up to 55% of the NPC's voting weight to the CEO's choice. 

## Training Evidence

We trained **Qwen3 (1.7B/0.6B)** using **GRPO (Group Relative Policy Optimization)** via Unsloth in 4-bit.

![Reward Curve](assets/reward_curve.png)

**Key Takeaways from the Training Graphs:**
- **Pitch Rate Convergence**: The agent quickly realizes that writing targeted pitches is a structural advantage. Pitch usage goes from erratic to exactly **1.0 (100%)**.
- **Terminal Reward Spikes**: The reward graphs show distinct spikes up to `+30`. This proves the model isn't just surviving; it's actively navigating the environment to trigger the massive "Strategic Acquisition" terminal bonuses.
- **Loss & Variance**: `reward_std` and `loss` show high initial exploration variance that stabilizes as the policy masters the environment's asymmetric dynamics.

For a full breakdown of how we quantify this learning via our **Theory-of-Mind (ToM) Probe**, please read our [blog.md](blog.md).

## Running the Code

**Hosted environment:**
```python
from board_sim_env import BoardSimEnv
from board_sim_env.models import BoardSimAction

with BoardSimEnv(base_url="https://stavankhobare-sst-metaxpytorch-hackathon.hf.space").sync() as env:
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

**Evaluate locally:**
```bash
python inference.py --mode interactive                # human-play one episode
python inference.py --mode test --episodes 10         # test the environment logic
```

**Train:**
Run the `notebooks/FinalTrainingScript.ipynb` in Colab or Kaggle.

---
**License**: Apache-2.0
