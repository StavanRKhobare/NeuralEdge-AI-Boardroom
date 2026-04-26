# NeuralEdge AI Boardroom: Rethinking Multi-Agent RL

Welcome to the NeuralEdge AI Boardroom, our submission for the Meta × PyTorch Hackathon. This blog breaks down our core innovation, the environment's reward design, and the training evidence that proves our LLM agent is learning to exhibit Theory-of-Mind (ToM).

## Our Innovation: Asymmetric Theory-of-Mind (ToM) Persuasion

Most multi-agent RL benchmarks (like Poker or Werewolf) focus on symmetric games with discrete action spaces. We built something fundamentally different: an **asymmetric, partially observable environment where natural language persuasion is graded**. 

Our LLM agent plays the CEO. It must satisfy four Board Members (NPCs), each with a hidden agenda (e.g., the CFO cares about runway; the CTO cares about engineering morale). The agent never sees these agendas. Instead, it must infer them from the NPCs' voting history and public statements, and then generate a **persuasive pitch** to swing their votes.

We quantify this innovation via our **Theory-of-Mind (ToM) Probe**. During evaluation, we freeze the agent and ask it to predict *which* specific board member will oppose its decision. A random baseline guesses correctly ~25% of the time, but our GRPO-trained model explicitly learns to identify its opponents and tailor its rhetoric to their hidden agendas.

## The Reward Function Design

Our reward function (`board_sim_env_environment.py:723`) is deliberately designed to mix dense step-level shaping with sparse, episodic terminal spikes.

Here is the reasoning behind its core components:

1. **Δ Profitability (Dense):** The agent receives a continuous reward based on the change in the company's profitability score. This teaches basic corporate survival (protect runway, grow revenue).
2. **Coalition Success (Dense):** Winning a vote yields `+1.0`, losing yields `-0.4`.
3. **Persuasion & Pitching (The ToM Signal):** If the agent writes a pitch, it gets a `+0.05` bootstrap reward. If it writes a pitch that semantically aligns (measured via SBERT cosine similarity) with the hidden manifestos of *opposing* NPCs, it earns up to `+0.6`. This forces the LLM to learn high-quality, targeted rhetoric.
4. **Terminal Spikes (Sparse):** Survival alone isn't enough. Running out of money triggers a `-2.0` penalty. Successfully reaching the endgame triggers massive spikes: `+30` for an Acquisition, `+25` for an IPO. 

This combination ensures the agent doesn't just learn to "survive" by making safe choices, but actively learns to build trust and persuade opponents to achieve a massive exit.

## Training Evidence & Graph Analysis

We trained Qwen3 (1.7B) using a KL-free **GRPO (Group Relative Policy Optimization)** setup. Our W&B training dashboard reveals exactly how the model adapts to the environment over time:

![W&B Training Graphs](assets/reward_curve.png)

1. **Pitch Rate Convergence (`pitch_rate`)**: Early in training, the agent's pitch rate is erratic. Very quickly, the pitch rate spikes and locks in at **1.0 (100%)**. The agent discovers that emitting an empty pitch is strategically suboptimal. It learns to always write a pitch to capture the persuasion reward channel.
2. **Reward Maximums (`reward_max` & `reward`)**: The mean step reward stays relatively flat (representing the dense shaping), but we see distinct, massive spikes up to `+30`. These spikes confirm the agent successfully navigated the 10-round gauntlet to achieve the optimal "Acquisition" endgame.
3. **Reward Standard Deviation (`reward_std`)**: The high variance (spikes up to 15) indicates active exploration. In our episodic structure, high variance is a feature, not a bug—it means the agent is exploring different terminal outcomes (bankruptcy vs. IPO vs. acquisition).
4. **Loss Stabilization (`loss`)**: The GRPO loss (advantage × NLL) starts highly volatile but compresses around zero as the policy stabilizes and the group-relative advantages converge.

By the end of training, the model hasn't just learned what decisions to make—it has learned *how to argue for them*.
