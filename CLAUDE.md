# NeuralEdge AI Boardroom — Project Context

Submission for the Meta × PyTorch × HuggingFace OpenEnv Hackathon (India finale, Scaler Bangalore, Apr 25–26 2026), Theme 1 — Multi-Agent Interactions.

## Stack

- OpenEnv `v0.2.3` (do not downgrade or use pre-0.2 APIs)
- Qwen3-0.6B base + Unsloth 4-bit LoRA (r=32, α=64)
- GRPO-style group-relative advantages over a custom training loop on top of `TRL>=0.12,<2.0`
- FastAPI server hosted on HF Spaces (Docker)

## Canonical entry points

- `README.md` — judge entry point (problem → environment → reward → baseline → training → next steps)
- `MECHANICS.md` — full math reference for the environment
- `inference.py` — interactive / eval / compare runner
- `Training.py` — canonical training script (notebook is generated from this)
- `notebooks/train_grpo_v2.ipynb` — Colab notebook
- `envs/board_sim_env/server/board_sim_env_environment.py` — environment, NPC simulation, reward
- `scripts/random_baseline.py` — 200-episode random-policy baseline

## Judging weights

40% Environment Innovation · 30% Storytelling · 20% Reward Improvement Evidence · 10% Reward & Training Pipeline. The README is structured to surface the first two weights immediately.
