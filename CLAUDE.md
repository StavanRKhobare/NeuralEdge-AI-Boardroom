# OpenEnv Hackathon — Project Context

This directory is the Meta PyTorch × Hugging Face OpenEnv Hackathon India finale submission
(Scaler Bangalore, Apr 25–26 2026). Deadline: **Apr 26, 5:00 PM IST** (Google Form).

## Rules for Claude working in this repo

1. **Use the `openenv-hackathon` skill** at `.claude/skills/openenv-hackathon/SKILL.md` for any
   task involving the environment, training, README, deployment, or submission. It has the
   hackathon calendar, judging rubric, file templates, and hard rules. For the human-readable
   one-stop briefing, see `HANDOFF.md` at the repo root.
2. **OpenEnv version is 0.2.3.** Never downgrade or use pre-0.2 APIs.
3. **Training framework is TRL `GRPOTrainer`** with `environment_factory=`. Base model defaults
   to `Qwen/Qwen3-0.6B` unless the team says otherwise.
4. **Hosting is HF Spaces via `python -m openenv.cli push`**. The Space MUST be public.
5. **Judging weights**: 40% Innovation, 30% Storytelling, 20% Reward Improvement Evidence,
   10% Reward & Training Pipeline. Bias every decision toward the first two.
6. **Never commit secrets** (`HF_TOKEN`, `WANDB_API_KEY`, `.env`). `.gitignore` covers them.
7. **Never amend commits after Apr 26 5:00 PM IST** — the URL is frozen at deadline.

## Local environment (already verified Apr 24, 2026)
- Python 3.12.7
- openenv-core 0.2.3, trl 1.2.0, transformers 5.4.0, torch 2.5.1+cu121
- Docker 29.1.5, git 2.52
- OpenEnv CLI runs as: `python -m openenv.cli <subcommand>` (NOT bare `openenv`).

## Directory layout
```
envs/<env_name>_env/     # scaffolded via `python -m openenv.cli init <name>_env --output-dir envs`
notebooks/train_grpo.ipynb
assets/                  # reward_curve.png, before_after.png — must be committed
README.md                # judge entry point
requirements.txt
.claude/skills/openenv-hackathon/  # the skill + reference docs
```

## What's still TODO (as of Apr 24, 2026)
- [ ] Theme lock-in (team decides Apr 25, 1:00 PM IST)
- [ ] Environment name + `openenv.cli init`
- [ ] Fill `envs/<name>_env/` files (models → environment → app → client)
- [ ] `openenv push` to HF Space
- [ ] Write `notebooks/train_grpo.ipynb`
- [ ] Run training long enough for real reward curve
- [ ] Commit `assets/*.png`
- [ ] Fill README TBDs
- [ ] Record ≤2-min video OR write HF blog
- [ ] Submit Google Form by Apr 26 5:00 PM IST
