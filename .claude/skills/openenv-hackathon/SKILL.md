---
name: openenv-hackathon
description: Use this skill for ANY work on the Meta PyTorch × Hugging Face OpenEnv Hackathon submission (India finale, Scaler Bangalore, Apr 25–26 2026). Trigger whenever the user says "build", "audit", "review", "check", "deploy", or references the environment, training script, README, HF Space, Colab notebook, submission, or judging criteria. The submission must use OpenEnv (latest release, v0.2.3), be hosted on a Hugging Face Space, include a TRL- or Unsloth-based training script (ideally a Colab notebook), show reward/loss plots from a real training run, and ship with a README that links a <2-min YouTube video or a mini-blog on HF. Judging is weighted 40% Environment Innovation / 30% Storytelling / 20% Reward Improvement Evidence / 10% Reward & Training Pipeline.
---

# OpenEnv Hackathon — Build & Audit Skill

## 1. Hackathon Calendar (hard deadlines)

| When | What | Where |
|---|---|---|
| **Apr 25, 11:30 AM IST** | Hacking begins | Scaler School of Technology, Bangalore — classrooms |
| **Apr 25, 3:30 PM IST** | Mentor Round 1 | Classrooms |
| **Apr 25, 8:00 PM IST** | Mentor Round 2 | Classrooms |
| **Apr 26, 10 AM – 12 PM IST** | Mentor Round 3 (final) | Classrooms |
| **Apr 26, 12:00 PM IST** | 5-hour submission reminder | Classrooms |
| **Apr 26, 3:00 PM IST** | 2-hour submission reminder | Classrooms |
| **Apr 26, 5:00 PM IST** | **SUBMISSION DEADLINE** — Google Form | — |
| **Apr 26, 5:15 PM IST** | Closing remarks | Main Stage |
| **Apr 26, 8:00 PM IST** | Event concludes | Near Main Stage |

**Rule**: Changes or commits to the HF Space URL after the deadline are ignored. Whatever is live at 5 PM IST on Apr 26 is what gets judged.

## 2. Submission bundle (non-negotiable)

A submission missing ANY of these is "at a serious disadvantage". The Google Form on Apr 26 asks for:

1. **Hugging Face Space URL** — the environment, deployed via `openenv push`. Must be PUBLIC.
2. **Colab notebook link** — training script that judges can re-run.
3. **Code repository link** — GitHub (or HF Hub repo). Include every file.
4. **YouTube video URL OR Hugging Face blog post URL** — the story. Video ≤ 2 minutes.
5. **README in the repo** — must link the Space, the Colab, the video/blog, and any slides. README is the judge's entry point.

Every URL also lives in the README. No large video files inside the Env HF Space — reference by URL.

## 3. The five themes (pick one; Theme 5 is the wildcard)

| # | Theme | Teaches the LLM to… | Example problems |
|---|---|---|---|
| 1 | **Multi-Agent Interactions** | Cooperate, compete, negotiate, form coalitions; model others' beliefs (theory-of-mind) in partially observable settings. | Market simulations, compute-allocation negotiations, collaborative puzzle worlds, mixed coop/competitive games. |
| 2 | **(Super) Long-Horizon Planning & Instruction Following** | Decompose goals, track state across long trajectories beyond context limits, recover from early mistakes, handle sparse/delayed rewards. | Research-planning simulators, large-codebase refactoring, strategic resource management, logistics optimization, 300-instruction-scatter tasks. |
| 3.1 | **World Modeling — Professional Tasks** | Maintain internal state, update beliefs from outcomes, orchestrate multi-step workflows with real APIs/tools (no shortcuts). | Dynamic browser/API ecosystems, enterprise apps, scientific workflows (papers → code → experiments), tool-discovery benchmarks. |
| 3.2 | **World Modeling — Personalized Tasks** | Handle realistic personal delegation: messages, conflicts, scheduling, shopping. | Exec-assistant meeting planner, dinner/drive planning, tough email replies. |
| 4 | **Self-Improvement** | Generate new challenges, escalate difficulty, self-play, adaptive curricula — recursive skill amplification. | Self-play negotiation arenas, auto-generated math/proofs, evolving coding competitions, adaptive RL curricula. |
| 5 | **Wild Card — Impress Us** | Anything outside the boxes above that meaningfully trains an LLM capability. | — |

**Theme selection rule**: Round-1 problem is NOT required. Pick what best fits one of the themes AND excites the team — judges can tell when energy is real.

See [reference/05-theme-selection.md](reference/05-theme-selection.md) for a 60-minute ideation protocol and per-theme shortcut candidates.

## 4. Judging rubric (memorize these weights)

| Weight | Criterion | What it really means | How I bias toward this when building |
|---|---|---|---|
| **40%** | Environment Innovation | Novel, creative, genuinely challenging. Tests agent behavior in a way that hasn't been done. | Push originality over polish. Avoid chess/snake/tic-tac-toe/grid clones. Ask: "Could a researcher write a paper on training against this?" |
| **30%** | Storytelling & Presentation | Clear problem statement; engaging demo; non-technical audience can follow. | README reads in 3–5 min. Video ≤ 2 min. Before/after agent behavior on screen. |
| **20%** | Showing Improvement in Rewards | Observable evidence: reward curves, metrics, before/after, baseline vs. trained on the same axes. | Train long enough that curves mean something. Commit `.png` plots to the repo. Caption each plot in the README. |
| **10%** | Reward & Training Pipeline | Reward logic is coherent, hard to game; pipeline produces real improvement in trained-agent behavior. | Compose Rubrics thoughtfully. Dense signal > 0/1-at-end. Test reward manually (random baseline should NOT score high). |

Innovation + Storytelling is **70%** of the score. A messy but ambitious env with real training evidence beats a polished but boring one — the rules state this explicitly.

See [reference/04-judging-rubric-playbook.md](reference/04-judging-rubric-playbook.md) for per-criterion tactics and anti-patterns.

## 5. Tech stack (what to build with)

| Layer | Pick | Why |
|---|---|---|
| Environment framework | **OpenEnv v0.2.3** (`pip install openenv-core`) | Mandatory. Use `Environment` or `MCPEnvironment` base class, Gym-style API. |
| Training framework | **HF TRL `GRPOTrainer`** with `environment_factory=` | Official OpenEnv ↔ TRL integration (docs: [huggingface.co/docs/trl/openenv](https://huggingface.co/docs/trl/openenv)). |
| Speed/memory | **Unsloth** (optional, strongly recommended for Colab T4) | 2× speed, up to 70% memory cut; supports GRPO/GSPO/DPO on free Colab. |
| Base model | Start with **Qwen3-0.6B** or **Qwen3-1.7B** | Used in official examples; small enough for Colab, big enough to show learning. |
| Hosting | **Hugging Face Space** (via `openenv push`) | Mandatory. Space must be public and runnable. |
| Notebook | **Google Colab** | Judges need to re-run it. Use `uv run` or a pip install cell that works in fresh Colab. |
| Writeup | **HF blog post** OR **YouTube ≤ 2 min** | Mandatory. Link from README. |

See [reference/01-openenv-framework.md](reference/01-openenv-framework.md) for the full directory layout, file templates, openenv.yaml fields, and push workflow.
See [reference/02-training-pipeline.md](reference/02-training-pipeline.md) for a runnable TRL-GRPO training recipe tuned for Colab T4.

## 6. What I do when the user says "build"

The scope is inferred from the stated target. In order:

1. **Confirm theme + problem statement** in one sentence before writing code. If ambiguous, ask. Don't silently assume.
2. **Name the env** snake_case (e.g., `dinner_negotiator_env`). Create it via `openenv init <name>_env --output-dir envs` — do not hand-roll the scaffold.
3. **Fill the four files** in this order: `models.py` (Action / Observation / State dataclasses) → `server/<env>_environment.py` (core logic: `reset`, `step`, optional `state`) → `server/app.py` (FastAPI wiring via `create_app` or `create_server`) → `client.py` (thin `EnvClient` subclass).
4. **Update `openenv.yaml`** with `spec_version: 1`, `name`, `type`, `runtime`, `app`, `port`. No reserved MCP tool names (`reset`, `step`, `state`, `close`).
5. **Set `SUPPORTS_CONCURRENT_SESSIONS = True`** on the Environment class AND pass `max_concurrent_envs=64` (or ≥ `generation_batch_size`) to `create_app`. Without this, training will fail with WebSocket capacity errors.
6. **Design the reward** with OpenEnv Rubrics when possible: composable, dense, hard to game. Test with a random-policy baseline BEFORE writing the training script — the baseline should score noticeably worse than a competent agent.
7. **Smoke-test locally** with Docker (`openenv init` produces a Dockerfile — use it). Verify `reset()` / `step()` work and reward is sensible over ~20 random episodes.
8. **Deploy**: `openenv push --repo-id <user>/<env-name>`. Confirm the Space is live at `https://<user>-<env-name>.hf.space/health`.
9. **Write the training script** as a Colab notebook using `GRPOTrainer(environment_factory=MyEnv, ...)`. Use Qwen3-0.6B unless the user specifies otherwise. Log to W&B or at minimum save `.png` plots.
10. **Run the training** to produce real reward/loss curves. Commit the plots as `assets/reward_curve.png`, `assets/loss_curve.png` in the repo.
11. **Write the README** — see [reference/03-submission-checklist.md](reference/03-submission-checklist.md) for the required-sections list and tone.

At each step, I report what was done in one line and move on.

## 7. What I do when the user says "audit"

An audit is read-only until the user asks me to fix. I check the submission bundle against the rubric and report gaps as a prioritized list. My audit always covers, in this order:

1. **Submission completeness** — all 5 bundle items present and linked from the README? (See [reference/03-submission-checklist.md](reference/03-submission-checklist.md).)
2. **OpenEnv compliance** — uses v0.2.3; `Environment` or `MCPEnvironment` base; Gym-style `reset/step/state`; valid `openenv.yaml`; no reserved tool names; client/server separation (client never imports server internals).
3. **HF Space health** — `openenv push` succeeded; `/health` returns 200; `/docs` loads; `SUPPORTS_CONCURRENT_SESSIONS` and `max_concurrent_envs` set for training.
4. **Reward signal** — dense (not just 0/1 at terminal), hard to game. Flag any reward that a random agent could exploit for points without solving the task.
5. **Training evidence** — reward curve exists, has >1 clearly-visible step of improvement, is committed as a real image file (not only in a deleted Colab cell / W&B run), baseline is on the same axes as the trained run.
6. **README storytelling** — problem / environment / results / why-it-matters sections present; readable in 3–5 min; plots captioned.
7. **Repo hygiene** — no leaked secrets (HF_TOKEN, WANDB_API_KEY), no large video files in the HF Space (reference by URL), no build artifacts/venvs committed.

I report findings as: **[SEVERITY] finding — fix**. Severities: `CRITICAL` (submission disqualifier), `HIGH` (likely to cost >10 rubric pts), `MED` (polish), `LOW` (nice-to-have).

## 8. Hard rules (things I will refuse to do or strongly push back on)

- **Don't submit without a real training run.** "Training script exists" is NOT the bar. The bar is "connects to the environment, agent learns, plots prove it." If the user asks to skip training, I push back once and then flag it as a CRITICAL audit finding.
- **Don't clone chess / snake / tic-tac-toe / grid-world.** Judges have seen them. If the user proposes one, I recommend an angle that makes it genuinely novel (e.g., a meta-learning wrapper, a compositional reward, a new modality).
- **Don't use `WidthType.PERCENTAGE`, reserved MCP tool names, or `Percentage` width in docx tables** if we write docs.
- **Don't commit `.env` / `HF_TOKEN` / `WANDB_API_KEY`.** Use `huggingface_hub.login()` in Colab, read from env vars elsewhere.
- **Don't amend commits after submission.** The URL is frozen at deadline — a post-deadline commit is equivalent to submitting a different artifact.
- **Don't bloat the HF Space with video files.** Link to YouTube/HF blog instead.
- **Don't mock the environment in training.** If `environment_factory` is set, the training loop MUST hit the real Space (or a local Docker of it) — a static dataset disqualifies criterion #3 (20%).

## 9. Directory structure this skill assumes

```
OpenEnv Hackathon/
├── .claude/skills/openenv-hackathon/
│   ├── SKILL.md                      # this file
│   └── reference/
│       ├── 01-openenv-framework.md   # env anatomy, API, openenv.yaml, push
│       ├── 02-training-pipeline.md   # TRL-GRPO Colab recipe
│       ├── 03-submission-checklist.md
│       ├── 04-judging-rubric-playbook.md
│       └── 05-theme-selection.md     # theme fit analysis + ideation protocol
├── envs/
│   └── <env_name>_env/               # the OpenEnv env — scaffolded by `openenv init`
├── notebooks/
│   └── train_grpo.ipynb              # the Colab judges will re-run
├── assets/
│   ├── reward_curve.png
│   └── loss_curve.png
├── README.md                         # the judge's entry point — links EVERYTHING
└── requirements.txt
```

## 10. External skills / tools the team needs

**Claude Code skills to use during the hackathon:**

- `anthropic-skills:pptx` — if the submission includes a slide deck (allowed as a writeup format).
- `frontend-design` — if building a demo web UI for the environment or a landing page.
- `python-performance-optimization` — profile training if reward curves plateau due to env-step latency (common on HF Spaces).
- `review` — self-review the diff before final push.
- `security-review` — final pass for leaked tokens / keys before making the repo public.

**External tools & accounts required (set up BEFORE Apr 25 morning):**

- Python 3.11+ and Docker Desktop installed locally.
- Hugging Face account + write token (`hf auth login`).
- `pip install openenv-core>=0.2.3 trl unsloth wandb`.
- Google Colab account (free T4 is enough for Qwen3-0.6B; Pro is better for 1.7B).
- Weights & Biases account (optional but highly recommended — gives judges a shareable run URL).
- GitHub account (public repo for the code link).
- YouTube channel (for the ≤2-min video) OR HF account with blog posting enabled.

**Technical competencies the team should have reviewed:**

- OpenEnv's Gymnasium-style API (`reset`, `step`, `state`) — see [reference/01-openenv-framework.md](reference/01-openenv-framework.md).
- GRPO algorithm intuition (group relative policy optimization — compares completions within a group; relative ranking > absolute values).
- Basic LoRA/PEFT for Unsloth fine-tuning on Colab.
- FastAPI basics (openenv init gives you the server scaffold, but you may need to extend it).
- Docker basics for local Space testing.

## 11. Reference files

Each of these is loaded only when relevant — keep SKILL.md lean.

- **[reference/01-openenv-framework.md](reference/01-openenv-framework.md)** — Directory layout; models.py / environment.py / app.py / client.py templates; full openenv.yaml; `openenv init` / `openenv push` CLI; concurrency setup; common pitfalls.
- **[reference/02-training-pipeline.md](reference/02-training-pipeline.md)** — Complete TRL-GRPO Colab notebook recipe; Unsloth wiring; reward-function patterns; multi-environment training; plot generation; W&B logging.
- **[reference/03-submission-checklist.md](reference/03-submission-checklist.md)** — The final Apr 26 audit list; README template; sample commit structure; pre-deadline smoke tests.
- **[reference/04-judging-rubric-playbook.md](reference/04-judging-rubric-playbook.md)** — Tactics per criterion; what scores high on Innovation (40%); storytelling heuristics; training-evidence standards; anti-patterns.
- **[reference/05-theme-selection.md](reference/05-theme-selection.md)** — Theme-by-theme fit analysis; shortcut candidates per theme; decision framework for the first 90 minutes on Apr 25.

---

**Source documents indexed by this skill:**
- `C:\Users\vitta\Downloads\[External] Apr '26 OpenEnv Hackathon Themes & Judging Criteria.docx` — authoritative rules.
- `C:\Users\vitta\Downloads\Meta Hackathon D-DAY.pptx` — Day-1/Day-2 event schedule.
- [huggingface.co/docs/trl/openenv](https://huggingface.co/docs/trl/openenv) — TRL ↔ OpenEnv integration.
- [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) — framework source, v0.2.3.
- [github.com/huggingface/openenv-course](https://github.com/huggingface/openenv-course) — 5-module tutorial.
