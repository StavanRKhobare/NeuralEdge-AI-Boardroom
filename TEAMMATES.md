# Teammate Onboarding — OpenEnv Hackathon

Share this folder with your teammates. Anyone running Claude Code against it will get the
hackathon rules, rubric, deadlines, and file templates automatically via the skill at
`.claude/skills/openenv-hackathon/`.

**Read [HANDOFF.md](HANDOFF.md) first — it's the one-stop briefing with the Day-1/Day-2
agenda, all 5 themes, the judging rubric, submission requirements, and file-sharing list.
This file covers setup mechanics and split-of-work.**

## 1. What to share (exact list)

Ship the ENTIRE `OpenEnv Hackathon/` directory. Easiest path: push it to a private GitHub
repo, teammates clone. If sharing via zip/drive, include these files verbatim:

**Required (Claude autoloads these):**
- `CLAUDE.md` — project instructions, injected into every Claude session.
- `.claude/skills/openenv-hackathon/SKILL.md` — the hackathon skill.
- `.claude/skills/openenv-hackathon/reference/*.md` — five reference docs (framework, training
  pipeline, submission checklist, judging playbook, theme selection).

**Required (humans use these):**
- `README.md` — fill placeholders as decisions get made.
- `requirements.txt` — pinned dependencies.
- `.gitignore` — blocks secrets and build artifacts.
- `TEAMMATES.md` — this file.

**Populated during the hackathon:**
- `envs/<env_name>_env/` — created by `python -m openenv.cli init`.
- `notebooks/train_grpo.ipynb` — Colab training script.
- `assets/reward_curve.png`, `assets/before_after.png` — plots from the real training run.

**Do NOT share:**
- `.claude/settings.local.json` — per-user settings, already in `.gitignore`.
- Any `.env`, `HF_TOKEN`, `WANDB_API_KEY`.

## 2. One-time setup (each teammate, before Apr 25 morning)

```bash
# Tools
python --version               # need 3.11+ (project uses 3.12)
docker --version               # need Docker Desktop running for local Space tests
git --version

# Python deps
pip install -r requirements.txt

# HF login (needed for `openenv push`)
hf auth login                  # paste your HF write token

# Optional but recommended: W&B for a public training-run URL in the README
wandb login
```

## 3. How Claude Code picks up the context

When a teammate runs `claude` inside this folder, the harness auto-loads:
1. The user's global `~/.claude/CLAUDE.md` (workflow preferences).
2. This project's `CLAUDE.md` (hackathon rules).
3. Any matching skill in `.claude/skills/` — the `openenv-hackathon` skill triggers on keywords
   like "build", "audit", "deploy", "environment", "README", "submit".

Teammates should simply cd into the project folder and ask Claude normally. Example prompts:
- "Audit the current submission bundle against the checklist."
- "Scaffold an env named `inbox_triage_env` under envs/."
- "Write the Colab training notebook for GRPO with Qwen3-0.6B."
- "Review the README for storytelling clarity."

## 4. Running the OpenEnv CLI

The console script `openenv` may not be on PATH on Windows. Use the module form — it works
everywhere:

```bash
python -m openenv.cli init <name>_env --output-dir envs
python -m openenv.cli validate envs/<name>_env
python -m openenv.cli build envs/<name>_env
python -m openenv.cli push envs/<name>_env --repo-id <user>/<env-name>
```

## 5. Split-of-work suggestion (for a 3-person team)

| Role | Owner | Deliverable |
|---|---|---|
| Environment builder | A | `envs/<name>_env/` + `python -m openenv.cli push` → Space live |
| Training engineer | B | `notebooks/train_grpo.ipynb` + `assets/reward_curve.png` |
| Storyteller | C | README + ≤2-min video or HF blog + Google Form submission |

Mentor rounds (Apr 25 3:30 PM, 8:00 PM; Apr 26 10:00 AM) — all three attend together. Claude
is most useful BEFORE these rounds to prep concrete questions, not during.

## 6. Hard deadlines (paste on a whiteboard)

| Time (IST) | Event |
|---|---|
| Apr 25, 11:30 AM | Hacking begins |
| Apr 25, 1:00 PM  | **Theme + problem statement locked** (self-imposed) |
| Apr 25, 3:30 PM  | Mentor Round 1 |
| Apr 25, 8:00 PM  | Mentor Round 2 |
| Apr 26, 10:00 AM | Mentor Round 3 (final) |
| Apr 26, 12:00 PM | 5-hour submission reminder |
| Apr 26, 3:00 PM  | 2-hour submission reminder |
| **Apr 26, 5:00 PM** | **SUBMISSION DEADLINE — Google Form** |

Post-deadline commits to the HF Space URL are ignored. Whatever is live at 5 PM is judged.
