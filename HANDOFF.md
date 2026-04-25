# Morning-of Briefing — OpenEnv Hackathon (Apr 25–26, 2026)

One stop for every fact about the hackathon. Read this the morning of Apr 25 before
heading to Scaler. Consolidates the two source docs (`Themes & Judging Criteria.docx`
and `Meta Hackathon D-DAY.pptx`) plus the skill at `.claude/skills/openenv-hackathon/`.

---

## 1. At-a-glance

- **Event**: Meta PyTorch × Hugging Face OpenEnv Hackathon — India finale.
- **Where**: Scaler School of Technology, Bangalore.
- **When**: Apr 25 (build day) + Apr 26 (submission day).
- **Submission deadline**: **Apr 26, 5:00 PM IST** (Google Form). Commits/changes to
  the HF Space after this time are NOT considered. Whatever is live at 5 PM is judged.
- **Team cap**: one submission per team. If you have multiple ideas, pick the best one.

## 2. Day-1 agenda (Apr 25, Saturday)

| Time (IST) | What | Where |
|---|---|---|
| 7:00 – 10:30 AM | Registration & Arrival | Registration Desk, Scaler Campus |
| 8:00 – 9:15 AM  | Breakfast | Food Zones |
| 10:00 – 10:15 AM | Opening Ceremony | Main Stage |
| 10:15 – 10:30 AM | Problem Themes Overview & Briefing | Main Stage |
| 10:30 – 11:00 AM | Address by Meta Team | Main Stage |
| 11:00 – 11:30 AM | Move to Build Zones | All Classrooms |
| **11:30 AM** | **Hacking begins** | All Classrooms |
| ~1:00 PM (self-imposed) | **Theme + problem statement LOCKED** | Our classroom |
| 1:00 PM | Lunch | Food Zones |
| **3:30 – 4:30 PM** | **Mentor Round 1** | Classrooms |
| 5:00 – 5:30 PM | Talk + High Tea | Main Stage |
| 8:00 – 10:00 PM | Dinner | Food Zones |
| **~9:30 PM** | **Mentor Round 2** | Classrooms |
| 2:00 AM | Midnight snacks | Food Zones |

## 3. Day-2 agenda (Apr 26, Sunday)

| Time (IST) | What | Where |
|---|---|---|
| 8:00 AM | Breakfast | Food Zones |
| **10:00 AM – 12:00 PM** | **Mentor Round 3 (FINAL)** | Classrooms |
| 12:00 PM | ⏰ 5-hour submission reminder | Classrooms |
| 2:00 PM | Lunch | Food Zones |
| 3:00 PM | ⏰ 2-hour submission reminder | Classrooms |
| 3:30 – 4:30 PM | Final build push | Classrooms |
| **🏁 5:00 PM** | **SUBMISSION DEADLINE — Google Form closes** | — |
| 5:15 PM | Closing Remarks | Main Stage |
| 5:30 – 8:00 PM | Open Networking | Near Main Stage |
| 8:00 PM | Event concludes | Near Main Stage |

## 4. The 5 themes (pick one)

| # | Theme | What it teaches the LLM | Example environments |
|---|---|---|---|
| 1 | **Multi-Agent Interactions** | Cooperation, competition, negotiation, coalition formation. Theory-of-mind reasoning. Model others' beliefs in partially observable settings. | Market simulations, compute-allocation negotiations, collaborative puzzle worlds, mixed coop/competitive games. |
| 2 | **(Super) Long-Horizon Planning & Instruction Following** | Decompose goals, track state across long trajectories, recover from early mistakes, handle sparse/delayed rewards — beyond context-window limits. | Research-planning simulators, large-codebase refactoring, strategic resource management, logistics, 300-instruction-scatter tasks. |
| 3.1 | **World Modeling — Professional Tasks** | Maintain internal state, update beliefs from outcomes, orchestrate multi-step workflows using real tools/APIs. No shortcuts. | Dynamic browser/API ecosystems, enterprise apps, scientific workflows (papers → code → experiments), tool-discovery. |
| 3.2 | **World Modeling — Personalized Tasks** | Handle realistic personal delegation: messages, conflicts, scheduling, shopping. | Exec-assistant meeting planner, dinner/drive planning, tough email replies. |
| 4 | **Self-Improvement** | Generate new challenges, escalate difficulty, self-play, adaptive curricula. Recursive skill amplification. | Self-play negotiation arenas, auto-generated math/proofs, evolving coding competitions, adaptive RL curricula. |
| 5 | **Wild Card — Impress Us** | Anything outside the above that meaningfully trains an LLM capability. | — (judges explicitly said they WILL reward out-of-box). |

**Rules on theme**:
- Round-1 problem is NOT required — pick whatever best fits.
- Judges have seen a lot of chess, snake, tic-tac-toe, and grid-world clones. Don't.
- Pick a problem that genuinely excites the team — "that energy comes through in the pitch".

Theme-by-theme shortcut candidates live at [.claude/skills/openenv-hackathon/reference/05-theme-selection.md](.claude/skills/openenv-hackathon/reference/05-theme-selection.md).

## 5. Judging rubric (memorize these weights)

| Weight | Criterion | What judges are checking |
|---|---|---|
| **40%** | **Environment Innovation** | Is the env novel, creative, genuinely challenging? Does it test agent behavior in a way that hasn't been done before? Could a researcher write a paper on training against it? |
| **30%** | **Storytelling & Presentation** | Can you clearly explain the problem, the env, what the agent learned? Is the demo engaging for a non-technical audience? README readable in 3–5 minutes. |
| **20%** | **Showing Improvement in Rewards** | Observable evidence of training progress: reward curves, metrics, before/after, baseline vs. trained on the same axes. |
| **10%** | **Reward & Training Pipeline** | Is the reward logic coherent and hard to game? Does the pipeline produce real improvement in trained-agent behavior? |

**Innovation + Storytelling is 70% of the score.** The docx states explicitly:
> A messy but ambitious environment with real training evidence beats a polished but
> boring one.

## 6. Minimum submission requirements (non-negotiable)

Submissions missing ANY of these are "at a serious disadvantage". The Google Form asks for:

1. **Hugging Face Space URL** — the env, deployed via `python -m openenv.cli push`. Must be PUBLIC and runnable.
2. **Colab notebook link** — training script using Unsloth or HF TRL. Judges re-run it.
3. **Code repository link** — GitHub or HF Hub repo. Every file included.
4. **YouTube video URL OR Hugging Face blog post URL** — the story. Video ≤2 minutes. A slide deck is also an acceptable writeup format.
5. **README in the repo** — links all of the above, plus any extras (W&B runs, slides). README IS the judge's entry point.

Additional rules:
- Do **NOT** put large video files inside the Env HF Space — use a URL reference.
- Every extra material (W&B, slides, blog, video) must be linked FROM the README.

## 7. What makes a submission stand out (from the docx)

From "OpenEnv Hackathon — What Judges Look For":

- **Pick ambitious, original problem**. Ask: "Does this teach the LLM something it currently can't do well? Could someone write a paper about training on this?"
- **Design a reward that teaches**: rich/informative (not 0/1 at the end), captures something hard-to-measure cleverly, uses OpenEnv's Rubric system (composable > monolithic), hard to game.
- **Show real training, end to end**: the loop connects to the env (not a static dataset), trains long enough that curves mean something, baseline vs. trained on the same axes.
- **Readable plots**: label both axes + units; save as `.png`/`.jpg` and commit to the repo (don't leave only in a deleted Colab cell or expired W&B run); embed in README with a one-line caption; overlay comparisons on shared axes.
- **Tell a story, not an API doc**: Problem → Environment → Results → Why does it matter. A reviewer should read it in 3–5 min and WANT to try it.
- **Engineering table stakes**: OpenEnv `Environment`/`MCPEnvironment` base class, client/server separation (client never imports server internals), Gym-style API, valid `openenv.yaml`, no reserved MCP tool names (`reset`, `step`, `state`, `close`).

## 8. Files to share with teammates

Push the ENTIRE `OpenEnv Hackathon/` directory (easiest: private GitHub repo, they clone).
If sharing via zip / Drive, include these files verbatim:

**Context for humans** (read these first):
- [HANDOFF.md](HANDOFF.md) — this file. One-stop briefing.
- [README.md](README.md) — judge-facing template, fill placeholders as decisions get made.
- [TEAMMATES.md](TEAMMATES.md) — setup steps, CLI commands, split-of-work suggestion.
- [CLAUDE.md](CLAUDE.md) — project rules, loaded automatically by Claude Code.

**Context for Claude Code** (auto-loaded when teammates run `claude` in this folder):
- [.claude/skills/openenv-hackathon/SKILL.md](.claude/skills/openenv-hackathon/SKILL.md) — the hackathon skill.
- [.claude/skills/openenv-hackathon/reference/01-openenv-framework.md](.claude/skills/openenv-hackathon/reference/01-openenv-framework.md) — env anatomy, file templates, `openenv.yaml`, push workflow.
- [.claude/skills/openenv-hackathon/reference/02-training-pipeline.md](.claude/skills/openenv-hackathon/reference/02-training-pipeline.md) — TRL-GRPO Colab recipe.
- [.claude/skills/openenv-hackathon/reference/03-submission-checklist.md](.claude/skills/openenv-hackathon/reference/03-submission-checklist.md) — final Apr 26 audit list.
- [.claude/skills/openenv-hackathon/reference/04-judging-rubric-playbook.md](.claude/skills/openenv-hackathon/reference/04-judging-rubric-playbook.md) — tactics per criterion.
- [.claude/skills/openenv-hackathon/reference/05-theme-selection.md](.claude/skills/openenv-hackathon/reference/05-theme-selection.md) — theme fit + 60-min ideation protocol.

**Scaffolding for the build**:
- `requirements.txt` — pinned deps.
- `.gitignore` — blocks secrets.
- `envs/.gitkeep`, `notebooks/.gitkeep`, `assets/.gitkeep` — directory layout.

**Do NOT share**:
- `.claude/settings.local.json` — per-user Claude settings.
- Any `.env`, `HF_TOKEN`, `WANDB_API_KEY`.
- The two source docs from `Downloads/` — superseded by this HANDOFF.md.

## 9. Pre-hackathon checklist (each teammate, before Apr 25 morning)

```bash
# Tools
python --version       # need 3.11+ (project uses 3.12.7)
docker --version       # need Docker Desktop running for local Space tests
git --version

# Python deps
pip install -r requirements.txt

# Hugging Face (required for openenv push)
hf auth login          # paste a WRITE-scoped HF token

# W&B (optional, gives judges a shareable run URL — highly recommended)
wandb login

# Sanity check: verify the OpenEnv CLI works
python -m openenv.cli --help
```

**Accounts to have ready**:
- Hugging Face (write token).
- GitHub (public repo for the code link).
- Google Colab (free T4 is enough for Qwen3-0.6B; Pro helps for 1.7B).
- Weights & Biases (optional).
- YouTube channel (for ≤2-min video) OR HF blog posting enabled.

## 10. Split-of-work suggestion (3-person team)

| Role | Deliverable | Key files |
|---|---|---|
| **Environment builder** | `envs/<name>_env/` scaffolded, filled, pushed to HF Space | `envs/<name>_env/models.py`, `environment.py`, `app.py`, `client.py`, `openenv.yaml` |
| **Training engineer** | Colab notebook that actually trains + committed plots | `notebooks/train_grpo.ipynb`, `assets/reward_curve.png`, `assets/before_after.png` |
| **Storyteller** | README filled, video/blog recorded, Google Form submitted | `README.md`, YouTube URL / HF blog URL |

All three attend every mentor round together. Claude is most useful BEFORE mentor
rounds (prep concrete questions), not during.

## 11. Hard rules (do not violate)

1. **OpenEnv v0.2.3** — never downgrade or use pre-0.2 APIs.
2. **Training must use real env**, not a static dataset — TRL `GRPOTrainer` with `environment_factory=`.
3. **HF Space must be public** and discoverable.
4. **No secrets committed** — `HF_TOKEN`, `WANDB_API_KEY`, `.env` all in `.gitignore`.
5. **No commits after Apr 26, 5:00 PM IST** — URL is frozen at deadline.
6. **OpenEnv CLI on Windows**: use `python -m openenv.cli <subcommand>`, NOT bare `openenv`.
7. **No reserved MCP tool names**: `reset`, `step`, `state`, `close`.
