# Submission Checklist (run this before 5 PM IST Apr 26)

The Google Form on Apr 26 asks for: HF Space URL, Colab notebook link, code repo link, YouTube OR HF blog link. Every URL must ALSO be in the README.

## Tier 1 — disqualifiers (run these first)

- [ ] HF Space is **public** and `https://<user>-<env>.hf.space/health` returns 200.
- [ ] The Space was deployed via `openenv push` (not hand-rolled).
- [ ] The env uses `openenv-core >= 0.2.3`.
- [ ] A Colab notebook link is included; clicking "Run all" on a fresh Colab works end-to-end.
- [ ] The training script connects to the **real environment** (not a static dataset).
- [ ] `reward_curve.png` (or equivalent) exists IN THE REPO (committed, not only in Colab).
- [ ] README links: Space URL, Colab URL, video/blog URL, slide deck URL if any.
- [ ] No HF_TOKEN / WANDB_API_KEY / other secrets committed anywhere.
- [ ] No large video files in the Env HF Space (link to YouTube instead).

## Tier 2 — rubric-boosters

### Environment Innovation (40%)
- [ ] The env is NOT a chess/snake/tic-tac-toe/grid-world clone.
- [ ] One sentence explains what capability gap it targets.
- [ ] A researcher could plausibly write a paper about training on it.
- [ ] Reward is composed (OpenEnv Rubrics) not monolithic.

### Storytelling & Presentation (30%)
- [ ] README reads in 3–5 minutes for a non-technical reviewer.
- [ ] Video is ≤ 2 minutes AND embeds/links from README.
- [ ] README has 4 sections: Problem / Environment / Results / Why it matters.
- [ ] Plots have captions explaining what the reviewer is looking at.
- [ ] At least one before/after comparison (text or visual) of agent behavior.

### Showing Improvement in Rewards (20%)
- [ ] Reward curve shows a visible upward trend.
- [ ] Baseline (random or untrained) is plotted ON THE SAME AXES as the trained run.
- [ ] Training ran long enough that the curve has real signal (not 10 steps).
- [ ] W&B public run link is in the README (or plots are committed as real PNGs).
- [ ] Axes labeled: x = "training step" or "episode", y = "reward" or "loss", with units if applicable.

### Reward & Training Pipeline (10%)
- [ ] Reward is hard to game — a random agent cannot score well.
- [ ] Pipeline is reproducible: `pip install -r requirements.txt && jupyter run notebooks/train_grpo.ipynb` works.
- [ ] Uses TRL `GRPOTrainer` with `environment_factory=` (or justified alternative).
- [ ] `SUPPORTS_CONCURRENT_SESSIONS=True` and `max_concurrent_envs ≥ generation_batch_size`.

## Tier 3 — engineering hygiene

- [ ] Client never imports from `server/` (verified by grep).
- [ ] No reserved MCP tool names (`reset`, `step`, `state`, `close`).
- [ ] `openenv.yaml` is the current v0.2.3 format (spec_version: 1, name, type, runtime, app, port).
- [ ] `requirements.txt` pins major versions.
- [ ] No `__pycache__`, `.venv`, `dist/`, `.env` in the repo.
- [ ] LICENSE file present (recommend Apache-2.0 or MIT).

## README template (paste and fill)

```markdown
# <Env Name> — OpenEnv Hackathon Submission

> 1-sentence hook: what capability does this environment teach?

## Links
- **HF Space (the environment)**: https://<user>-<env>.hf.space
- **Colab (training notebook)**: https://colab.research.google.com/drive/...
- **Code repo**: https://github.com/<user>/<repo>
- **Video (≤2 min)**: https://youtu.be/...
- **Blog**: https://huggingface.co/blog/<user>/<slug>
- **W&B training run**: https://wandb.ai/<user>/<project>/runs/...

## Problem
What capability gap does this target? Why is the current state of LLMs insufficient here? (2–3 sentences.)

## Environment
- **Theme**: Multi-Agent / Long-Horizon / World Modeling / Self-Improvement / Wild Card
- **Agent observes**: …
- **Agent acts by**: …
- **Reward signal**: …
- **Episode ends when**: …

## Quick start
```bash
pip install "my-env @ git+https://huggingface.co/spaces/<user>/<env>"
```
```python
from my_env import MyEnv
from my_env.models import MyAction
with MyEnv(base_url="https://<user>-<env>.hf.space").sync() as env:
    print(env.reset())
    print(env.step(MyAction(move="alpha")))
```

## Results

![Reward curve](assets/reward_curve.png)
*Mean group reward over training steps. Qwen3-0.6B trained with GRPO for 500 steps.*

![Before vs after](assets/before_after.png)
*Mean episode reward (n=50) before and after training. Error bars = 1σ.*

| Metric | Baseline (random) | Untrained Qwen3-0.6B | Trained Qwen3-0.6B |
|---|---|---|---|
| Mean reward | 0.04 | 0.12 | 0.78 |
| Success rate | 4% | 12% | 78% |

## Training recipe
- Model: Qwen/Qwen3-0.6B
- Algorithm: GRPO (TRL v1.0+)
- Compute: 1× T4 on Colab
- Training time: ~30 min
- Episodes: 500

See [notebooks/train_grpo.ipynb](notebooks/train_grpo.ipynb) for the full pipeline.

## Why this matters
Who benefits from an LLM trained on this? What can the resulting agent do that an untrained one cannot? (2–3 sentences.)

## Team
<names, colleges, contact>
```

## Video (≤2 min) — storyboard template

| 0:00–0:15 | Hook — show an LLM failing at the task |
| 0:15–0:45 | Explain the environment in one sentence; show the agent's observation/action |
| 0:45–1:15 | Show the reward curve going up |
| 1:15–1:45 | Show the trained agent succeeding at the task |
| 1:45–2:00 | Call to action — "try it at <HF Space URL>" |

Record on OBS / Loom; upload unlisted to YouTube; paste URL in README.

## Final commit discipline

```bash
git status                    # confirm no secrets / artifacts
git add README.md assets/ notebooks/ envs/
git commit -m "final submission: <env-name>"
git push origin main
# don't touch the HF Space URL after the deadline
```

Then fill the Google Form with the 4 URLs. The README URL is fine as the "code repo link".
