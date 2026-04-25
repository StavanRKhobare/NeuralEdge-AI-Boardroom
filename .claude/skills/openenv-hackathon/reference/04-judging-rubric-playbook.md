# Judging Rubric Playbook — How to score high

Rubric weights, verbatim from the hackathon rules:

| Weight | Criterion |
|---|---|
| **40%** | Environment Innovation — Is the environment novel, creative, or genuinely challenging? Does it meaningfully test agent behavior in a way that hasn't been done before? |
| **30%** | Storytelling & Presentation — Can you clearly explain the problem, the environment, and what the agent learned? Is the demo engaging and easy to follow for a non-technical audience? |
| **20%** | Showing Improvement in Rewards — Is there observable evidence of training progress? Reward curves, before/after behavior, comparison against a baseline. |
| **10%** | Reward & Training Pipeline — Is the reward logic coherent? Does the pipeline produce meaningful improvement in the trained agent's behavior? |

## Innovation (40%) — the biggest lever

**The official rule: "Judges have seen a lot of chess, snake, tic-tac-toe, and grid-world clones."** Do not ship one.

### Questions the rules tell you to ask yourself
1. Does this environment exist to teach an LLM something it currently can't do well?
2. Is the domain underexplored in RL/LLM training?
3. Could a researcher write a paper about training on this?

### High-innovation patterns that match the themes
- **Partially observable negotiation** (Theme 1) where each agent has private info — e.g., dinner planning with hidden allergies/budgets.
- **Tool-discovery benchmarks** (Theme 3.1) where the agent must read API docs at runtime and figure out which tool applies.
- **300-instruction instruction-following** scattered across a long document (Theme 2) — tests selective attention and durable memory.
- **Self-play curriculum generation** (Theme 4) where the env generates harder variants of whatever task the agent is currently solving.
- **Real-personal-delegation** (Theme 3.2) — e.g., the agent receives a realistic Slack-style thread with 3 people proposing 5 meeting times, must pick one and reply to everyone.

### Anti-innovation (avoid)
- Classic games with cosmetic reskin.
- Single-turn QA / classification dressed up as an env.
- Anything where the "environment" is actually just a frozen dataset with a scoring function.
- Reward = string-match against a ground-truth answer (doesn't need an env).

## Storytelling (30%)

### What the README must do
- **Open with a hook in ≤20 words.** "This env teaches an LLM to negotiate dinner plans across 3 people with conflicting dietary restrictions and hidden preferences."
- **Show, don't tell.** Before/after behavior transcript beats prose.
- **Name the audience.** "This matters to anyone building personal-assistant LLMs that handle real delegation."
- **Embed plots inline** with one-line captions.
- **Link everything from the top** — Space, Colab, video, blog, W&B run.

### What the video (≤2 min) must do
- **Open with failure.** Untrained model doing something dumb.
- **Show the env's rules in one visual.** Observation → action → reward diagram.
- **Show the reward curve going up.**
- **Show the trained model succeeding.**
- **End with a URL** the viewer can click.

### Storytelling anti-patterns
- API docs masquerading as a README.
- Pure prose with no images.
- Video that explains the code instead of the capability.
- Demo that needs narration to understand what's happening on screen.

## Reward-improvement evidence (20%)

### Minimum viable evidence
- Reward curve committed as `assets/reward_curve.png` with captioned embed in README.
- Loss curve also helpful (proves the training actually updated weights).
- Baseline on the SAME AXES as the trained run — a single line going up is easy to dismiss.
- Explicit numbers: "baseline 4% success → trained 78% success (n=50)".

### Patterns that separate top-10% from median
- **W&B public run link** in the README → reviewers can dig into any metric.
- **Ablation plot**: trained with reward v1 vs reward v2 vs random baseline, all on one axis.
- **Qualitative transcript**: one full agent trajectory before training, one after — side-by-side.
- **Multiple seeds**: 3 runs with error bars, not 1.

### Traps that score 0 on this criterion
- Reward curve only exists in a deleted W&B run.
- Plot saved only in a Colab cell (disappears when Colab times out).
- Curve flat or noisy with no smoothed trendline.
- No baseline for comparison.
- 10 training steps — "noise, not signal".

## Reward & pipeline coherence (10%)

### What "coherent reward" means
- **Dense informative signal** — not just 0/1 at the terminal state. OR, if 0/1, it's a hard problem where that's appropriate.
- **Composable via OpenEnv Rubrics** — multiple sub-rubrics combined, not one monolithic score.
- **Hard to game** — test by running a random agent; if it scores near the trained agent, reward is broken.

### What "coherent pipeline" means
- `environment_factory=` wired correctly; generation → tool parse → env step → reward → training — all handled by TRL.
- Concurrency configured: `SUPPORTS_CONCURRENT_SESSIONS=True` on env, `max_concurrent_envs ≥ generation_batch_size` on the app.
- Tool methods have docstrings with `Args:` blocks (TRL uses these to build the tool schema).
- Tool names are **specific** (`guess`, `negotiate`, `buy`) — not generic (`step`, `act`).

### Red flags
- Custom rollout loop when `environment_factory` would have worked (the rubric favors the standard pattern).
- Reward hacked by a hard-coded regex against the model's output.
- Training against a mocked env (disqualifies the criterion — must hit the real deployed Space or a local Docker).

## The 70% that's actually under your control

Innovation (40%) + Storytelling (30%) = **70% of the score**, and both are set mostly by Day-1 decisions:

1. **Pick the right problem by noon on Apr 25.** A bad problem with great execution still caps around the median.
2. **Draft the README hook and 2-min video storyboard before you write any env code.** If you can't explain it in one sentence, it's not ambitious enough yet.
3. **Build the smallest viable env first, then iterate on innovation.** It's better to have a shippable boring env + a clear story than a brilliant env you couldn't deploy.
4. **Record the video on Apr 26 morning, not at 4:55 PM.** Leave 90 minutes for recording + upload.
