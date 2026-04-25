# Theme Selection — Decision Framework for the First 90 Minutes on Apr 25

The hackathon explicitly allows picking a NEW problem at the finale — Round-1 entries are not required. The first 90 minutes is the single highest-leverage window of the whole event.

## Decision tree

```
Do we have a concrete problem that clearly fits one of Themes 1–4 or 5?
├── YES → lock it in. Go to OpenEnv scaffold.
└── NO  → run the 60-min ideation below, then decide.
```

## 60-minute ideation protocol

**Minute 0–15 — read the themes aloud to the team.** One person reads, others note any phrase that sparks an idea. Don't filter yet.

**Minute 15–30 — write one-sentence problem statements**, one per sticky note. Format: `"An environment that teaches an LLM to ___ by ___"`. Aim for 10–15 candidates.

**Minute 30–45 — score each on 3 axes (1–5 each):**
- **Novelty** — has a judge seen this before? (5 = never, 1 = clone)
- **Shippability in 30h** — can we deploy this by 5 PM tomorrow? (5 = trivial, 1 = heroic)
- **Reward learnability** — can a 0.6B–1.7B model actually improve on it in 30 min of Colab? (5 = yes, 1 = needs a 70B)

**Minute 45–60 — pick the highest total.** Ties broken by team excitement (the rules explicitly say this).

## Shortcut candidates per theme (research-done for you)

### Theme 1 — Multi-Agent Interactions
- **Hidden-role party negotiator** — 4 LLM "guests" with hidden dietary/budget constraints must agree on a restaurant in ≤5 turns. The agent-under-training is one of them. Reward = Pareto-optimality of the agreement.
- **Compute allocator** — N services bid for shared GPU time under changing priority. Agent-under-training learns to negotiate SLAs.

### Theme 2 — Long-Horizon Planning
- **300-instruction document follower** — a fake product spec has 300 tiny requirements scattered across 50 pages. Agent must produce output that satisfies ≥K of them. Tests durable internal representation.
- **Research-plan simulator** — agent drafts a research plan, gets fake "reviewer feedback" across 10 rounds, must incorporate it.

### Theme 3.1 — World Modeling, Professional
- **Tool-discovery env** — agent is given an undocumented API with 50 endpoints and must figure out how to accomplish a task through experimentation. Reward = success with minimum API calls.
- **Scientific-workflow loop** — paper → extracted hypothesis → pseudo-code → pseudo-experiment result → next paper. Agent learns to iterate.

### Theme 3.2 — World Modeling, Personal
- **Inbox-triage env** — 20 emails arrive; agent must reply-all / reply-one / archive / snooze / delegate. Reward = combined latency + correctness per sender.
- **Calendar conflict resolver** — three colleagues propose 5 meeting times each; agent replies to each with the one that works for everyone.

### Theme 4 — Self-Improvement
- **Proof-difficulty escalator** — agent generates math problems, tries to solve them, gets harder problems when it succeeds. Reward = steady-state difficulty reached.
- **Self-adversarial Wordle** — one agent proposes words, another tries to guess; roles rotate. Both improve.

### Theme 5 — Wild Card
Use sparingly. Only if the idea doesn't map to 1–4 AND you can explain in one sentence why an LLM trained on this is more useful than before. The rules promise rewards for out-of-box ideas — but they also warn submissions "must meaningfully add value to LLM training".

## Lock-in rule

Once the team commits (by 1:00 PM Apr 25), **stop idea-generating**. Every hour spent re-debating the problem is an hour not spent shipping. Write the one-sentence problem statement on a whiteboard. Everything after this point serves THAT sentence.
