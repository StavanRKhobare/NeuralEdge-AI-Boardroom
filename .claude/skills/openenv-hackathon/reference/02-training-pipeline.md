# Training Pipeline Reference (TRL + Unsloth, Colab-ready)

The hackathon rubric is explicit: **reward-curve evidence is 20%** and **pipeline coherence is 10%** — so the training script must actually run end-to-end and produce plots. This file is the runnable recipe.

## 1. Why GRPO (not PPO / DPO / SFT)

- GRPO is what the official TRL ↔ OpenEnv integration is built around.
- No separate reward model required — the environment IS the reward.
- Works with small models (Qwen3-0.6B trains on free Colab T4).
- Supports multi-turn tool-calling loops natively via `environment_factory=...`.

## 2. Colab notebook skeleton

Put the whole thing in `notebooks/train_grpo.ipynb`. Cells below:

### Cell 1 — installs

```python
!pip install -q --upgrade \
    "openenv-core>=0.2.3" \
    "trl>=1.0" \
    "transformers>=4.56" \
    "accelerate" \
    "peft" \
    "datasets" \
    "wandb" \
    "unsloth"  # optional; huge speedup on T4

# install your environment client
!pip install -q "my-env @ git+https://huggingface.co/spaces/<user>/my_env_env"
```

### Cell 2 — auth

```python
import os
from huggingface_hub import login
from google.colab import userdata  # if on Colab

login(token=userdata.get("HF_TOKEN"))
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

**NEVER hardcode tokens in the notebook.** Use Colab Secrets (lock icon in left sidebar) or env vars.

### Cell 3 — environment wrapper

```python
from my_env import MyEnv
from my_env.models import MyAction

ENV_URL = "https://<user>-my-env-env.hf.space"

class MyToolEnv:
    """Wrapper that exposes env methods as tool-callable functions for TRL."""

    def __init__(self):
        self.client = MyEnv(base_url=ENV_URL)
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        result = self.client.reset()
        self.reward = 0.0
        self.done = False
        return result.observation.text

    def pick(self, choice: str) -> str:
        """Make a choice in the environment.

        Args:
            choice: One of 'alpha', 'bravo', 'charlie'.

        Returns:
            Feedback message from the environment.
        """
        if self.done:
            raise ValueError("Episode is over.")
        result = self.client.step(MyAction(move=choice))
        self.reward = result.reward
        self.done = result.done
        return result.observation.text
```

**Rules for the wrapper class:**
- `__init__` takes no args other than `self`.
- `reset(**kwargs)` receives dataset columns; returns str | None.
- Every public method (not `_prefixed`) becomes a tool. Give them **specific names** (`guess`, `move`, `buy`, NOT `step`/`action`) — the model uses names to learn tool use.
- Tool methods need docstrings with `Args:` / `Returns:` — TRL generates the tool schema from these.
- Store reward/done on `self` — the reward function reads them later.
- Raise `ValueError("...")` when the episode should end — TRL feeds the message back to the model as a tool response.

### Cell 4 — reward function

```python
def reward_func(environments, **kwargs) -> list[float]:
    """Called once per group after rollout. Returns one reward per env instance."""
    return [env.reward for env in environments]
```

Guidance from the TRL OpenEnv docs:
- **Binary (1.0 / 0.0) rewards often beat shaped rewards** for GRPO — relative ranking within the group matters more than absolute values.
- **Score outcomes, not paths** — let the env judge success; don't check for specific action sequences.
- **Sanity-test with a random policy before training** — if a random agent scores as high as a capable one, the reward is broken.

### Cell 5 — dataset

```python
from datasets import Dataset

system_prompt = """You are an agent interacting with the 'pick' environment.
You have one tool: pick(choice). Call it with 'alpha', 'bravo', or 'charlie'.
Only one choice is correct per episode. Use feedback from the environment to learn."""

n = 500   # episodes per epoch
dataset = Dataset.from_dict({
    "prompt": [[{"role": "user", "content": system_prompt}]] * n
})
```

For multi-env training, add an `"env"` column and route in `reset(**kwargs)` — see the TRL multi_env.py example.

### Cell 6 — trainer

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir="./grpo_my_env",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,      # effective batch = 8
    num_generations=4,                  # group size for GRPO
    max_completion_length=1024,         # TOTAL tokens across multi-turn — raise for long episodes
    use_vllm=True,
    vllm_mode="colocate",               # single-GPU Colab
    learning_rate=1e-6,
    chat_template_kwargs={"enable_thinking": False},
    log_completions=True,
    report_to=["wandb"],
    run_name="grpo-my-env-v1",
    logging_steps=1,
    save_steps=50,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    reward_funcs=reward_func,
    args=config,
    environment_factory=MyToolEnv,     # pass the CLASS, not an instance
)

trainer.train()
```

### Cell 7 — Unsloth speedup (optional, ~2× faster on T4)

Swap the model loader before constructing the trainer:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-0.6B",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=32, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,                # pass the unsloth tokenizer
    ... # same as before
)
```

### Cell 8 — save plots as `.png` (REQUIRED for judging)

```python
import matplotlib.pyplot as plt
import pandas as pd

# trainer.state.log_history is a list of dicts logged during training
log = pd.DataFrame(trainer.state.log_history)

# Reward curve (raw + smoothed)
fig, ax = plt.subplots(figsize=(8, 4))
if "reward" in log.columns:
    ax.plot(log["step"], log["reward"], alpha=0.3, label="reward (raw)")
    ax.plot(log["step"], log["reward"].rolling(20, min_periods=1).mean(), label="reward (smoothed)")
ax.set_xlabel("training step")
ax.set_ylabel("mean reward per group")
ax.set_title("GRPO training — reward over time")
ax.legend()
plt.tight_layout()
plt.savefig("assets/reward_curve.png", dpi=150)

# Loss curve
fig, ax = plt.subplots(figsize=(8, 4))
if "loss" in log.columns:
    ax.plot(log["step"], log["loss"], label="policy loss")
ax.set_xlabel("training step")
ax.set_ylabel("loss")
ax.set_title("GRPO training — loss over time")
ax.legend()
plt.tight_layout()
plt.savefig("assets/loss_curve.png", dpi=150)
```

Then commit both PNGs to the repo — judges MUST see them in the README.

### Cell 9 — baseline-vs-trained comparison (scores high on rubric)

```python
import numpy as np

def eval_model(model, n_episodes=50):
    env = MyToolEnv()
    rewards = []
    for _ in range(n_episodes):
        env.reset()
        # ... run model for up to max_turns, collecting env.reward
        rewards.append(env.reward)
    return np.mean(rewards), np.std(rewards)

base_mean, base_std = eval_model("Qwen/Qwen3-0.6B")
trained_mean, trained_std = eval_model(trainer.model)

print(f"baseline:  {base_mean:.3f} ± {base_std:.3f}")
print(f"trained:   {trained_mean:.3f} ± {trained_std:.3f}")

# plot on same axes
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["baseline", "trained"], [base_mean, trained_mean],
       yerr=[base_std, trained_std], capsize=6)
ax.set_ylabel("mean episode reward (n=50)")
ax.set_title("Before vs. after GRPO training")
plt.tight_layout()
plt.savefig("assets/before_after.png", dpi=150)
```

## 3. Concurrency — DO NOT SKIP

TRL opens one WebSocket per generation. With `gradient_accumulation_steps=8` × `per_device_train_batch_size=1` × `num_generations=4` = 32 concurrent sessions. The Space must allow this.

On the environment side:
```python
# server/<name>_environment.py
class MyEnvironment(Environment[...]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
```
```python
# server/app.py
app = create_app(..., max_concurrent_envs=64)
```

On the Space side (via HF Space → Settings → Variables):
```
MAX_CONCURRENT_ENVS=64
WORKERS=2
```

**Always duplicate the Space to your own account before training.** Public shared Spaces get rate-limited.

## 4. Colab T4 reality check

- **Qwen3-0.6B** trains in ~30 min for a 500-episode Wordle-style task.
- **Qwen3-1.7B** needs Colab Pro (A100) for a comparable run; T4 will OOM.
- **Gradient accumulation** > 8 on T4 with Unsloth + LoRA.
- **vLLM colocate mode** reclaims ~3 GB by sharing weights between gen and training.
- Save checkpoints every 50 steps so a Colab disconnect doesn't nuke progress.

## 5. What failure looks like, and how to recover fast

| Symptom | Cause | Fix |
|---|---|---|
| Training hangs after batch 1 | `SUPPORTS_CONCURRENT_SESSIONS=False` | Set True; redeploy. |
| Reward flat at 0 the whole run | Reward function returns wrong key, or tool method never called | Log `env.reward` + `env.done` per episode in Cell 3. |
| Reward saturates at 1.0 instantly | Reward is game-able (model finds shortcut) | Tighten env; add adversarial check; switch to binary terminal reward. |
| W&B run disappears | Colab session timeout + no local save | Set `save_steps=50` and download `output_dir` as a tarball. |
| `max_completion_length` exceeded errors | Episodes too long for the budget | Raise to 2048 or 4096; OR cap env turn count. |
| OOM on T4 | Batch × group × seq too large | Lower `num_generations` to 2, or switch to Unsloth 4-bit. |

## 6. Official reference implementations to clone from

- **Echo** (simplest): [github.com/huggingface/trl/blob/main/examples/scripts/openenv/echo.py](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/echo.py)
- **Wordle** (multi-turn, exception-based episode end): [github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb)
- **Multi-env** (routing between 2 envs in one run): [github.com/huggingface/trl/blob/main/examples/scripts/openenv/multi_env.py](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/multi_env.py)

When unsure about any pattern, open the Wordle notebook — it is the canonical example.
