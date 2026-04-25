# OpenEnv Framework Reference

OpenEnv v0.2.3 (released Mar 28 2026). Install: `pip install "openenv-core>=0.2.3"`.

## 1. The 3 APIs

All OpenEnv environments expose the Gymnasium-style trio:

| Method | Purpose | Returns |
|---|---|---|
| `reset(seed=None, episode_id=None, **kwargs)` | Start a new episode. | Initial `Observation` |
| `step(action, timeout_s=None, **kwargs)` | Apply one `Action`. | `Observation`, reward, done |
| `state()` | Metadata snapshot (episode_id, step_count, etc.) | `State` |

The client side mirrors this with both async and sync wrappers:

```python
# async (preferred)
async with EchoEnv(base_url="https://...hf.space") as env:
    obs = await env.reset()
    obs = await env.step(EchoAction(message="hi"))

# sync
with EchoEnv(base_url="https://...hf.space").sync() as env:
    obs = env.reset()
    obs = env.step(EchoAction(message="hi"))
```

## 2. Two environment archetypes

**Typed step/reset (default)** — you define explicit `Action`/`Observation` dataclasses and implement `step(action)`. Use when actions are structured and enumerable (moves, choices, form submissions).

**MCP tool environment** — extend `MCPEnvironment`; the environment exposes named tools (e.g., `search`, `open_file`, `send_email`). Use when the agent should discover and call a set of tools. TRL's `environment_factory` loop automatically exposes every public method as an MCP-style tool.

## 3. Directory layout (what `openenv init <name>_env` produces)

```
<name>_env/
├── openenv.yaml               # manifest
├── pyproject.toml             # package metadata + deps
├── README.md
├── Dockerfile                 # container for the HF Space
├── requirements.txt
├── client.py                  # class <Name>Env(EnvClient)
├── models.py                  # Action, Observation, State dataclasses
└── server/
    ├── __init__.py
    ├── <name>_environment.py  # class <Name>Environment(Environment[...])
    └── app.py                 # FastAPI wiring via create_app(...)
```

Scaffold the whole thing with:

```bash
openenv init my_env_env --output-dir envs
```

Do NOT hand-roll the directory — the scaffold format changes across versions.

## 4. `openenv.yaml` — full example

```yaml
spec_version: 1
name: dinner_negotiator_env
type: environment                # or mcp_environment
version: "0.1.0"
description: >
  Multi-agent dinner-planning negotiation where the LLM must reconcile
  dietary restrictions, budget, and scheduling conflicts across 3 family
  members with hidden preferences.

runtime:
  python: "3.11"
  dependencies:
    - openenv-core>=0.2.3
    - fastapi
    - pydantic

app:
  module: server.app
  factory: app                    # FastAPI ASGI app object
  host: 0.0.0.0
  port: 8000

max_concurrent_envs: 64           # ≥ generation_batch_size for TRL training
```

Fields `spec_version`, `name`, `type`, `runtime`, `app`, `port` are required.

## 5. Template — `models.py`

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MyAction:
    """Structured action from the agent."""
    move: str
    target: Optional[str] = None

@dataclass
class MyObservation:
    """What the agent sees after each step."""
    text: str
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)

@dataclass
class MyState:
    """Episode metadata (returned by state())."""
    episode_id: str
    step_count: int
    target: str
    remaining_turns: int
```

## 6. Template — `server/<name>_environment.py`

```python
import random, uuid
from typing import Optional

try:
    from openenv.core import Environment
except ImportError:
    from openenv_core import Environment  # dual-import pattern for Docker

from ..models import MyAction, MyObservation, MyState

class MyEnvironment(Environment[MyAction, MyObservation, MyState]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True   # REQUIRED for TRL training

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._reset_state()

    def _reset_state(self):
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._remaining = self.max_turns
        self._target = random.choice(["alpha", "bravo", "charlie"])

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              **kwargs) -> MyObservation:
        if seed is not None:
            random.seed(seed)
        self._reset_state()
        if episode_id:
            self._episode_id = episode_id
        return MyObservation(
            text=f"New episode. Pick one of: alpha | bravo | charlie. {self._remaining} turns left.",
        )

    def step(self, action: MyAction, timeout_s: Optional[float] = None,
             **kwargs) -> MyObservation:
        self._step_count += 1
        self._remaining -= 1

        correct = action.move == self._target
        done = correct or self._remaining <= 0
        reward = 1.0 if correct else (-0.1 if done else 0.0)

        if correct:
            text = f"Correct! Target was {self._target}."
        elif done:
            text = f"Out of turns. Target was {self._target}."
        else:
            text = f"Wrong. {self._remaining} turns left."

        return MyObservation(text=text, reward=reward, done=done,
                             info={"step": self._step_count})

    def state(self) -> MyState:
        return MyState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            target=self._target,
            remaining_turns=self._remaining,
        )
```

**Key rules:**
- `SUPPORTS_CONCURRENT_SESSIONS = True` — MUST be set for TRL training; otherwise only 1 WebSocket connects.
- Use `try/except` dual import — Docker runs from a different module root than the repo.
- Never use `reset`, `step`, `state`, `close` as MCP tool names — they collide with the base API.

## 7. Template — `server/app.py`

```python
try:
    from openenv.server import create_app
except ImportError:
    from openenv_core.server import create_app

from .my_environment import MyEnvironment
from ..models import MyAction, MyObservation

app = create_app(
    environment_factory=lambda: MyEnvironment(max_turns=10),
    action_type=MyAction,
    observation_type=MyObservation,
    max_concurrent_envs=64,     # match or exceed generation_batch_size
)
```

## 8. Template — `client.py`

```python
try:
    from openenv.client import EnvClient
except ImportError:
    from openenv_core.client import EnvClient

from .models import MyAction, MyObservation, MyState

class MyEnv(EnvClient[MyAction, MyObservation, MyState]):
    ACTION_TYPE = MyAction
    OBSERVATION_TYPE = MyObservation
    STATE_TYPE = MyState
```

Thin wrapper — the base class handles WebSocket, serialization, async/sync.

## 9. Local testing (before push)

```bash
# from repo root
pip install -e envs/my_env_env
python -m uvicorn envs.my_env_env.server.app:app --host 0.0.0.0 --port 8001

# in another shell
python -c "
from envs.my_env_env.client import MyEnv
from envs.my_env_env.models import MyAction
with MyEnv(base_url='http://0.0.0.0:8001').sync() as env:
    print(env.reset())
    print(env.step(MyAction(move='alpha')))
"
```

Docker test (mirrors what the HF Space will run):

```bash
docker build -t my_env envs/my_env_env
docker run -d -p 8001:8000 my_env
curl http://localhost:8001/health    # expect 200 {"status": "ok"}
```

## 10. Push to HF Spaces

```bash
cd envs/my_env_env
huggingface-cli login              # one-time
openenv push --repo-id <user>/my_env_env
# add --private to stage privately, then flip to public before submission
```

After push, verify:
- `https://<user>-my-env-env.hf.space/health` → 200
- `https://<user>-my-env-env.hf.space/docs` → FastAPI Swagger UI
- `https://<user>-my-env-env.hf.space/web` → web UI (if enabled)

## 11. Environment variables the Space respects

| Var | Default | Use |
|---|---|---|
| `WORKERS` | 4 | Uvicorn worker processes |
| `PORT` | 8000 | Internal port |
| `HOST` | 0.0.0.0 | Bind address |
| `MAX_CONCURRENT_ENVS` | 100 | WebSocket sessions cap |
| `ENABLE_WEB_INTERFACE` | auto | Toggle `/web` UI |

Set via HF Space → Settings → Variables & Secrets.

## 12. Common pitfalls (cross-referenced from the official skill)

- **Forgetting `SUPPORTS_CONCURRENT_SESSIONS`** → training hangs after first batch.
- **Reserved MCP tool name** (`reset`/`step`/`state`/`close`) → silent conflict with base API.
- **Client importing server internals** → import cycle at container start. Client must ONLY import from `models.py`.
- **Committing build artifacts** (`__pycache__`, `.venv`, `dist/`) to the Space → slow push, bloated Space.
- **Using `openenv push` without first testing Docker locally** → broken Space, debug-via-logs-only loop.
- **Missing `xml:space="preserve"`** on docx edits (not relevant to env, noted only if generating docs).
