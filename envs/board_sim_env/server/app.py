# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Board Sim Env Environment.

The openenv framework's built-in /reset and /step endpoints are stateless
(fresh env per request). We add custom /game/reset and /game/step routes
that use a single persistent GameManager instance so multi-round episodes
work correctly from the frontend.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import BoardSimAction, BoardSimObservation
    from .board_sim_env_environment import BoardSimEnvironment
except (ImportError, ValueError):
    # Direct uvicorn launch from envs/board_sim_env/: package context not available.
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import BoardSimAction, BoardSimObservation  # type: ignore
    from server.board_sim_env_environment import BoardSimEnvironment  # type: ignore

import json
import httpx

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from starlette.middleware.cors import CORSMiddleware


# ── Stateful game manager (single instance, shared across requests) ─
class GameManager:
    """Holds one persistent BoardSimEnvironment so state is preserved
    between /game/reset and /game/step calls."""

    def __init__(self):
        self._env: Optional[BoardSimEnvironment] = None

    def reset(self, seed: int = 42) -> Dict[str, Any]:
        self._env = BoardSimEnvironment()
        obs = self._env.reset(seed=seed)
        return self._obs_to_dict(obs)

    def step(self, decision: str, coalition_pitch: str = '') -> Dict[str, Any]:
        if self._env is None:
            raise RuntimeError("Call /game/reset before /game/step")
        action = BoardSimAction(decision=decision, coalition_pitch=coalition_pitch)
        obs = self._env.step(action)
        return self._obs_to_dict(obs)

    @staticmethod
    def _obs_to_dict(obs: BoardSimObservation) -> Dict[str, Any]:
        return {
            "observation": {
                "state":          obs.state,
                "event":          obs.event,
                "options":        obs.options,
                "npc_statements": obs.npc_statements,
                "round":          obs.round,
            },
            "reward": getattr(obs, "reward", 0.0),
            "done":   getattr(obs, "done", False),
            "info":   {},
        }


_game = GameManager()


# ── Pydantic request models ────────────────────────────────────────
class GameResetRequest(BaseModel):
    seed: int = 42


class GameStepRequest(BaseModel):
    decision: str
    coalition_pitch: str = ""


class QwenDecideRequest(BaseModel):
    """Board observation forwarded from the frontend for Qwen inference."""
    state: Dict[str, Any]
    event: str
    options: List[str]
    npc_statements: List[Dict[str, Any]] = []
    round: int = 1


# ── Greedy fallback (mirrors frontend greedyPick) ──────────────────
_ROLE_WEIGHT = {
    'CEO': 1.5, 'CTO': 1.2, 'CFO': 1.0, 'Investor Rep': 1.3, 'Independent': 0.8,
}

def _greedy_pick(options: List[str], npc_statements: List[Dict[str, Any]]) -> str:
    tally = {opt: 0.0 for opt in options}
    for npc in npc_statements:
        vote = npc.get('vote', '')
        if vote in tally:
            tally[vote] += _ROLE_WEIGHT.get(npc.get('role', ''), 0.8) * float(npc.get('confidence', 0.5))
    return max(tally, key=lambda k: tally[k])


# ── Qwen system prompt ─────────────────────────────────────────────
_QWEN_SYSTEM = (
    "You are the CEO agent in a boardroom simulation. "
    "Given the board state and NPC positions, choose the best strategic decision "
    "and craft a short coalition pitch to win over dissenters. "
    "Always respond with ONLY a valid JSON object in the exact format: "
    '{"decision": "<one of the listed options>", "coalition_pitch": "<1-2 sentence pitch>"}'
    " — no markdown, no explanation, no extra keys."
)


# ── Create the openenv app (for /health, /schema, /ws, etc.) ───────
app = create_app(
    BoardSimEnvironment,
    BoardSimAction,
    BoardSimObservation,
    env_name="board_sim_env",
    max_concurrent_envs=64,
)

# CORS — allow React dev server and any origin in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Stateful routes ────────────────────────────────────────────────
@app.post("/game/reset")
def game_reset(req: GameResetRequest):
    """Reset the persistent game environment and return initial observation."""
    return _game.reset(seed=req.seed)


@app.post("/game/step")
def game_step(req: GameStepRequest):
    """Step the persistent game environment with the given decision."""
    return _game.step(decision=req.decision, coalition_pitch=req.coalition_pitch)


# ── LM Studio Local Server Config ──────────────────────────────────
_LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"


@app.post("/qwen/decide")
async def qwen_decide(req: QwenDecideRequest):
    """
    Call the Qwen model via local LM Studio server.
    Returns {decision, coalition_pitch, source} where source is
    'qwen_lmstudio' on success or 'local_error_fallback' on failure.
    """
    npc_summary = "\n".join(
        f"  - {n.get('role','?')} ({n.get('role','?')}): votes '{n.get('vote','?')}' "
        f"(confidence {n.get('confidence', 0.5):.2f}) — '{n.get('statement','')[:120]}'"
        for n in req.npc_statements
    )
    user_prompt = (
        f"Round: {req.round}\n"
        f"Company state: {json.dumps(req.state)}\n"
        f"Current crisis/event: {req.event}\n"
        f"Available options: {req.options}\n"
        f"Board member positions:\n{npc_summary}\n\n"
        "Your JSON decision:"
    )

    try:
        # payload for OpenAI-compatible local server (LM Studio)
        payload = {
            "model": "qwen", # LM Studio usually ignores this and uses the loaded model
            "messages": [
                {"role": "system", "content": _QWEN_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.1,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(_LM_STUDIO_URL, json=payload)
            
        resp.raise_for_status()
        data = resp.json()
        raw_content = data["choices"][0]["message"]["content"].strip()

        # Handle potential markdown code blocks
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_content:
            raw_content = raw_content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(raw_content)
        decision = str(parsed.get("decision", "")).strip()
        pitch = str(parsed.get("coalition_pitch", "")).strip()

        # Validate decision is one of the legal options
        if decision not in req.options:
            decision = _greedy_pick(req.options, req.npc_statements)

        return {"decision": decision, "coalition_pitch": pitch, "source": "qwen_lmstudio"}

    except Exception as exc:
        # LM Studio not running or model not loaded → greedy fallback
        fallback = _greedy_pick(req.options, req.npc_statements)
        return {
            "decision": fallback,
            "coalition_pitch": "",
            "source": "greedy_fallback",
            "error": str(exc),
        }


# ── Entry point ────────────────────────────────────────────────────
def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
