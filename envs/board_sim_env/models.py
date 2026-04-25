# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Action / Observation / State types for the Board-Sim Env."""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State as BaseState
from pydantic import Field


class BoardSimAction(Action):
    """The agent (CEO) picks one of 3 string decisions for the current event.

    Optional `coalition_pitch` is reserved for future reward shaping; v1
    does not consume it but it is accepted to keep the action schema stable.
    """

    decision: str = Field(
        ...,
        description="Exactly one of the strings in the latest observation's `options` list.",
    )
    coalition_pitch: Optional[str] = Field(
        default="",
        description="Optional natural-language argument to the board (unused in v1 reward).",
    )


class BoardSimObservation(Observation):
    """What the agent sees each step.

    `state` excludes NPC hidden agendas (those are private). NPC statements +
    votes shown here are the SAME ones used at vote-resolve time — i.e. the
    environment is deterministic given (seed, round)."""

    state: Dict[str, Any] = Field(..., description="Public startup state metrics + trust + history.")
    event: str = Field(..., description="This round's market-crisis event title + description.")
    options: List[str] = Field(..., description="Three valid decision strings for this round.")
    npc_statements: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="One dict per NPC: {role, statement, vote, confidence}.",
    )
    round: int = Field(..., description="1-indexed round number (1..10).")
<<<<<<< HEAD
    done: bool = Field(default=False, description="Whether the episode is terminal.")
    reward: float = Field(default=0.0, description="Reward from the latest step.")
    event_idx: Optional[int] = Field(default=None, description="Internal index in the EVENTS list.")
=======
>>>>>>> 220bc90 (Initial commit for OpenEnv Hackathon submission)


class BoardState(BaseState):
    """Server-internal state. The `state_dict` mirrors what's visible in
    observations plus internal bookkeeping (history, done_reason)."""

    state_dict: Dict[str, Any] = Field(default_factory=dict)
