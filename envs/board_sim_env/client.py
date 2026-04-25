# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Board Sim Env Environment Client."""

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import BoardSimAction, BoardSimObservation, BoardState

class BoardSimEnv(EnvClient[BoardSimAction, BoardSimObservation, BoardState]):
    """Client for the Board Sim Env Environment."""

    def _step_payload(self, action: BoardSimAction) -> Dict:
        return {
            "decision": action.decision,
            "coalition_pitch": action.coalition_pitch,
        }

    def _parse_result(self, payload: Dict) -> StepResult[BoardSimObservation]:
        obs_data = payload.get("observation", {})
        observation = BoardSimObservation(
            state=obs_data.get("state", {}),
            event=obs_data.get("event", ""),
            options=obs_data.get("options", []),
            npc_statements=obs_data.get("npc_statements", []),
            round=obs_data.get("round", 1),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> BoardState:
        return BoardState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            state_dict=payload.get("state_dict", {}),
        )
