# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Traffic Signal Env Environment.

The traffic_signal_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List

class TrafficSignalAction(Action):
    """Action for Traffic Signal environment - set green light duration for each road."""
    green_durations: List[int] = Field(
        ..., 
        description="Green light duration in seconds for each road [north, south, east, west]"
    )

class TrafficSignalObservation(Observation):
    """Observation from Traffic Signal environment."""
    cars_waiting: List[int] = Field(
        default=[0, 0, 0, 0], 
        description="Number of cars waiting on each road [north, south, east, west]"
    )
    current_phase: int = Field(
        default=0, 
        description="Current active signal phase (0=NS green, 1=EW green)"
    )
    total_wait_time: float = Field(
        default=0.0, 
        description="Total accumulated wait time across all roads"
    )
    reward: float = Field(default=0.0, description="Reward for this step")
    done: bool = Field(default=False, description="Whether episode is complete")