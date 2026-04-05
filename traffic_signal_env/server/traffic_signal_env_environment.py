# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Traffic Signal Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import os
from uuid import uuid4
from typing import List, Optional
import random
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TrafficSignalAction, TrafficSignalObservation
except ImportError:
    from models import TrafficSignalAction, TrafficSignalObservation


TASKS = {
    "easy": {
        "max_steps": 10,
        "goal": "Keep total waiting cars below 20 at every step.",
        "traffic_spike": False,
        "max_cars_start": 5,
        "pass_threshold": 20,
    },
    "medium": {
        "max_steps": 30,
        "goal": "Minimize total wait time over 30 steps.",
        "traffic_spike": False,
        "max_cars_start": 8,
        "pass_threshold": None,
    },
    "hard": {
        "max_steps": 50,
        "goal": "Keep average cars per road under 5 despite random traffic spikes.",
        "traffic_spike": True,
        "max_cars_start": 10,
        "pass_threshold": 5,
    },
}


class TrafficSignalEnvironment(Environment):
    """
    Traffic Signal Controller Environment.
    The agent controls green light durations to minimize
    total car waiting time at a 4-way intersection.
    Supports 3 tasks: easy, medium, hard.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._cars_waiting: List[int] = [0, 0, 0, 0]
        self._total_wait_time: float = 0.0
        self._current_phase: int = 0
        self._task: str = "easy"
        self._max_steps: int = TASKS["easy"]["max_steps"]

    def reset(self, task: Optional[str] = None) -> TrafficSignalObservation:
        env_task = os.environ.get("TRAFFIC_TASK", "easy")
        if task and task in TASKS:
            self._task = task
        elif env_task in TASKS:
            self._task = env_task
        else:
            self._task = "easy"
            
        self._max_steps = TASKS[self._task]["max_steps"]

        cfg = TASKS[self._task]
        self._max_steps = cfg["max_steps"]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._cars_waiting = [random.randint(2, cfg["max_cars_start"]) for _ in range(4)]
        self._total_wait_time = 0.0
        self._current_phase = 0

        return TrafficSignalObservation(
            cars_waiting=self._cars_waiting,
            current_phase=self._current_phase,
            total_wait_time=self._total_wait_time,
            reward=0.0,
            done=False,
        )

    def step(self, action: TrafficSignalAction) -> TrafficSignalObservation:
        """Execute one step — apply signal timings and simulate traffic."""
        self._state.step_count += 1
        cfg = TASKS[self._task]

        green_durations = action.green_durations

        # Cars pass through based on green duration
        for i, duration in enumerate(green_durations):
            cars_passed = min(self._cars_waiting[i], duration // 10)
            self._cars_waiting[i] = max(0, self._cars_waiting[i] - cars_passed)

        # New arrivals — hard mode adds traffic spikes
        if cfg["traffic_spike"] and random.random() < 0.3:
            spike_road = random.randint(0, 3)
            new_arrivals = [random.randint(0, 3) for _ in range(4)]
            new_arrivals[spike_road] += random.randint(5, 10)
        else:
            new_arrivals = [random.randint(0, 3) for _ in range(4)]

        self._cars_waiting = [self._cars_waiting[i] + new_arrivals[i] for i in range(4)]

        total_waiting = sum(self._cars_waiting)
        self._total_wait_time += total_waiting

        # --- Reward and grading ---
        if self._task == "easy":
            # Reward +1 if under threshold, -1 if over
            passed = total_waiting < cfg["pass_threshold"]
            reward = 1.0 if passed else -1.0

        elif self._task == "medium":
            # Reward = negative wait, normalized
            reward = -total_waiting / 40.0

        else:  # hard
            # Reward based on average cars per road
            avg = total_waiting / 4.0
            reward = 1.0 if avg < cfg["pass_threshold"] else -avg / 10.0

        # Switch phase
        self._current_phase = 1 - self._current_phase

        done = self._state.step_count >= self._max_steps

        # Final grader score at episode end
        grade = None
        if done:
            grade = self._compute_grade(total_waiting)

        return TrafficSignalObservation(
            cars_waiting=self._cars_waiting,
            current_phase=self._current_phase,
            total_wait_time=self._total_wait_time,
            reward=reward,
            done=done,
        )

    def _compute_grade(self, final_total_waiting: int) -> float:
        """Compute final grade score 0.0 to 1.0 based on task."""
        if self._task == "easy":
            # Grade = how far under 20 cars we finished
            score = max(0.0, (20 - final_total_waiting) / 20.0)
        elif self._task == "medium":
            # Grade = inverse of average wait time per step
            avg_wait = self._total_wait_time / self._max_steps
            score = max(0.0, 1.0 - avg_wait / 100.0)
        else:  # hard
            # Grade = how close avg cars per road is to 0
            avg = final_total_waiting / 4.0
            score = max(0.0, 1.0 - avg / 20.0)
        return round(score, 3)

    @property
    def state(self) -> State:
        return self._state