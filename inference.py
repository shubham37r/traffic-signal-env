"""
inference.py — Smart Traffic Signal Controller
Meta x Scalar Open Envs Hackathon

This script demonstrates a simple rule-based agent interacting
with the TrafficSignalEnvironment across all 3 tasks.

The agent reads cars_waiting and gives more green time
to roads with more cars — a greedy heuristic strategy.
"""

import os
import requests
from openai import OpenAI

# Environment variables (required by OpenEnv spec)
API_BASE_URL = os.getenv("API_BASE_URL", "https://shubham37r-traffic-signal-env.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# OpenAI client configured via environment variables
client = OpenAI()


def get_action(cars_waiting):
    """Rule-based agent: give more green time to roads with more cars."""
    total_cars = sum(cars_waiting)
    if total_cars == 0:
        return [30, 30, 30, 30]
    durations = [int((cars / total_cars) * 120) for cars in cars_waiting]
    durations = [max(10, d) for d in durations]
    return durations


def run_task(task_name):
    """Run one full episode for the given task."""
    max_steps = {"easy": 10, "medium": 30, "hard": 50}
    step_limit = max_steps[task_name]

    # START log
    print(f"START task={task_name} max_steps={step_limit}")

    response = requests.post(f"{API_BASE_URL}/reset", json={
        "episode_id": f"inference-{task_name}",
        "seed": 42,
        "task": task_name
    })
    obs = response.json()["observation"]

    total_reward = 0.0

    for step in range(1, step_limit + 1):
        durations = get_action(obs["cars_waiting"])

        response = requests.post(f"{API_BASE_URL}/step", json={
            "action": {"green_durations": durations},
            "timeout_s": 30
        })
        result = response.json()
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        total_reward += reward

        # STEP log (structured format)
        print(f"STEP step={step} cars={obs['cars_waiting']} reward={reward:.2f} total_wait={obs['total_wait_time']:.0f}")

        if done:
            break

    # END log
    print(f"END task={task_name} total_reward={total_reward:.2f} final_wait={obs['total_wait_time']:.0f}")


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)