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
from typing import List, Optional
from openai import OpenAI

# Environment variables (required by OpenEnv spec)
API_BASE_URL = os.getenv("API_BASE_URL", "https://shubham37r-traffic-signal-env.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key"))

BENCHMARK = "traffic_signal_env"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action(cars_waiting):
    """Rule-based agent: give more green time to roads with more cars."""
    total_cars = sum(cars_waiting)
    if total_cars == 0:
        return [30, 30, 30, 30]
    durations = [int((cars / total_cars) * 120) for cars in cars_waiting]
    durations = [max(10, d) for d in durations]
    return durations


def parse_obs(data):
    """Safely extract observation from response dict."""
    if "observation" in data:
        return data["observation"]
    elif "obs" in data:
        return data["obs"]
    else:
        raise KeyError(f"No observation key found in response: {list(data.keys())}")


def run_task(task_name):
    """Run one full episode for the given task."""
    max_steps = {"easy": 10, "medium": 30, "hard": 50}
    step_limit = max_steps[task_name]

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    obs = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        response = requests.post(f"{API_BASE_URL}/reset", json={
            "episode_id": f"inference-{task_name}",
            "seed": 42,
            "task": task_name
        })
        response.raise_for_status()
        obs = parse_obs(response.json())

        for step in range(1, step_limit + 1):
            durations = get_action(obs["cars_waiting"])
            action_str = str(durations)

            response = requests.post(f"{API_BASE_URL}/step", json={
                "action": {"green_durations": durations},
                "timeout_s": 30
            })
            response.raise_for_status()
            result = response.json()
            obs = parse_obs(result)
            reward = result["reward"]
            done = result["done"]

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        # Score: clamp total_reward to [0, 1]
        total_reward = sum(rewards)
        score = min(max(total_reward / max(step_limit, 1), 0.0), 1.0)
        success = score > 0.0

    except Exception as e:
        print(f"[DEBUG] Exception in task {task_name}: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)