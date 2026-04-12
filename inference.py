"""
inference.py — Smart Traffic Signal Controller
Meta x Scalar Open Envs Hackathon

This script demonstrates a simple rule-based agent interacting
with the TrafficSignalEnvironment across all 3 tasks.

The agent reads cars_waiting and gives more green time
to roads with more cars — a greedy heuristic strategy.
"""

import json
import os
import requests
from typing import List, Optional
from openai import OpenAI

# Environment variables (required by OpenEnv spec)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "traffic_signal_env"
ENV_URL = "https://shubham37r-traffic-signal-env.hf.space"

# OpenAI client — uses validator's injected LLM proxy URL and API key
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "dummy-key"
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def get_action_fallback(cars_waiting):
    """Rule-based fallback: give more green time to roads with more cars."""
    total_cars = sum(cars_waiting)
    if total_cars == 0:
        return [30, 30, 30, 30]
    durations = [int((cars / total_cars) * 120) for cars in cars_waiting]
    durations = [max(10, d) for d in durations]
    return durations


def get_action_from_llm(obs):
    """Use LLM to decide green durations based on current traffic state."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a traffic signal controller. "
                        "Given the number of cars waiting at 4 roads and the current phase, "
                        "return green light durations as a JSON array of exactly 4 integers, "
                        "each between 10 and 60 seconds. "
                        "Give more time to roads with more cars. "
                        "Return ONLY the JSON array, nothing else. Example: [30, 20, 40, 10]"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Cars waiting at each road: {obs['cars_waiting']}. "
                        f"Current phase: {obs['current_phase']}. "
                        f"Total wait time so far: {obs['total_wait_time']}. "
                        "Return green durations as a JSON array of 4 integers."
                    )
                }
            ],
            max_tokens=50,
            temperature=0.0
        )
        text = completion.choices[0].message.content.strip()
        durations = json.loads(text)
        durations = [max(10, min(60, int(d))) for d in durations]
        if len(durations) != 4:
            raise ValueError(f"Expected 4 durations, got {len(durations)}")
        return durations
    except Exception as e:
        print(f"[DEBUG] LLM call failed, using fallback: {e}", flush=True)
        return get_action_fallback(obs["cars_waiting"])


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
    score = 0.01
    success = False
    obs = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        response = requests.post(f"{ENV_URL}/reset", json={
            "episode_id": f"inference-{task_name}",
            "seed": 42,
            "task": task_name
        })
        response.raise_for_status()
        obs = parse_obs(response.json())

        for step in range(1, step_limit + 1):
            durations = get_action_from_llm(obs)
            action_str = str(durations)

            response = requests.post(f"{ENV_URL}/step", json={
                "action": {"green_durations": durations},
                "timeout_s": 30
            })
            response.raise_for_status()
            result = response.json()
            obs = parse_obs(result)
            reward = result["reward"]
            done = result["done"]

            rewards.append(max(float(reward), 0.01))  # ✅ floor at 0.01

            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        total_reward = sum(rewards)
        score = min(max(total_reward / max(step_limit, 1), 0.01), 0.99)  # ✅ strictly (0, 1)
        success = score > 0.01

    except Exception as e:
        print(f"[DEBUG] Exception in task {task_name}: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)