"""
inference.py — Smart Traffic Signal Controller
Meta x Scalar Open Envs Hackathon

This script demonstrates a simple rule-based agent interacting
with the TrafficSignalEnvironment across all 3 tasks.

The agent reads cars_waiting and gives more green time
to roads with more cars — a greedy heuristic strategy.
"""

import requests

BASE_URL = "http://localhost:8000"


def get_action(cars_waiting):
    """
    Rule-based agent:
    Give more green time to roads with more waiting cars.
    Total budget = 120 seconds split across 4 roads.
    """
    total_cars = sum(cars_waiting)
    if total_cars == 0:
        return [30, 30, 30, 30]  # Equal split if no cars

    durations = [
        int((cars / total_cars) * 120) for cars in cars_waiting
    ]

    # Make sure each road gets at least 10 seconds
    durations = [max(10, d) for d in durations]
    return durations


def run_task(task_name):
    """Run one full episode for the given task."""
    print(f"\n{'='*40}")
    print(f"Running task: {task_name.upper()}")
    print(f"{'='*40}")

    # Reset environment with chosen task
    response = requests.post(f"{BASE_URL}/reset", json={
        "episode_id": f"inference-{task_name}",
        "seed": 42,
        "task": task_name
    })
    obs = response.json()["observation"]
    print(f"Initial state: cars_waiting={obs['cars_waiting']}")

    total_reward = 0.0
    step = 0

    max_steps = {"easy": 10, "medium": 30, "hard": 50}
    step_limit = max_steps[task_name]
    
    while step < step_limit:
        # Agent decides action based on current observation
        durations = get_action(obs["cars_waiting"])

        # Send action to environment
        response = requests.post(f"{BASE_URL}/step", json={
            "action": {"green_durations": durations},
            "timeout_s": 30
        })
        result = response.json()
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]

        total_reward += reward
        step += 1

        print(f"Step {step:2d} | cars={obs['cars_waiting']} | "
              f"reward={reward:.2f} | total_wait={obs['total_wait_time']:.0f}")

    print(f"\nTask '{task_name}' finished in {step} steps.")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final total wait time: {obs['total_wait_time']:.0f}")


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)

    print("\n All tasks complete!")