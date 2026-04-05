---
title: Smart Traffic Signal Controller
emoji: 🚦
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - traffic
  - fastapi
---

# 🚦 Smart Traffic Signal Controller

An OpenEnv reinforcement learning environment where an agent controls
green light durations at a 4-way intersection to minimize total car
waiting time. Built for the Meta x Scalar Open Envs Hackathon.

## Environment Overview

The agent observes cars waiting at 4 roads (North, South, East, West)
and decides how long each road gets a green light. The goal is to
clear traffic as efficiently as possible.

### Observation Space
| Field | Type | Description |
|---|---|---|
| `cars_waiting` | List[int] | Cars waiting per road [N, S, E, W] |
| `current_phase` | int | Active signal phase (0=NS green, 1=EW green) |
| `total_wait_time` | float | Accumulated wait time across all roads |

### Action Space
| Field | Type | Description |
|---|---|---|
| `green_durations` | List[int] | Green light seconds per road [N, S, E, W] |

### Reward
- **Easy/Hard**: +1.0 if under threshold, -1.0 if over
- **Medium**: `-total_waiting / 40.0` (minimize cars waiting)

---

## Tasks

### 🟢 Easy — Keep Traffic Flowing
- **Goal**: Keep total waiting cars below 20 at every step
- **Steps**: 10
- **Traffic**: Light, predictable arrivals

### 🟡 Medium — Minimize Wait Time
- **Goal**: Minimize total accumulated wait time
- **Steps**: 30
- **Traffic**: Moderate, random arrivals

### 🔴 Hard — Handle Traffic Spikes
- **Goal**: Keep average cars per road under 5
- **Steps**: 50
- **Traffic**: Random spikes on random roads (30% chance per step)

---

## Quick Start

### Connect to the live environment
```python
import requests

BASE_URL = "https://shubham37r-traffic-signal-env.hf.space"

# Reset environment (choose task: easy / medium / hard)
response = requests.post(f"{BASE_URL}/reset", json={
    "episode_id": "my-episode",
    "seed": 42,
    "task": "easy"
})
obs = response.json()["observation"]
print(obs)

# Take a step
response = requests.post(f"{BASE_URL}/step", json={
    "action": {"green_durations": [30, 30, 60, 60]},
    "timeout_s": 30
})
result = response.json()
print(result)
```

### Run the inference agent
```bash
git clone https://huggingface.co/spaces/shubham37r/traffic-signal-env
cd traffic-signal-env
pip install requests
python inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset environment, returns first observation |
| `/step` | POST | Send action, returns next observation + reward |
| `/state` | GET | Get current environment state |
| `/schema` | GET | Get action/observation schemas |
| `/docs` | GET | Interactive Swagger UI |
| `/web` | GET | Web interface |

---

## Project Structure
```
traffic_signal_env/
├── models.py                        # Action + Observation models
├── openenv.yaml                     # OpenEnv spec
├── README.md                        # This file
├── inference.py                     # Rule-based agent demo
└── server/
    ├── app.py                       # FastAPI application
    ├── Dockerfile                   # Container definition
    ├── requirements.txt             # Dependencies
    └── traffic_signal_env_environment.py  # Core logic + 3 tasks
```

---

## Local Development
```bash
# Install dependencies
pip install openenv-core fastapi uvicorn pydantic

# Run server
python -m traffic_signal_env.server.app

# Test via browser
open http://localhost:8000/docs

# Run agent
python inference.py
```

---

Built with ❤️ using FastAPI + OpenEnv by shubham37r