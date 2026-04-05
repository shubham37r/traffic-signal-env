import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from dqn_agent import DQN, ReplayMemory

# Settings
EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000

# Setup
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQN(state_size, action_size)
memory = ReplayMemory(MEMORY_SIZE)
optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

epsilon = EPSILON

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    for step in range(500):
        # Explore or exploit
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # random
        else:
            with torch.no_grad():
                q_values = agent(torch.FloatTensor(state))
                action = q_values.argmax().item()  # best action

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Save to memory
        memory.push(state, action, reward, next_state, done)
        state = next_state

        # Learn from memory
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)

            current_q = agent(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q = agent(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)

            loss = loss_fn(current_q, target_q.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # Decay epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if episode % 50 == 0:
        print(f"Episode {episode} | Total Reward: {total_reward} | Epsilon: {epsilon:.3f}")

env.close()
print("Training done!")
