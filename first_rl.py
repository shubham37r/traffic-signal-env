import gymnasium as gym

#creating an environment
env = gym.make("CartPole-v1", render_mode = "human")

#reset the environment
state, info = env.reset()

print("Starting State: ", state)

# Run the loop for 100 steps
for step in range(100):
    # Agent takes random action
    action = env.action_space.sample()

    # Environment gives back new state, reward, and if game is over
    new_state, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step+1} | Action : {action} | Reward : {reward} | Done: {terminated}")

    #if the pole falls, restart
    if terminated or truncated:
        print("Pole fell! Resetting...")
        state, info = env.reset()

env.close()