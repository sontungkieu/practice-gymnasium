# import gymnasium as gym
# env = gym.make("CartPole-v1", render_mode="human")

# observation, info = env.reset()

# # print(f"starting observation: {observation}")

# episode_over = False
# total_reward = 0

# while not episode_over:
#     action = env.action_space.sample()
#     print(f"action space: {env.action_space}")
#     print(f"observation space: {env.observation_space}")
#     observation, reward, terminated, truncated, info = env.step(action)
#     total_reward += reward
#     episode_over = terminated or truncated
#     print(f"observation: {observation}")
#     print(f"info: {info}")

# print(f"Episode finished! Total reward: {total_reward}")
# env.close()


import gymnasium as gym
import ale_py
# Initialise the environment
gym.register_envs(ale_py)
env = gym.make("ALE/Pacman-v5", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
