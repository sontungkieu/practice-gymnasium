import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training = True, render = False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n) )
    else:
        with open("frozenlake8x8.pkl","rb") as file:
            q = pickle.load(file)
    learning_rate_a = 0.9
    discount_factor_g = 0.9

    epsilon = 1
    epsilon_decay = 0.0001
    rng = np.random.default_rng() #random number generator

    reward_per_episode = np.zeros(episodes)

    for _ in range(episodes):
        state, observation = env.reset()
        # print(state)
        episode_over = False
        while not episode_over:
            if is_training and rng.random()<epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            new_state,reward, terminated, truncated, info = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a*(
                    reward + discount_factor_g*np.max(q[new_state,:])-q[state,action]
                )

            state = new_state
            episode_over = terminated or truncated
        
        epsilon = max(epsilon-epsilon_decay,0)

        if epsilon == 0:
            learning_rate_a = 0.0001
        
        if reward == 1:
            reward_per_episode[_] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t]=np.sum(reward_per_episode[max(0,t-100):t+1])
    plt.plot(sum_rewards)
    plt.savefig('frozenlake8x8.png')

    if is_training:
        f = open("frozenlake8x8.pkl","wb")
        pickle.dump(q,f)
        f.close()


if __name__ == '__main__':
    # run(100000,is_training=True, render=False)
    run(1,is_training=False, render=True)