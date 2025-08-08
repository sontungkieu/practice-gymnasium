import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training = True, render = False):

    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    n_states = 20*20
    pos_space = np.linspace(env.observation_space.low[0],env.observation_space.high[0],20)
    vel_space = np.linspace(env.observation_space.low[1],env.observation_space.high[1],20)
    
    if is_training:
        q = np.zeros((len(pos_space),len(vel_space), env.action_space.n) )
    else:
        with open("mountaincar.pkl","rb") as file:
            q = pickle.load(file)
    learning_rate_a = 0.9
    discount_factor_g = 0.9

    epsilon = 1
    epsilon_decay = 2/episodes
    rng = np.random.default_rng() #random number generator

    reward_per_episode = np.zeros(episodes)

    for _ in range(episodes):
        state = env.reset()[0]
        state_p = np.digitize(state[0],pos_space)
        state_v = np.digitize(state[1],vel_space)
        # print(state)
        episode_over = False
        rewards = 0
        while not episode_over:
            if is_training and rng.random()<epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p,state_v,:])

            new_state,reward, terminated, truncated, info = env.step(action)
            new_state_p = np.digitize(new_state[0],pos_space)
            new_state_v = np.digitize(new_state[1],vel_space)

            if is_training:
                q[state_p,state_v,action] = q[state_p,state_v,action] + learning_rate_a*(
                    reward + discount_factor_g*np.max(q[new_state_p,new_state_v,:])-q[state_p,state_v,action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            rewards += reward
            episode_over = terminated or rewards<-1000
        
        epsilon = max(epsilon-epsilon_decay,0)

        if epsilon == 0:
            learning_rate_a = 0.0001
        
        reward_per_episode[_] = rewards

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t]=np.mean(reward_per_episode[max(0,t-100):t+1])
    plt.plot(sum_rewards)
    plt.savefig('mountaincar.png')

    if is_training:
        f = open("mountaincar.pkl","wb")
        pickle.dump(q,f)
        f.close()


if __name__ == '__main__':
    run(1000,is_training=True, render=False)
    run(1,is_training=False, render=True)