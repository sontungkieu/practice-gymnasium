import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_states: int, h1_nodes: int, out_actions: int)->None:
        super().__init__()

        #Define network layer
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x
    

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([],maxlen=maxlen)
    
    def append(self,transition):
        self.memory.append(transition)

    def sample(self,sample_size):
        return random.sample(self.memory,sample_size)
    
    def __len__(self):
        return len(self.memory)
    
class FrozenLakeDQL():

    learning_rate_a = 0.001
    discount_factor_g = 0.9
    network_sync_rate = 10
    replay_memory_size = 1000
    mini_batch_size = 32

    #NN
    loss_fn = nn.MSELoss()
    optimizer = None

    ACTIONS = ['L', 'D', 'R', 'U']

    def train(self, episodes, render=False, is_slippery =False):
        # create train instance
        env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=is_slippery, render_mode = 'human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1
        # epsilon_decay = 0.001
        memory = ReplayMemory(maxlen=self.replay_memory_size)

        policy_dqn = DQN(in_states=num_states, h1_nodes=32,out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=32,out_actions=num_actions)

        #copy policy to target 
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy random before training')
        self.print_dqn(policy_dqn)

        self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        reward_per_episode = np.zeros(episodes)

        epsilon_history = []

        step_count = 0
        

        for i in range(episodes):
            state = env.reset()[0]

            episode_over = False

            while not episode_over:
                if random.random()<epsilon:
                    action = env.action_space.sample()
                else:        
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                new_state, reward, terminated, truncated, info = env.step(action)

                #save to experience replay
                memory.append((state,action,new_state,reward,terminated))

                state = new_state

                step_count += 1
                episode_over = truncated or terminated



            if reward == 1:
                reward_per_episode[i]=1

            if len(memory)>self.mini_batch_size and np.sum(reward_per_episode)>0:
                mini_batch = memory.sample(sample_size=self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                #decay epsilon
                epsilon = max(epsilon-1/episodes,0)
                epsilon_history.append(epsilon)

                #update network after certain number of step
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
        env.close()
        torch.save(policy_dqn.state_dict(),"frozen_lake_dqn.pt")

        plt.figure(1)
        
        sum_rewards= np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(reward_per_episode[max(0,x-100):x+1])
        
        plt.subplot(121)
        plt.plot(sum_rewards)
        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.savefig('frozen_lake_dqn.png')
    
    def state_to_dqn_input(self, state: int, num_states: int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    def optimize(self, mini_batch, policy_dqn: DQN, target_dqn: DQN):

        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                target = torch.FloatTensor([reward])

            else:
                with torch.no_grad():
                    target = torch.FloatTensor([reward + self.discount_factor_g*target_dqn(self.state_to_dqn_input(new_state,num_states)).max()])

            current_q = policy_dqn(self.state_to_dqn_input(state,num_states=num_states))
            current_q_list.append(current_q)
            
            target_q = target_dqn(self.state_to_dqn_input(state,num_states))
            target_q[action] = target
            target_q_list.append(target_q)
        loss = self.loss_fn(torch.stack(current_q_list),torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def test(self,episodes, is_slippery = False):
        env = gym.make('FrozenLake-v1', map_name = '4x4', is_slippery=is_slippery, render_mode= 'human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        #load dqn
        policy_dqn = DQN(in_states=num_states,h1_nodes=32,out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load('frozen_lake_dqn.pt'))
        policy_dqn.eval()
        print("Policy trained")
        self.print_dqn(policy_dqn)
        for i in range(episodes):
            state = env.reset()[0]
            episode_over = False
            while not episode_over:
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state,num_states)).argmax().item()
                new_state, reward, terminated, truncated, info = env.step(action)
                state = new_state
                episode_over = terminated or truncated
        env.close()

    def print_dqn(self, dqn: DQN):
        num_states = dqn.fc1.in_features

        for s in range(num_states):
            q_value = ''
            for q in dqn(self.state_to_dqn_input(state=s,num_states=num_states)).tolist():
                q_value+= f"{q:.2f} "

            q_value = q_value.rstrip()

            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s,num_states)).argmax()]

            print(f'{s:02},{best_action},[{q_value}]',end=' ')
            if (s+1)%4==0:
                print()



if __name__ == '__main__':
    frozen_lake  = FrozenLakeDQL()
    is_slippery = True
    frozen_lake.train(8000,is_slippery=is_slippery)
    frozen_lake.test(1, is_slippery=is_slippery)
