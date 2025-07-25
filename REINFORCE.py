import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

resolution = 11
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
# num_action = env.action_space.shape[0]
# action_size_space = [np.linspace(env.action_space.low[i],env.action_space.high[i],resolution)for i in range(num_action)]

# action_size = num_action*resolution
action_size = env.action_space.n

class Actor(nn.Module):
    def __init__(self,env):
        super().__init__()
        self.fc1 = nn.Linear(state_size,32,bias=True)
        self.fc2 = nn.Linear(32,action_size)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)



class EnvReinforce():
    learning_rate_a = 0.001
    discount_factor_g = 0.95
    replay_memory_size = 1000
    mini_batch_size = 32

    actor = None
    critic = None
    loss_fn = nn.CrossEntropyLoss()
    optimizer = None


    def __init__(self,env):
        actor = Actor(env)
        self.env = env
        self.optimizer = nn.CrossEntropyLoss(actor.parameters(), lr=self.learning_rate_a)
    
    def update_policy(self, rewards, log_probs, optimizer):
        log_probs = torch.stack(log_probs)
        loss = -torch.mean(log_probs) * (sum(rewards) -15)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self,episodes = 5000):
        env = self.env
        for i in episodes:
            state, _ = env.reset()
            done = False
            score = 0
            log_probs = []
            rewards = []

            while not done:
                # select action
                state = torch.tensor(state,dtype = torch.float32).reshape(1,-1)
                probs = self.actor(state)
                action = torch.multinomial(probs,1).item()
                log_prob = torch.log(probs[:,action])


                #take step
                next_state, reward, terminated, truncated, info = env.step(action)
                score += reward
                rewards.append(reward)
                log_probs.append(log_prob)
                state = next_state
            
            #update policy
            print(f"Episode {i}: {score}")
            self.update_policy(rewards=rewards, log_probs=log_probs,optimizer=self.optimizer)



if __name__ == '__main__':
    Trainer = EnvReinforce(env=env)
    Trainer.train()