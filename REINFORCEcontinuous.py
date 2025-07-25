from time import perf_counter
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

env = gym.make("HalfCheetah-v5",max_episode_steps=200,render_mode=None,)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
actions_low = torch.tensor(env.action_space.low)
actions_high = torch.tensor(env.action_space.high)

class Actor(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        # out1, in2 = state_dim*2,action_dim*2
        out1, in2 = 256,256
        self.fc1 = nn.Linear(state_dim,out1,bias=True)
        self.fc2 = nn.Linear(out1,in2)
        self.mu_head = nn.Linear(in2,action_dim)
        self.log_sigma_head = nn.Linear(in2,action_dim)

    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x).clamp(-20,2)
        return mu, log_sigma


class ReinforceContinuous():
    learning_rate_a = 0.001
    discount_factor_g = 0.95
    replay_memory_size = 1000
    mini_batch_size = 32

    actor = None
    critic = None
    loss_fn = nn.CrossEntropyLoss()
    optimizer = None


    def __init__(self,env, lr=0.001,gamma = 0.96):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.actor = Actor(state_dim,action_dim)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
    
    def update_policy(self, rewards, log_probs, optimizer):
        log_probs = torch.stack(log_probs)
        loss = -torch.mean(log_probs) * (sum(rewards) -15)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def compute_returns(self, rewards):
        R = 0
        Returns = []
        for r in reversed(rewards):
            R = r+ self.gamma*R
            Returns.append(R)
        Returns.reverse()
        Returns = torch.tensor(Returns,dtype = torch.float32)
        return (Returns-Returns.mean())/(Returns.std()+1e-8)

    def eval(self, env, episodes=1):
        actor = Actor(state_dim=state_dim,action_dim=action_dim)
        actor.load_state_dict(self.actor.state_dict())
        srw = []
        for i in range(episodes):
            tin = perf_counter()
            state, _ = env.reset()
            rewards = []
            log_probs = []
            done = False

            while not done:
                # select action
                state = torch.from_numpy(state).float().unsqueeze(0)
                mu, log_sigma = actor(state)
                sigma = log_sigma.exp()
                dist = torch.distributions.Normal(mu, sigma)
                raw_action = dist.sample()
                log_prob   = dist.log_prob(raw_action).sum(dim=-1)
                action     = torch.max(torch.min(raw_action, actions_high), actions_low)

                #take step
                next_state, reward, terminated, truncated, info = env.step(action.detach().numpy()[0])
                rewards.append(reward)
                log_probs.append(log_prob)
                state = next_state
                done = terminated or truncated
            
            # Tinh return va update
            u = sum(rewards)
            # print(f"    Episode {i}: {u:.2f}, in {perf_counter()-tin:.2f}s")
            srw.append(round(float(u),2))
        return srw



    def train(self,episodes = 500):
        eval_GG = []
        srw = []
        for i in range(episodes):
            tin = perf_counter()
            state, _ = self.env.reset()
            rewards = []
            log_probs = []
            done = False
            score = 0

            while not done:
                # select action
                state = torch.from_numpy(state).float().unsqueeze(0)
                mu, log_sigma = self.actor(state)
                sigma = log_sigma.exp()
                dist = torch.distributions.Normal(mu, sigma)
                raw_action = dist.sample()
                log_prob   = dist.log_prob(raw_action).sum(dim=-1)
                action     = torch.max(torch.min(raw_action, actions_high), actions_low)

                #take step
                next_state, reward, terminated, truncated, info = self.env.step(action.detach().numpy()[0])
                rewards.append(reward)
                log_probs.append(log_prob)
                state = next_state
                done = terminated or truncated
            
            # Tinh return va update
            returns = self.compute_returns(rewards=rewards)
            log_probs = torch.stack(log_probs)
            loss = -(log_probs*returns).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            u = sum(rewards)
            srw.append(round(float(u),2))
            eval_GG.append(self.eval(gym.make("HalfCheetah-v5",max_episode_steps=200,render_mode=None),episodes=5))
            print(f"Episode {i}: {u:.2f}, in {perf_counter()-tin:.2f}s")
        return srw, eval_GG

            #update policy
            # self.update_policy(rewards=rewards, log_probs=log_probs,optimizer=self.optimizer)



if __name__ == '__main__':
    Trainer = ReinforceContinuous(env=env)
    srw,data = Trainer.train()
    # print(srw, data)
    # Tính các thống kê
    data = np.array(data)
    means = data.mean(axis=1)
    stds = data.std(axis=1)
    mins = data.min(axis=1)
    maxs = data.max(axis=1)
    x = np.arange(data.shape[0])

    # Biểu đồ Mean ± Std
    plt.figure()
    plt.plot(x, means)
    plt.fill_between(x, means - stds, means + stds, alpha=0.3)
    plt.title("Mean ± Standard Deviation over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.show()

    # Biểu đồ Mean với Min–Max Envelope
    plt.figure()
    plt.plot(x, means)
    plt.fill_between(x, mins, maxs, alpha=0.3)
    plt.title("Mean with Min–Max Envelope over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.show()

    # srw = Trainer.eval(gym.make("HalfCheetah-v5",max_episode_steps=200,render_mode='human'),episodes=5)
    # srw = Trainer.eval(gym.make("HalfCheetah-v5",max_episode_steps=200,render_mode=None),episodes=5)
    # print(srw)