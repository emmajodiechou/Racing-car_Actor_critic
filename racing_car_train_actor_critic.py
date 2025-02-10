import gym

# Create environment
# env = gym.make("CarRacing-v2",render_mode="human")
# # Reset environment
# state = env.reset()

# # Run environment with random actions
# for _ in range(10000):  # Run for 100 steps
#     action = env.action_space.sample()  # Sample a random action
#     state, reward, done, truncated, info = env.step(action)  # Take action in env
#     env.render()  # Render environment
#     if done:  # If the episode is done, reset environment
#         env.reset()

# env.close()


# start to train: -------------
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple, deque
import random



# # Define the Prioritized Replay Buffer
# import torch
import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import gym_super_mario_bros
# from nes_py.wrappers import JoypadSpace
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# # Define Actor-Critic Networks
# # action [0.5 0.5 0.5] 



class ActorNet(nn.Module):
    def __init__(self, input_channels, action_dim):  #choose action(1,3,96,96),learn(32,3,96,96)
        super(ActorNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Tahn(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.Tahn(),
            nn.Linear(512, action_dim)  #(1,3)  #(32,3)
        )

    def forward(self, x):
        row= self.actor(x)
        action1 = torch.tanh(row[0])

        # 第二和第三个动作: 通过 sigmoid 限制在 [0,1]
        action2 = torch.sigmoid(row[1])
        action3 = torch.sigmoid(row[2])

        policy=torch.stack([action1, action2, action3], dim=0)
        return policy


class CriticNet(nn.Module):
    def __init__(self, input_channels):  
        super(CriticNet, self).__init__()
        self.critic = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Tahn(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        value = self.critic(x)##(1,3,96,96),learn(32,3,96,96)
        return  value #(1,1),(32,1)



# # Define the Agent with Actor-Critic
class Agent:
    def __init__(self, state_dim, action_dim, buffer_capacity=10000, gamma=0.99, lr=1e-3, alpha=0.6, beta=0.4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.buffer = []
        self.beta = beta

        # Initialize Actor-Critic networks
        self.actor_net = ActorNet(state_dim, 3).to(self.device)
        self.critic_net = CriticNet(state_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=lr)
        self.batch_size=32
    def choose_action(self, state):

        state = np.transpose(state, (2, 0, 1)).copy() #96*96*3->3*96*96   0120>201
 
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) #(1,3,96,96)

        with torch.no_grad():
            policy= self.actor_net(state)
           #dist = torch.distributions.Categorical(policy)
            action = np.array(policy)
        return action

    def push(self, state, action, reward, next_state, done):
        state = np.transpose(state, (2, 0, 1)).copy()
        next_state = np.transpose(next_state, (2, 0, 1)).copy()
        #transition = self.transition(state, action, reward, next_state, done)
        state = np.expand_dims(state, axis=0)  # 
        next_state = np.expand_dims(next_state, axis=0)  
        self.buffer.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.buffer.buffer) < self.batch_size:
            return

#         # Sample batch from buffer
        batch = random.sample(self.buffer, self.batch_size)  
#       #[(),(),(),()]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device).squeeze(1)  # (batch_size, state_dim)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)  # (batch_size, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # (batch_size, 1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device).squeeze(1)  # (batch_size, state_dim)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)  # (batch_size, 1)
        # print('state dimension',states.shape)
        # print('actions',actions.shape)
        # print('rewards',rewards.shape)
        # print('next_states',next_states.shape)
        # print('dones',dones.shape)
# 
  # 1. Compute TD Target
        with torch.no_grad():
            next_values = self.critic_net(next_states)
            td_target = rewards + self.gamma * (1 - dones) * next_values

#         # 2. Compute TD Error
        values = self.critic_net(states)
        td_error = td_target - values

#         # 3. Update Critic Network
        critic_loss = (td_error.pow(2)).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

#         # 4. Compute Advantage
        with torch.no_grad():
            advantage = td_error.detach()
            advantage=td_target.detech()
#         # 5. Update Actor Network
        policies= self.actor_net(states)
        log_probs = torch.log(policies.gather(1, actions)).squeeze()
        actor_loss = -(log_probs * advantage ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


env = gym.make("CarRacing-v2",render_mode="human", continuous=True)#96*96*3

agent=Agent(env.shape,3)
# Reset environment
state = env.reset()

# Run environment with random actions
for _ in range(10000):  # Run for 100 steps
    action = agent.choose_action(state) # Sample a random action
    next_state, reward, done, truncated, info = env.step(action)  # Take action in env
    agent.push(state, action, reward, next_state, done)
    #env.render()  # Render environment

    agent.learn()





    if done:  # If the episode is done, reset environment
        env.reset()

env.close()