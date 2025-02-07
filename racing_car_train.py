import gym

# Create environment
env = gym.make("CarRacing-v0")

# Reset environment
state = env.reset()

# Run environment with random actions
for _ in range(100):  # Run for 100 steps
    action = env.action_space.sample()  # Sample a random action
    state, reward, done, truncated, info = env.step(action)  # Take action in env
    env.render()  # Render environment
    if done:  # If the episode is done, reset environment
        env.reset()

env.close()


# start to train: -------------
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple, deque
import random



# Define the Prioritized Replay Buffer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Define Actor-Critic Networks
# action [0.5 0.5 0.5]
class ActorCriticNet(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(ActorCriticNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.actor = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        shared_out = self.shared(x)
        policy = self.actor(shared_out)
        value = self.critic(shared_out)
        return policy, value


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self,state, action, reward, next_state, done,  priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

# Define the Agent with Actor-Critic
class Agent:
    def __init__(self, state_dim, action_dim, buffer_capacity=10000, gamma=0.99, lr=1e-3, alpha=0.6, beta=0.4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.buffer = PrioritizedReplayBuffer(buffer_capacity, alpha)
        self.beta = beta

        # Initialize Actor-Critic networks
        self.actor_critic_net = ActorCriticNet(state_dim, 3).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic_net.parameters(), lr=lr)

        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            policy, _ = self.actor_critic_net(state)
           #dist = torch.distributions.Categorical(policy)
            action = np.array(policy)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        #transition = self.transition(state, action, reward, next_state, done)
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self, batch_size):
        if len(self.buffer.buffer) < batch_size:
            return

        # Sample batch from buffer
        experiences, indices, weights = self.buffer.sample(batch_size, self.beta)
        batch = self.transition(*zip(*experiences))
#[(),(),(),()]
        # Convert batch to tensors
        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        weights = weights.to(self.device)

        # 1. Compute TD Target
        with torch.no_grad():
            _, next_values = self.actor_critic_net(next_states)
            td_target = rewards + self.gamma * (1 - dones) * next_values.squeeze()

        # 2. Compute TD Error
        _, values = self.actor_critic_net(states)
        td_error = td_target - values.squeeze()

        # 3. Update Critic Network
       # critic_loss = (td_error.pow(2) * weights).mean()
        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

        # 4. Compute Advantage
        with torch.no_grad():
            advantage = td_error.detach()

        # 5. Update Actor Network
        policies, _ = self.actor_critic_net(states)
        log_probs = torch.log(policies.gather(1, actions)).squeeze()
        actor_loss = -(log_probs * advantage * weights).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Update priorities in replay buffer
        priorities = td_error.abs().cpu().numpy()
        self.buffer.update_priorities(indices, priorities)
