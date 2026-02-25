import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal   # CHANGED: using Normal instead of Categorical


"""
Memory function for PPO that stores 
1) state 
2) reward 
3) action 
4) log probability of action as per policy 
5) value as per critic
"""
class PPOMemory:
    def __init__(self, batch_size):
        self.rewards = []
        self.states = []
        self.actions = []
        self.values = [] 
        self.probs = []   # log probability of action under old policy
        self.dones = [] 
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.values), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches
    
    def store_memory(self, state, action, reward, value, prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.probs.append(prob)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []
        self.dones = []


"""
Actor Network with 2 hidden layers with 256 neurons.
REMOVED softmax.
"""
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions)   # outputs mean vector (size 12)
        )

        # CHANGED: learn log_std parameter for each action dimension
        self.log_std = nn.Parameter(T.zeros(n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        mean = self.actor(state)

        # convert log_std to std
        std = T.exp(self.log_std)
        std = std.expand_as(mean)

        dist = Normal(mean, std)   # CHANGED: Normal distribution
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2,
                 batch_size=64, n_epochs=10):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def save_memory(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, reward, vals, probs, done)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)

        raw_action = dist.sample()

        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        # squash to (-1, 1)
        squashed = T.tanh(raw_action)

        # scale to (0, 180)
        action = (squashed + 1) / 2 * 180.0

        action = T.squeeze(action).cpu().detach().numpy()
        log_prob = T.squeeze(log_prob).item()
        value = T.squeeze(value).item()

        return action, log_prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # calculation of advantage 
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (
                        reward_arr[k] +
                        self.gamma * values[k+1] *
                        (1 - int(dones_arr[k])) -
                        values[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.float).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()

                # CHANGED: sum log probs across action dims
                new_probs = dist.log_prob(actions).sum(dim=-1)

                prob_ratio = (new_probs - old_probs).exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(
                    prob_ratio,
                    1 - self.policy_clip,
                    1 + self.policy_clip
                ) * advantage[batch]

                actor_loss = -T.min(
                    weighted_probs,
                    weighted_clipped_probs
                ).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()