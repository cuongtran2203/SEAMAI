import torch
import torch.nn as nn
import tqdm
from torch.distributions import MultivariateNormal
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Memory class to store action, state, log probability, reward, and terminal state
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    # Clear memory
    def clear_memory(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()


# Actor-Critic model definition
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Sigmoid()
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.init_action_std = action_std
        self.action_std = action_std
        self.action_var = torch.full((action_dim,), action_std ** 2).to(device)
        self.conv_mat = None

    def forward(self):
        raise NotImplementedError

    # Action selection
    def act(self, state, memory):
        action_mean = self.actor(state).to(device, dtype=torch.float32)
        cov_mat = torch.diag(self.action_var).to(device, dtype=torch.float32)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach(), action_mean.detach()

    # Evaluate the policy
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device, dtype=torch.float32)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    # Decay the standard deviation of the action distribution
    def std_decay(self, epoch):
        self.action_std = self.init_action_std * (0.9 ** epoch)
        self.action_var = torch.full((self.action_dim,), self.action_std ** 2).to(device)


# Proximal Policy Optimization (PPO) algorithm
class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device, dtype=torch.float32)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device, dtype=torch.float32)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    # Decay the exploration standard deviation
    def explore_decay(self, epoch):
        self.policy.std_decay(epoch)
        self.policy_old.std_decay(epoch)

    # Select action based on the current policy
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device, dtype=torch.float32)
        actions = self.policy_old.act(state, memory)
        stds = self.policy_old.action_var
        return actions[0].cpu().data.numpy().flatten(), actions[
            1].cpu().data.numpy().flatten(), stds.cpu().data.numpy().flatten()

    # Select action based on the mean of the action distribution (exploitation)
    def exploit(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device, dtype=torch.float32)
        action_mean = self.policy.actor(state)
        return action_mean[0].cpu().data.numpy()

    # Update the policy
    def update(self, memory, r=0):
        try:
            # Monte Carlo estimate of rewards
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Normalize the rewards
            rewards = torch.tensor(rewards).to(device, dtype=torch.float32)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            # Convert list to tensor
            old_states = torch.squeeze(torch.stack(memory.states).to(device, dtype=torch.float32), 1).detach()
            old_actions = torch.squeeze(torch.stack(memory.actions).to(device, dtype=torch.float32), 1).detach()
            old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device, dtype=torch.float32).detach()

            logger.info('====================================>')
            logger.info('Agent Training')

            # Optimize policy for K epochs
            for _ in tqdm.tqdm(range(self.K_epochs)):
                # Evaluate old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

                # Calculate the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Calculate surrogate loss
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

                # Take a gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

            logger.info('Agent loss updated: ' + str(loss.mean()))

            # Copy new weights into old policy
            self.policy_old.load_state_dict(self.policy.state_dict())
            logger.info('Agent has been updated: ' + str(r))
        except Exception as e:
            logger.error('Agent NOT updated: ' + str(r) + ' due to ' + str(e))
