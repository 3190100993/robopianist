import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
from dataclasses import dataclass

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_std = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        std = torch.clamp(self.fc_std(x), min=-20, max=2).exp()
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc_value(x)
        return value

class PPO:
    def __init__(self, spec, config, seed, discount):
        self.input_dim = spec.observation_dim
        self.output_dim = spec.action_dim
        self.discount = discount
        self.clip_eps = config.clip_eps
        self.epochs = config.epochs
        self.lr = config.lr

        self.policy = PolicyNetwork(self.input_dim, self.output_dim)
        self.value = ValueNetwork(self.input_dim)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr)

    def sample_actions(self, observation):
        observation = torch.FloatTensor(observation).unsqueeze(0)
        mean, std = self.policy(observation)
        dist = distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return self, action.squeeze(0).detach().numpy()

    def eval_actions(self, observation):
        observation = torch.FloatTensor(observation).unsqueeze(0)
        mean, _ = self.policy(observation)
        return mean.squeeze(0).detach().numpy()

    def update(self, transitions):
        states, actions, log_probs_old, rewards, dones = transitions

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # 方法1：检查所有维度是否有任何一个为True（最通用）
        # 适用于只要有一个条件满足就终止的情况
        dones_global = dones.any(dim=1, keepdim=True).float()

        # # 在update方法中添加
        # print(f"原始dones形状: {dones.shape}")  # (256, 1164)
        # print(f"处理后dones形状: {dones_global.shape}")  # 应是 (256, 1)
        # print(f"dones的统计: 均值={dones.mean()}, 最大值={dones.max()}, 最小值={dones.min()}")
        # print(f"dones_global的统计: 均值={dones_global.mean()}, 终止比例={(dones_global > 0.5).float().mean()}")


        returns = torch.zeros_like(rewards)
        R = 0
        for i in reversed(range(len(rewards))):
            if dones_global[i].item() > 0.5:
                R = 0
            R = rewards[i] + self.discount * R
            returns[i] = R
        returns = torch.FloatTensor(returns)

        for _ in range(self.epochs):
            mean, std = self.policy(states)
            dist = distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(-1)
            values = self.value(states).squeeze()

            advantages = returns - values.detach()

            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.MSELoss()(values, returns)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }
        return self, metrics

@dataclass(frozen=True)
class PPOConfig:
    clip_eps: float = 0.2
    epochs: int = 10
    lr: float = 0.001
