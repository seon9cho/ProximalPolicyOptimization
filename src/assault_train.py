from . import assault_net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import gym


def generate_rollout(env, episode_length, policy):
    state = env.reset()
    state = torch.tensor(state).cuda(async=True).view(3, 250, 160).float()
    experience = []
    total_reward = 0
    for t in range(episode_length):
        env.render()
        param, action = policy(state.unsqueeze(0))
        param = param[0]
        action = action[0]
        next_state, reward, done, _ = env.step(action)
        experience.append([state.cpu().detach().numpy(), reward, action, param.cpu().detach().numpy()])
        state = torch.tensor(next_state).cuda(async=True).view(3, 250, 160).float()
        total_reward += reward
        if done:
            return total_reward, np.array(experience)
    return total_reward, np.array(experience)

def calculate_returns(rollouts, gamma):
    all_returns = []
    for r in rollouts:
        returns = [0]
        for i, s in enumerate(np.flip(r, axis=0)):
            reward = s[1]
            discounted_sum = gamma*returns[i]
            returns.append(reward + discounted_sum)
        all_returns.append(returns[1:][::-1])
    return all_returns

class ExperienceDataset(Dataset):
    def __init__(self, experience):
        super().__init__()
        self.exp_joined = []
        for e in experience:
            self.exp_joined.extend(e.tolist())

    def __getitem__(self, index):
        return self.exp_joined[index]

    def __len__(self):
        return len(self.exp_joined)

def likelihood(probs, actions):
    return probs[range(probs.shape[0]), actions].unsqueeze(1)

def main():

    env = gym.make('Assault-v0')
    policy = AssaultPolicyNetwork()
    policy.cuda()
    value = AssaultValueNetwork()
    value.cuda()

    policy_optim = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.01)
    value_optim = optim.Adam(value.parameters(), lr=1e-3, weight_decay=1)

    value_criterion = nn.MSELoss()

    # Hyperparameters
    epochs = 20
    env_samples = 100
    episode_length = 100
    gamma = 0.99
    value_epochs = 2
    policy_epochs = 5
    batch_size = 32
    policy_batch_size = 256
    epsilon = 0.2
    loop = tqdm(total=epochs, position=0, leave=False)

    avl_list = []
    apl_list = []
    ar_list = []
    policy_loss = torch.tensor([np.nan])
    for k in range(epochs):
        # generate rollouts
        rewards = []
        rollouts = []
        for _ in range(env_samples):
            # don't forget to reset the environment at the beginning of each episode!
            # rollout for a certain number of steps!
            total_reward, experience = generate_rollout(env, episode_length, policy)
            rollouts.append(experience)
            rewards.append(total_reward)

        avg_rewards = np.mean(rewards)
        ar_list.append(avg_rewards)
        returns = calculate_returns(rollouts, gamma)

        for i in range(env_samples):
            rollouts[i] = np.column_stack([rollouts[i], returns[i]])

        # Approximate the value function
        dataset = ExperienceDataset(rollouts)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        avl = 0
        for p in policy.parameters():
            p.requires_grad = False
        for p in value.parameters():
            p.requires_grad = True
        for _ in range(value_epochs):
            # train value network
            for states, rewards, actions, params, returns in data_loader:
                size = len(returns)
                states = torch.tensor(states).cuda(async=True).view(size, 3, 250, 160).float()

                baseline = value(states)
                returns = returns.cuda(async=True).float().unsqueeze(1)

                value_optim.zero_grad()
                value_loss = value_criterion(baseline, returns)
                value_loss.backward()
                value_optim.step()
                avl += value_loss.item()

                loop.set_description('value loss:{:.4f}, policy loss:{:.3f}, Average rewards: {}'.format(value_loss.item(), policy_loss.item(), avg_rewards))

        avl /= (len(data_loader) * value_epochs)
        avl_list.append(avl)

        data_loader = DataLoader(dataset, batch_size=policy_batch_size, shuffle=True, pin_memory=True)
        apl = 0
        for p in policy.parameters():
            p.requires_grad = True
        for p in value.parameters():
            p.requires_grad = False
        # Learn a policy
        for j in range(policy_epochs):
            # train policy network
            for states, rewards, actions, params, returns in data_loader:
                size = len(returns)
                states = torch.tensor(states).cuda(async=True).view(size, 3, 250, 160).float()

                policy_optim.zero_grad()
                baseline = value(states)
                baseline = baseline.detach()
                returns = returns.cuda(async=True).float().unsqueeze(1)

                new_params = policy(states, False)

                old_likelihood = likelihood(params.cuda(async=True), actions)
                new_likelihood = likelihood(new_params, actions)

                ratio = new_likelihood / old_likelihood
                advantage = returns - baseline

                l_1 = ratio * advantage
                l_2 = torch.clamp(ratio, 1-epsilon, 1+epsilon)*advantage
                policy_loss = -torch.mean(torch.min(l_1, l_2))
                policy_loss.backward()
                policy_optim.step()

                apl += policy_loss.item()

                loop.set_description('value loss:{:.4f}, policy loss:{:.3f}, Average rewards: {}'.format(value_loss.item(), policy_loss.item(), avg_rewards))

        apl /= (len(data_loader) * policy_epochs)
        apl_list.append(apl)

        loop.update(1)

    return avl_list, apl_list, ar_list, policy, value