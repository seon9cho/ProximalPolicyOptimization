from . import cartpole_net
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
    experience = []
    for t in range(episode_length):
        env.render()
        action, prob = policy(state.reshape(1, len(state)))
        action = action[0]
        prob = prob[0]
        next_state, reward, done, info = env.step(action)
        experience.append([state, reward, action, prob.detach().numpy(), next_state])
        state = next_state
        if done:
            return np.array(experience)
    return np.array(experience)

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

def likelihood(probs, actions):
    return probs[range(probs.shape[0]), actions].unsqueeze(1)

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

def main():

    env = gym.make('CartPole-v0')
    policy = CartpolePolicyNetwork()
    value = CartpoleValueNetwork()

    policy_optim = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.01)
    value_optim = optim.Adam(value.parameters(), lr=1e-3, weight_decay=1)

    value_criterion = nn.MSELoss()

    # Hyperparameters
    epochs = 30
    env_samples = 100
    episode_length = 200
    gamma = 0.99
    value_epochs = 2
    policy_epochs = 5
    batch_size = 32
    policy_batch_size = 256
    epsilon = 0.2
    loop = tqdm(total=epochs, position=0, leave=False)

    policy_loss = torch.tensor([np.nan])
    avl_list = []
    apl_list = []
    ast_list = []
    for _ in range(epochs):
        # generate rollouts
        rollouts = []
        for _ in range(env_samples):
            # don't forget to reset the environment at the beginning of each episode!
            # rollout for a certain number of steps!
            experience = generate_rollout(env, episode_length, policy)
            rollouts.append(experience)

        standing_len = sum(len(experience) for experience in rollouts)
        ast = standing_len / env_samples
        #print('avg standing time:', standing_len / env_samples)
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
            for states, rewards, actions, probs, next_states, returns in data_loader:
                size = len(returns)
                value_optim.zero_grad()
                baseline = value(states)
                returns = returns.float()
                value_loss = value_criterion(baseline, returns.reshape(size, 1))
                value_loss.backward()
                value_optim.step()

                avl += value_loss.item()
                loop.set_description('value loss:{:.4f}, policy loss:{:.3f}, standing time: {}'.format(value_loss.item(), policy_loss.item(), ast))

        apl = 0
        for p in policy.parameters():
            p.requires_grad = True
        for p in value.parameters():
            p.requires_grad = False
        # Learn a policy
        for _ in range(policy_epochs):
            # train policy network
            for states, rewards, actions, probs, next_states, returns in data_loader:
                size = len(returns)
                policy_optim.zero_grad()
                baseline = value(states)
                baseline = baseline.detach()

                returns = returns.float().reshape(size, 1)
                new_probs = policy(states, False)

                old_likelihood = likelihood(probs, actions)
                new_likelihood = likelihood(new_probs, actions)

                ratio = new_likelihood / old_likelihood
                advantage = returns - baseline

                l_1 = ratio * advantage
                l_2 = torch.clamp(ratio, 1-epsilon, 1+epsilon)*advantage
                policy_loss = -torch.mean(torch.min(l_1, l_2))
                policy_loss.backward()
                policy_optim.step()

                apl += policy_loss.item()
                loop.set_description('value loss:{:.4f}, policy loss:{:.3f}, standing time: {}'.format(value_loss.item(), policy_loss.item(), ast))

        avl /= (len(data_loader) * value_epochs)
        apl /= (len(data_loader) * policy_epochs)

        avl_list.append(avl)
        apl_list.append(apl)
        ast_list.append(ast)

        loop.update(1)

    return avl_list, apl_list, ast_list, policy, value
