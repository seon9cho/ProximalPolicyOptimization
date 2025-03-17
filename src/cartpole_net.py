import torch
import torch.nn as nn
import numpy as np

cartpole_state_dim = 4
cartpole_action_dim = 2
class CartpolePolicyNetwork(nn.Module):
  def __init__(self):
    super(CartpolePolicyNetwork, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(cartpole_state_dim, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, cartpole_action_dim)
    )
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input, get_action=True):
    batch_dim, state_dim = input.shape
    input = torch.tensor(input).double()
    action_scores = self.net(input.float())
    action_probs = self.softmax(action_scores)
    if not get_action:
        return action_probs
    # Sample from the distribution if get_action=True
    actions = [np.random.choice(np.arange(cartpole_action_dim),
                                p=prob.detach().numpy())
               for prob in action_probs]
    return np.array(actions), action_probs

class CartpoleValueNetwork(nn.Module):
  def __init__(self):
    super(CartpoleValueNetwork, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(cartpole_state_dim, 10),
        nn. ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
    )

  def forward(self, input):
    input = torch.tensor(input).double()
    value = self.net(input.float())
    return value