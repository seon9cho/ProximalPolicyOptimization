import torch
import torch.nn as nn
import numpy as np

class AssaultPolicyNetwork(nn.Module):
  def __init__(self, sigma=0.1):
    super(AssaultPolicyNetwork, self).__init__()
    self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
    self.bn1 = nn.BatchNorm2d(16)

    self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
    self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
    self.bn2 = nn.BatchNorm2d(32)

    self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
    self.conv6 = nn.Conv2d(32, 32, kernel_size=(4,2))
    self.linear = nn.Linear(32, 7)

    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input, get_action=True):
    d1, d2, d3, d4 = input.shape
    layer1 = self.relu(self.conv1(input))
    layer2 = self.relu(self.bn1(self.conv2(layer1)))
    layer3 = self.relu(self.conv3(layer2))
    layer4 = self.relu(self.bn2(self.conv4(layer3)))
    layer5 = self.relu(self.conv5(layer4))
    layer6 = self.relu(self.conv6(layer5))
    linear_layer = layer6.view(d1, 32)
    params = self.softmax(self.linear(linear_layer))
    if get_action == False:
        return params

    actions = [np.random.choice(np.arange(7),
                                p=param.cpu().detach().numpy())
                                for param in params]
    return params, np.array(actions)

class AssaultValueNetwork(nn.Module):
  def __init__(self):
    super(AssaultValueNetwork, self).__init__()
    self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
    self.bn1 = nn.BatchNorm2d(16)

    self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
    self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
    self.bn2 = nn.BatchNorm2d(32)

    self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
    self.conv6 = nn.Conv2d(32, 32, kernel_size=(4,2))
    self.linear = nn.Linear(32, 1)

    self.relu = nn.ReLU()

  def forward(self, input):
    d1, d2, d3, d4 = input.shape
    layer1 = self.relu(self.conv1(input))
    layer2 = self.relu(self.bn1(self.conv2(layer1)))
    layer3 = self.relu(self.conv3(layer2))
    layer4 = self.relu(self.bn2(self.conv4(layer3)))
    layer5 = self.relu(self.conv5(layer4))
    layer6 = self.relu(self.conv6(layer5))
    linear_layer = layer6.view(d1, 32)
    output = self.linear(linear_layer)
    return output