import torch
from torch import nn

import numpy as np

from Agent import Agent
from QualityFunction import Quality


class DeepAgent(Agent):
    def __init__(self, **kwargs):
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.decay_rate = kwargs.get('decay_rate', 0.02)
        self.decay_period = kwargs.get('decay_period', 200)

        self.num_actions = kwargs.get('num_actions', 0)

        self.Q = Quality(self.num_actions)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.01)

        self.last_action = None
        self.last_state = None
        self.last_values = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cpu = 'cpu'

        self.Q.to(self.device)
        self.Q.train()

        self.counter = 0

    def argmax(self, values):
        ties = []
        maximum = torch.tensor([float('-inf')]).to(self.device)

        for index, v in enumerate(values):
            if v == maximum:
                ties.append(index)

            if v > maximum:
                ties = [index]
                maximum = v

        action = np.random.choice(ties)

        return action

    def choose_action(self, state):
        values = self.Q(state)

        if np.random.random() > self.epsilon:
            action = self.argmax(values)
        else:
            action = np.random.choice(self.num_actions)

        return action, values

    def start(self, state):
        observation = torch.tensor(state).to(self.device)

        action, values = self.choose_action(observation)

        self.last_state = observation
        self.last_action = action
        self.last_values = values

        return self.last_action

    def step(self, state, reward):
        observation = torch.tensor(state).to(self.device)

        with torch.no_grad():
            action, new_values = self.choose_action(observation)
            td_error = reward + self.gamma * torch.max(new_values)
            target = self.last_values.clone().detach()
            target[self.last_action] = td_error

        loss_value = self.loss(target, self.last_values)

        self.optimizer.zero_grad()
        loss_value.backward()
        self.optimizer.step()

        self.last_values = self.Q(observation)
        self.last_action = action
        self.last_state = observation

        return self.last_action

    def end(self, reward):
        with torch.no_grad():
            td_error = reward
            target = self.last_values.clone().detach()
            target[self.last_action] = td_error

        loss_value = self.loss(target, self.last_values)

        self.optimizer.zero_grad()
        loss_value.backward()
        self.optimizer.step()

        self.counter += 1

        if self.counter == self.decay_period:
            self.epsilon = max(self.epsilon - self.decay_rate, 0)
            self.counter = 0

    def save(self, name='model.pth'):
        torch.save(self.Q.state_dict(), name)

    def load_model(self):
        self.Q.load_state_dict(torch.load('policy.pth'))


