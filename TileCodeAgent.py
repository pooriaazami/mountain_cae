import numpy as np

from Agent import Agent
from TileCoding import TileCode


class TileCodeAgent(Agent):
    def __init__(self, **kwargs):
        self.x_tile_count = kwargs.get('x_tile_count', 8)
        self.y_tile_count = kwargs.get('y_tile_count', 8)
        self.tile_count = kwargs.get('tile_count', 9)

        self.gamma = kwargs.get('gamma', 1)
        self.epsilon = kwargs.get('epsilon', 0.0)
        self.alpha = kwargs.get('alpha', 0.01)

        self.num_actions = kwargs.get('num_actions', 3)

        self.tile_coding = TileCode((self.x_tile_count, self.y_tile_count), self.tile_count)
        self.weights = np.ones((self.num_actions, self.tile_coding.tile_count))

        self.decay_rate = kwargs.get('decay_rate', 0.005)
        self.decay_period = kwargs.get('decay_period', 200)

        self.last_state = None
        self.last_action = None
        self.last_tiles = None

        self.counter = 0
        self.error_sum = 0
        self.error_average = 0

    def argmax(self, values):
        ties = []
        maximum = float('-inf')

        for index, value in enumerate(values):
            if value == maximum:
                ties.append(index)

            if value > maximum:
                maximum = value
                ties = [index]

        return np.random.choice(ties)

    def normalize_state(self, state):
        normalized_state = np.copy(state)
        normalized_state += np.array([1.2, 0.07])
        normalized_state /= np.array([1.7, 0.14])

        return normalized_state

    def test_tiling(self):
        points = [
            np.array([0.115, 0.114]),
            self.normalize_state(np.array([-1.0, 0.01])),
            self.normalize_state(np.array([0.1, -0.01])),
            self.normalize_state(np.array([0.2, -0.05])),
            self.normalize_state(np.array([-1.0, 0.011])),
            self.normalize_state(np.array([0.2, -0.05])),
            # np.array([0.5, 0.07]),
            # np.array([0.5, -0.07]),
            # np.array([-1.2, 0.07]),
            # np.array([-1.2, -0.07]),
        ]

        for point in points:
            print('normalized: ', point)
            print('answer: ', self.tile_coding.get_tile(point))

        self.tile_coding.plot_tiles(points)

    def choose_action(self, state):
        tiles = self.tile_coding.get_tile(self.normalize_state(state))
        # print(tiles)
        values = []
        for action in range(self.num_actions):
            values.append(np.sum(self.weights[action][tiles]))

        if np.random.random() > self.epsilon:
            action = self.argmax(values)
        else:
            action = np.random.choice(self.num_actions)

        return action, values[action], tiles

    def start(self, state):
        action, value, tiles = self.choose_action(state)

        self.last_state = state
        self.last_action = action
        self.last_tiles = tiles

        self.error_average += 1

        return action

    def step(self, state, reward):
        action, value, tiles = self.choose_action(state)

        last_action_values = np.sum(self.weights[self.last_action][self.last_tiles])
        td_error = reward + self.gamma * value - last_action_values
        grad = np.ones(self.weights[self.last_action][self.last_tiles].shape)
        self.error_sum += td_error
        self.error_average += 1
        # print(value)
        # print('-' * 200)
        # print(self.weights[self.last_action][self.last_tiles], self.last_action, self.last_tiles)
        self.weights[self.last_action][self.last_tiles] += self.alpha * td_error * grad

        self.last_action = action
        self.last_state = state
        self.last_tiles = tiles

        return action

    def end(self, reward):
        last_action_values = np.sum(self.weights[self.last_action][self.last_tiles])
        td_error = reward - last_action_values
        grad = np.ones(self.weights[self.last_action][self.last_tiles].shape)

        self.weights[self.last_action][self.last_tiles] += self.alpha * td_error * grad
        self.error_sum += td_error

        # self.counter += 1
        self.error_average += 1

        # if self.counter == self.decay_period:
        #     self.epsilon = max(self.epsilon - self.decay_rate, 0)
        #     self.counter = 0

        error = self.error_sum / self.error_average

        self.error_sum = 0
        self.error_average = 0

        return error

    def save(self, file_name):
        np.save(file_name, self.weights)

    def test(self):
        self.epsilon = 0
        self.alpha = 0
