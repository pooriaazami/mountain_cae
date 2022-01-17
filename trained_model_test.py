import numpy as np
import gym

from Agent import Agent
from TileCoding import TileCode


class TrainedAgent(Agent):
    def __init__(self, **kwargs):
        self.x_tile_count = kwargs.get('x_tile_count', 8)
        self.y_tile_count = kwargs.get('y_tile_count', 8)
        self.tile_count = kwargs.get('tile_count', 9)

        self.num_actions = kwargs.get('num_actions', 3)

        self.tile_coding = TileCode((self.x_tile_count, self.y_tile_count), self.tile_count)
        self.weights = np.load('policy.npy')

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

    def choose_action(self, state):
        tiles = self.tile_coding.get_tile(self.normalize_state(state))
        values = []
        for action in range(self.num_actions):
            values.append(np.sum(self.weights[action][tiles]))

        action = self.argmax(values)

        return action, values[action], tiles

    def start(self, state):
        action, value, tiles = self.choose_action(state)

        return action

    def step(self, state):
        action, value, tiles = self.choose_action(state)

        return action


def run_episode(env, agent):
    observation, done = env.reset(), False
    action = agent.start(observation)

    counter = 0
    reward_sum = 0

    while not done:
        env.render()
        observation, reward, done, message = env.step(action)
        action = agent.step(observation)
        reward_sum += reward
        counter += 1


def main():
    env = gym.make('MountainCar-v0')
    agent = TrainedAgent()

    for _ in range(100):
        run_episode(env, agent)

    env.close()


if __name__ == '__main__':
    main()
