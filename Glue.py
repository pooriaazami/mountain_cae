import gym
from gym import Env

from DeepAgent import DeepAgent
from TileCodeAgent import TileCodeAgent


class Glue:
    def __init__(self, load_model=False, **kwargs):
        self.env: Env = gym.make('MountainCar-v0')
        # self.agent = DeepAgent(**kwargs)
        self.agent = TileCodeAgent(**kwargs)

        if load_model:
            self.agent.load_model()

    def run_episode(self):
        observation, done = self.env.reset(), False

        reward_sum, step_count = 0, 0
        action = self.agent.start(observation)
        while not done:
            self.env.render()
            observation, reward, done, _ = self.env.step(action)

            action = self.agent.step(observation, reward)
            reward_sum += reward
            step_count += 1

        error = self.agent.end(reward)

        return reward_sum // step_count, error

    def close(self):
        self.env.close()
        self.agent.save('model.npy')

    def save_model(self, name):
        self.agent.save(name)

    def test(self):
        self.agent.test()
