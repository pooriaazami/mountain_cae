import numpy as np

from TileCodeAgent import TileCodeAgent


def main():
    agent = TileCodeAgent()
    action = agent.choose_action(np.array([0., 0.]))


if __name__ == '__main__':
    main()
