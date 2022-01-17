from Glue import Glue
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

from TileCodeAgent import TileCodeAgent


def main():
    glue = Glue(load_model=False, num_actions=3, epsilon=0.1, alpha=0.0625)

    data = []
    for i in tqdm(range(1, 1001)):
        _, error = glue.run_episode()
        data.append(error)

        if i % 50 == 0:
            glue.save_model(f'model_v{i // 50}.npy')

    plt.plot(data)
    plt.show()

    data = []

    glue.test()
    for _ in tqdm(range(1, 51)):
        _, error = glue.run_episode()
        data.append(error)

    glue.close()

    plt.plot(data)
    plt.show()
    # agent = TileCodeAgent()
    #
    # print(agent.normalize_state(np.array([-1.2, -0.07])))
    # print(agent.normalize_state(np.array([-1.2, 0.07])))
    # print(agent.normalize_state(np.array([0.5, -0.07])))
    # print(agent.normalize_state(np.array([0.5, 0.07])))
    # agent.test_tiling()


if __name__ == '__main__':
    main()
