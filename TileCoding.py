import numpy as np

import matplotlib.pyplot as plt


class TileCode:
    def __init__(self, tiles, count):
        self.shape = np.array(tiles)
        self.coefficients = 1.0 / self.shape

        self.single_tile_count = np.product(self.shape)
        self.tile_count = count * self.single_tile_count

        self.count = count
        self.offset = (count - 1) / (count * self.single_tile_count)

        self.mask = self.generate_mask()

    def generate_mask(self):
        mask = []

        last_product = 1
        for i in self.shape:
            mask.append(last_product)
            last_product *= i

        return np.array(mask)

    def get_index(self, x):
        mask = np.dot(self.mask, x)

        index = np.sum(mask)
        return index

    def get_tile(self, x):
        tiles = []

        for i in range(self.count):
            index = ((x - self.offset * i) // self.coefficients).astype(np.int32)
            index = self.get_index(index)

            # if index >= 0:
            tiles.append(index)

        return np.array(tiles)

    def get_feature_vector(self, x):
        active_tiles = self.get_tile(x)

        stack = []
        for tile in active_tiles:
            one_hot = np.zeros(self.single_tile_count)

            one_hot[tile] = 1
            stack.append(one_hot)

        feature_vector = np.hstack(stack)
        return feature_vector

    def plot_tiles(self, points=[]):
        x_count, y_count = self.shape
        x_distance, y_distance = self.coefficients

        x = np.linspace(0, 1, 1000)
        y = np.linspace(0, 1, 1000)
        ones = np.ones_like(x)

        # plt.plot(x, ones, color='black')
        # plt.plot(ones, y, color='black')
        # plt.plot(ones * 0, y, color='black')
        # plt.plot(x, ones * 0, color='black')

        def plot_rectangle(offset, color):
            for k in range(x_count + 1):
                plt.plot(ones * 0 + x_distance * k + offset, y + offset, color=color)
            for j in range(y_count + 1):
                plt.plot(x + offset, ones * 0 + y_distance * j + offset, color=color)

        colors = ['black', 'blue', 'red', 'green', 'brown', 'pink', 'yellow', 'gray', 'chartreuse']

        for i in range(self.count):
            plot_rectangle(i * self.offset, colors[i])

        for point in points:
            plt.scatter(point[0], point[1])

        plt.show()

        for index, point in enumerate(points):
            for i in range(self.count):
                plot_rectangle(i * self.offset, 'black')
                plt.scatter(point[0], point[1], color=colors[index])
                plt.show()
