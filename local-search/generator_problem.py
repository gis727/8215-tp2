
import random
import math
import matplotlib.pyplot as plt

class GeneratorProblem:

    def __init__(self, n_generator, n_device, generator_coordinates, device_coordinates, opening_cost):

        self.n_generator = n_generator
        self.n_device = n_device
        self.opening_cost = opening_cost
        self.generator_coordinates = generator_coordinates
        self.device_coordinates = device_coordinates

    def solution_checker(self, assigned_generators, opened_generators):

        assert len(assigned_generators) == self.n_device, "Wrong solution: length of assigned_genetators does not match the number of devices"
        assert len(opened_generators) == self.n_generator, "Wrong solution: length of opened_generators does not match the number of generators"

        for generator in assigned_generators:

            assert generator >= 0, "Wrong solution: index of generator does not exist (< 0)"
            assert generator < self.n_generator, "Wrong solution: index of generator does not exist (>= n)"
            assert opened_generators[generator], "Wrong solution: assignation to a closed generator"

        for state in opened_generators:

            assert state in [0,1], "Wrong solution: value different than 0/1 in opened_generators"

    def get_distance(self, x1, y1, x2, y2):

        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_solution_cost(self, assigned_generators, opened_generators):
        '''
        :param assigned_generators: list where the element at index $i$ correspond to the generator associated to device $i$
        :param opened_generators: list where the element at index $i$ is a boolean stating if the generator $i$ is opened
        '''

        assert len(assigned_generators) == self.n_device
        assert len(opened_generators) == self.n_generator

        total_opening_cost = sum([opened_generators[i] * self.opening_cost[i] for i in range(self.n_generator)])

        total_distance_cost = 0

        for i in range(self.n_device):
            generator_coord = self.generator_coordinates[assigned_generators[i]]
            device_coord = self.device_coordinates[i]
            total_distance_cost += self.get_distance(device_coord[0], device_coord[1], generator_coord[0], generator_coord[1])

        return total_distance_cost + total_opening_cost

    def plot_solution(self, assigned_generators, opened_generators):
        '''
        :param assigned_generators: list where the element at index $i$ correspond to the generator associated to device $i$
        :param opened_generators: list where the element at index $i$ is a boolean stating if the generator $i$ is opened
        '''

        assert len(assigned_generators) == self.n_device
        assert len(opened_generators) == self.n_generator

        plt.clf()
        plt.figure(figsize=(15, 15))

        for i, gen in enumerate(assigned_generators):

            device_coord = self.device_coordinates[i]
            gen_coord = self.generator_coordinates[gen]

            plt.plot([device_coord[0], gen_coord[0]], [device_coord[1], gen_coord[1]], color="k")



        x_device = []
        y_device = []

        for i, coord in enumerate(self.device_coordinates):
            x_device.append(coord[0])
            y_device.append(coord[1])
            plt.text(coord[0], coord[1], "D%d" % i, color="b", fontsize=8)


        plt.plot(x_device, y_device, "co", color="m")

        x_gen_opened = []
        y_gen_opened = []
        x_gen_closed = []
        y_gen_closed = []

        for i, coord in enumerate(self.generator_coordinates):

            if opened_generators[i]:
                x_gen_opened.append(coord[0])
                y_gen_opened.append(coord[1])
            else:
                x_gen_closed.append(coord[0])
                y_gen_closed.append(coord[1])
            plt.text(coord[0], coord[1], "G%d" % i, color="b", fontsize=8)

        plt.plot(x_gen_opened, y_gen_opened, "co", color="g", markersize=10, marker="^")
        plt.plot(x_gen_closed, y_gen_closed, "co", color="r", markersize=10, marker="^")


        plt.savefig("solution.png")

    @staticmethod
    def generate_random_instance(n_generator, n_device, seed):

        grid_size = 100
        lb_cost, ub_cost = 200, 1000

        rand = random.Random()
        rand.seed(seed)

        x_coord_generator = [rand.uniform(0, grid_size) for _ in range(n_generator)]
        y_coord_generator = [rand.uniform(0, grid_size) for _ in range(n_generator)]

        x_coord_device = [rand.uniform(0, grid_size) for _ in range(n_device)]
        y_coord_device = [rand.uniform(0, grid_size) for _ in range(n_device)]

        generator_coordinates = list(zip(x_coord_generator, y_coord_generator))
        device_coordinates = list(zip(x_coord_device, y_coord_device))

        opening_cost = [rand.uniform(lb_cost, ub_cost) for _ in range(n_generator)]

        return GeneratorProblem(n_generator, n_device, generator_coordinates, device_coordinates, opening_cost)










