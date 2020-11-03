from generator_problem import GeneratorProblem
from dataclasses import dataclass
from typing import List
import itertools
import math
import time

# We define what a solution is
@dataclass
class Solution():
    assigned_generators: List[int]
    opened_generators: List[int]
    cost: int

    def get_copy(self):
        return Solution(list(self.assigned_generators), list(self.opened_generators), int(self.cost))

    def get_immutable(self):
        return (tuple(self.assigned_generators), tuple(self.opened_generators))


class Solve:

    def __init__(self, n_generator, n_device, seed):
        self.n_generator = n_generator
        self.n_device = n_device
        self.seed = seed

        self.instance = GeneratorProblem.generate_random_instance(self.n_generator, self.n_device, self.seed)

    def init_heuristic(self, time_to_run):
        ''' Sets the parameters for the heuristic search and finds a naive solution to get started '''

        # set parameters
        self.cluster_count = 1
        self.start_time = time.time()
        self.time_to_run = time_to_run

        # find and set the naive solution
        assigned_generators, opened_generators, total_cost = self.solve_naive()
        self.solution = Solution(assigned_generators, opened_generators, total_cost) # the current best solution in a local search
        self.best_solution = self.solution.get_copy() # the overall best solution

    def solve_naive(self, full_gen_index=0):
        '''Set all devices on the generator with index = full_gen_index'''
        
        opened_generators = [0 for _ in range(self.n_generator)]
        opened_generators[full_gen_index] = 1

        assigned_generators = [full_gen_index for _ in range(self.n_device)]

        self.instance.solution_checker(assigned_generators, opened_generators)
        total_cost = self.instance.get_solution_cost(assigned_generators, opened_generators)
        return assigned_generators, opened_generators, total_cost

    def solve_heuristc(self, time_to_run=30):
        ''' Solves with an heuristc algorithm in the provided time (IN SECOND) '''

        print("Solve with an heuristc algorithm")

        # set params and find a naive solution
        self.init_heuristic(time_to_run)

        while not self.time_over():

            neighbourhood = self.get_neighbourhood()
            
            self.local_search(neighbourhood)

            self.change_neighbourhood()

        self.print_solution()

    def print_solution(self, ):

        # validate the final solution
        solution = self.best_solution
        self.instance.solution_checker(solution.assigned_generators, solution.opened_generators)
        self.instance.plot_solution(solution.assigned_generators, solution.opened_generators)
        total_cost = self.instance.get_solution_cost(solution.assigned_generators, solution.opened_generators)
        
        # print the final solution
        print("[ASSIGNED-GENERATOR]", solution.assigned_generators)
        print("[OPENED-GENERATOR]", solution.opened_generators)
        print("[SOLUTION-COST]", total_cost)
        print("[FOUND IN]", time.time() - self.start_time, " s")

    def local_search(self, neighbourhood):

        # sort neighbours by cost
        neighbourhood = sorted(neighbourhood, key=lambda neighbour: neighbour.cost)

        return self.set_solution(neighbourhood[0])

    def set_solution(self, solution):
        ''' Selects the provided solution as the best one if it is validated '''

        if self.solution_is_valid(self.solution, solution):
            self.solution = solution.get_copy() # we found a new best local solution

        if self.solution_is_valid(self.best_solution, solution):
            self.best_solution = solution.get_copy() # we found a new best overall solution
            print("NEW BEST => ", self.best_solution.cost)

    def get_opened_gens(self, solution):
        ''' Returns the indexes of all opened generators '''
        opened_gens_indexes = []

        # get indexes of all opened generators
        for gen_index in range(self.n_generator):
            if solution.opened_generators[gen_index] == 1:
                opened_gens_indexes.append(gen_index)

        return opened_gens_indexes

    def distribute(self, solution):
        ''' Assigns each device to the closest opened generator '''

        opened_gens_indexes = self.get_opened_gens(solution)

        # link each device to the closest gen
        for device_index in range(self.n_device):

            closest_gen_index = 0
            closest_gen_dist = math.inf

            # find the closest gen
            for gen_index in opened_gens_indexes:

                gen_coord = self.instance.generator_coordinates[gen_index]
                device_coord = self.instance.device_coordinates[device_index]

                dist_to_gen = self.instance.get_distance(device_coord[0], device_coord[1], gen_coord[0], gen_coord[1])

                if dist_to_gen < closest_gen_dist:
                    closest_gen_dist = dist_to_gen
                    closest_gen_index = gen_index

            # link the device to the closest gen
            solution.assigned_generators[device_index] = closest_gen_index

        return solution
                
    def change_neighbourhood(self):
        ''' Adds 1 cluster '''

        if self.cluster_count < self.n_generator:
            self.cluster_count += 1

    def get_neighbourhood(self):
        ''' Builds a neighbourhood made of all possible combinations of opened generators for the current cluster count '''

        neighbourhood = []

        opened_gens_combinations = list(itertools.combinations([i for i in range(self.n_generator)], self.cluster_count))
        
        for gen_indexes in opened_gens_combinations:

            # create a new solution
            solution = self.solution.get_copy()

            # open the relevant generators
            solution.opened_generators = [1 if x in gen_indexes else 0 for x in range(self.n_generator)]

            # assign the devices to their closest generator
            solution = self.distribute(solution)

            # compute the new cost
            solution.cost = self.instance.get_solution_cost(solution.assigned_generators, solution.opened_generators)

            neighbourhood.append(solution)

            if self.time_over(): break
        
        return neighbourhood

    def solution_is_valid(sel, prev_solution, new_solution):
        return new_solution.cost < prev_solution.cost

    def time_over(self):
        return (time.time() - self.start_time) >= self.time_to_run
