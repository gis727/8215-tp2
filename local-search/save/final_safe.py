
from generator_problem import GeneratorProblem
from typing import List
from dataclasses import dataclass
from collections import deque
import time
import random
import sys
#9821.300297539407 python main.py --n_generator=25 --n_device=100 --seed=1
#34  /  748
# We define what a solution is
@dataclass
class Solution():
    assigned_generators: List[int]
    opened_generators: List[int]
    cost: int

    def get_copy(self):
        return Solution(list(self.assigned_generators), list(self.opened_generators), int(self.cost))



class Solve:

    def __init__(self, n_generator, n_device, seed):

        self.n_generator = n_generator
        self.n_device = n_device
        self.seed = seed

        self.instance = GeneratorProblem.generate_random_instance(self.n_generator, self.n_device, self.seed)

    def init_heuristic(self, tabu_list_length, run_time_in_seconds):
        # set parameters
        self.tabu = deque(maxlen=tabu_list_length)
        self.run_time = run_time_in_seconds

        self.tabu_denials = 0
        self.requests = 0

        # find and set the naive solution
        assigned_generators, opened_generators, total_cost = self.solve_naive()
        self.solution = Solution(assigned_generators, opened_generators, total_cost)

    def solve_naive(self):
        opened_generators = [1 for _ in range(self.n_generator)]

        assigned_generators = [None for _ in range(self.n_device)]

        for i in range(self.n_device):
            closest_generator = min(range(self.n_generator),
                                    key=lambda j: self.instance.get_distance(self.instance.device_coordinates[i][0],
                                                                      self.instance.device_coordinates[i][1],
                                                                      self.instance.generator_coordinates[j][0],
                                                                      self.instance.generator_coordinates[j][1])
                                    )

            assigned_generators[i] = closest_generator

        self.instance.solution_checker(assigned_generators, opened_generators)
        total_cost = self.instance.get_solution_cost(assigned_generators, opened_generators)
        return assigned_generators, opened_generators, total_cost

    def solve_heuristc(self, tabu_list_length=200000, run_time_in_seconds=150):

        print("Solve with an heuristc algorithm")

        # set params and find a naive solution
        self.init_heuristic(tabu_list_length, run_time_in_seconds)

        time_over = False
        start_time = time.time()

        while not time_over:

            # get the neighbourhood
            neighbourhood = self.get_neighbourhood()

            # search a local optimum
            self.local_search(neighbourhood)

            # check if time is over
            time_over = (time.time() - start_time) >= self.run_time

            # jump to another neighbourhood
            if not time_over:
                self.jump_to_random_neighbourhood()

        # validate and show the final solution
        self.print_solution()

    def print_solution(self):

        # validate the final solution
        self.instance.solution_checker(self.solution.assigned_generators, self.solution.opened_generators)
        total_cost = self.instance.get_solution_cost(self.solution.assigned_generators, self.solution.opened_generators)
        self.instance.plot_solution(self.solution.assigned_generators, self.solution.opened_generators)
        
        # print the final solution
        #print("[ASSIGNED-GENERATOR]", self.best_assigned_gens)
        #print("[OPENED-GENERATOR]", self.best_opened_gens)
        print("[SOLUTION-COST]", total_cost)
        print("TABU ===> ", self.tabu_denials, " / ", self.requests)

    def local_search(self, neighbourhood):

        # sort neighbours by cost
        neighbourhood = sorted(neighbourhood, key=lambda neighbour: neighbour.cost)

        for neighbour in neighbourhood:

            solution_is_valid = self.solution_is_valid(neighbour)

            if solution_is_valid:
                self.set_best_solution(neighbour)
                break

    def set_best_solution(self, solution: Solution):
        self.solution = solution
        print("==> ", self.solution.cost, "----",sys.getsizeof(self.tabu)*0.000001)

    def jump_to_random_neighbourhood(self):

        new_solution = self.solution
        solution_is_tabu = True

        i = 0
        self.requests += 1

        while solution_is_tabu:

            if i > 0: self.tabu_denials += 1
            i += 1

            new_solution = self.get_random_solution()
            solution_is_tabu = self.solution_is_tabu(new_solution)
        self.solution = new_solution

    def solution_is_tabu(self, solution):
        return self.tabu.count(solution) > 0

    def get_random_solution(self, device_count=1):

        '''Randomly generates a new solution'''

        solution = self.solution.get_copy()

        unexplored_devices = list(range(self.n_device))

        for x in range(device_count):
            rand = random.Random()
            device_index = rand.choice(unexplored_devices)
            del unexplored_devices[device_index]
            generator_index = rand.randrange(self.n_generator - 1)

            solution.assigned_generators[device_index] = generator_index
            solution.opened_generators[generator_index] = 1

        # update the solution cost
        solution.cost = self.instance.get_solution_cost(solution.assigned_generators, solution.opened_generators)

        return solution

    # NEIGHBOURHOOD
    def get_neighbourhood(self):
        ''' N(s) = [ n ∈ S tel que pour tout device i, un generateur j connecte a i, et l'ensemble des generateurs J: j ∈ J]'''
        ''' |N(s)| = G * D avec G = n_generator et D = n_device'''

        neighbourhood = []

        for device_index in range(self.n_device):
            for gen_index in range(self.n_generator):

                if gen_index != self.solution.assigned_generators[device_index]:
                    # create a new solution
                    solution = self.solution.get_copy()

                    # set new values
                    solution.assigned_generators[device_index] = gen_index

                    # shut down unused generators if any
                    for gen in range(self.n_generator):
                        if gen not in solution.assigned_generators:
                            solution.opened_generators[gen] = 0

                    # append in the neighbourhood
                    if self.solution_is_correct(solution):
                        solution.cost = self.instance.get_solution_cost(solution.assigned_generators, solution.opened_generators)
                        neighbourhood.append(solution)
                        self.tabu.append(solution)

        return neighbourhood

    # VALIDATOR
    def solution_is_valid(self, solution):
        return solution.cost < self.solution.cost

    # EVALUATOR
    def solution_is_correct(self, solution):
        ''' Returns True if all devices are assigned and all opened generators are used '''

        # check if the all devices are assigned
        try:
            self.instance.solution_checker(solution.assigned_generators, solution.opened_generators)
        except Exception as e:
            return False

        # check if all opened generators are used
        all_opened_gens_are_used = True
        for gen_index in range(self.n_generator):
            if solution.opened_generators[gen_index] == 1:
                all_opened_gens_are_used = all_opened_gens_are_used and gen_index in solution.assigned_generators

        return all_opened_gens_are_used


'''
Fonction de validité (L): Toutes les machines ont 1 generateur et tous les generateurs allumes ont au moins 1 machine
Fonction de sélection (H): 
'''
