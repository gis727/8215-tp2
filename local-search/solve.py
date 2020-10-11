
from generator_problem import GeneratorProblem
from typing import List
from dataclasses import dataclass
from collections import deque
import time
import random
import sys
#9521.123214669946 python main.py --n_generator=25 --n_device=100 --seed=1
#34  /  748

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

    def init_heuristic(self, tabu_len, run_time_sec):
        # set parameters
        self.tabu_len = tabu_len
        self.tabu = set()
        self.run_time = run_time_sec

        # performances params (metaheuristic)
        self.rand_d_count = 1 # number of devices to re-assign during the random selection in the ILS
        self.in_bad_iter = False # True if the previous iteration landed a result worst than the current best solution
        self.bad_iter_count = 0 # number of consecutive iterations that landed a result worst than the current best solution

        # find and set the naive solution
        assigned_generators, opened_generators, total_cost = self.solve_naive()
        self.solution = Solution(assigned_generators, opened_generators, total_cost) # the current best solution in a local search
        self.best_solution = self.solution.get_copy() # the overall best solution

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

    def solve_heuristc(self, tabu_len=8000, run_time_sec=60):

        print("Solve with an heuristc algorithm")

        # set params and find a naive solution
        self.init_heuristic(tabu_len, run_time_sec)

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
        self.instance.solution_checker(self.best_solution.assigned_generators, self.best_solution.opened_generators)
        total_cost = self.instance.get_solution_cost(self.best_solution.assigned_generators, self.best_solution.opened_generators)
        self.instance.plot_solution(self.best_solution.assigned_generators, self.best_solution.opened_generators)
        
        # print the final solution
        #print("[ASSIGNED-GENERATOR]", self.best_assigned_gens)
        #print("[OPENED-GENERATOR]", self.best_solution.opened_generators)
        print("[SOLUTION-COST]", total_cost)

    def local_search(self, neighbourhood):

        # sort neighbours by cost
        neighbourhood = sorted(neighbourhood, key=lambda neighbour: neighbour.cost)
        selected_neighbour = None

        for neighbour in neighbourhood:
            self.add_tabu(neighbour)

            previous_best_solution = self.best_solution.get_copy()

            if self.set_solution(neighbour):
                selected_neighbour = neighbour
                break

        self.update_perf(previous_best_solution, selected_neighbour)

    def set_solution(self, solution, force=False):
        '''If force=True, then the new solution will be set as the best local solution'''

        solution_is_valid = False

        if self.solution_is_valid(self.solution, solution) or force:
            self.solution = solution.get_copy()
            solution_is_valid = True

        if self.solution_is_valid(self.best_solution, solution):
            self.best_solution = solution.get_copy()
            print("BEST => ", self.best_solution.cost)

        return solution_is_valid

    def update_perf(self, previous_best_solution, solution=None):
        ''' Updates the algorithm performance stats'''

        treshold_iter_counts = [5,15,25,30,40,50] # the numbers of iteration after which we increase the devices relinks
        bad_iter_limit       = treshold_iter_counts.pop() # the number of bad iterations after which we restart counting
        extra_devices        = int(self.bad_iter_count/5) # the number of devices to relink
        extra_devices_limit  = int(self.n_device / 6)

        def reset_bad_iter_count():
            self.in_bad_iter    = False
            self.bad_iter_count = 0

        # update the bad iterations count
        if solution and self.solution_is_valid(previous_best_solution, solution):
            reset_bad_iter_count()
        else:
            if self.in_bad_iter:
                self.bad_iter_count += 1
            else:
                self.in_bad_iter = True
        
        # TO REM
        #if(solution): print(self.rand_d_count, " ===> ", self.best_solution.cost, " | ", not self.solution_is_valid(previous_best_solution, solution), " - ", len(self.tabu))

        # update the number of devices to be relinked for the next iteration
        if self.bad_iter_count in treshold_iter_counts:
            self.rand_d_count += extra_devices

        elif self.bad_iter_count > bad_iter_limit:
            self.rand_d_count   = extra_devices_limit
            reset_bad_iter_count()
        else:
            self.rand_d_count = self.rand_d_count - 1 if self.rand_d_count > 1 else 1

    def jump_to_random_neighbourhood(self):

        new_solution = self.best_solution
        solution_is_tabu = True

        while solution_is_tabu:

            new_solution = self.get_random_solution()
            solution_is_tabu = self.solution_is_tabu(new_solution)

        self.set_solution(new_solution, force=True)

    def get_random_solution(self, device_count=5):

        '''Generates a new solution by assigning a random generator to "self.rand_d_count" devices selected randomly'''

        solution = self.best_solution.get_copy()
        rand = random.Random()

        unexplored_devices = list(range(self.n_device))

        for x in range(self.rand_d_count):

            # select a random device and a random generator
            device_index = rand.choice(unexplored_devices)
            unexplored_devices.remove(device_index)
            generator_index = rand.randrange(self.n_generator - 1)

            # link them together
            solution.assigned_generators[device_index] = generator_index

            # make sure the generator is powered up
            solution.opened_generators[generator_index] = 1

        # update the solution cost
        solution.cost = self.instance.get_solution_cost(solution.assigned_generators, solution.opened_generators)

        return solution

    def get_neighbourhood(self):
        ''' N(s) = [ n ∈ S tel que pour tout device i, un generateur j connecte a i, et l'ensemble des generateurs J: j ∈ J]'''
        ''' |N(s)| = G * D avec G = n_generator et D = n_device'''

        neighbourhood = []

        for device_index in range(self.n_device):

            for gen_index in range(self.n_generator):

                if gen_index != self.solution.assigned_generators[device_index]:
                    # create a new solution
                    solution = self.solution.get_copy()
                    changed_gen_indexes = [] # the index(es) of the affected generator(s)
                    changed_gen_states = [] # whether the affected generator(s) were turned down (-1) or powered up (+1)

                    # save previous generator index
                    previous_gen = solution.assigned_generators[device_index]

                    # set the new assignation for the device
                    solution.assigned_generators[device_index] = gen_index

                    # update the affected generator
                    if solution.opened_generators[gen_index] != 1: # the affected generator must be started
                        solution.opened_generators[gen_index] = 1
                        changed_gen_indexes.append(gen_index)
                        changed_gen_states.append(1)

                    # shut down the previous generator if its now unused
                    if previous_gen not in solution.assigned_generators:
                        solution.opened_generators[previous_gen] = 0
                        changed_gen_indexes.append(previous_gen)
                        changed_gen_states.append(-1)

                    # compute the new cost
                    solution.cost = self.get_new_cost(self.solution, solution, device_index, changed_gen_indexes, changed_gen_states)
                    
                    # append in the neighbourhood
                    neighbourhood.append(solution)

        return neighbourhood

    def solution_is_valid(sel, prev_solution, new_solution):
        return new_solution.cost < prev_solution.cost

    def get_new_cost(self, prev_solution, new_solution, device_index, gen_indexes, gen_states):
        '''
        Returns the cost of "new_solution" by computing the costs variations from "prev_solution".
        Note:
        - this method is a replacement for "GeneratorProblem.get_solution_cost" since it is approx 10 times faster
        - the resulting cost has an error of approx 1e-12 wich remains acceptable for the current problem
        '''

        # get the coordinates of the device and the affected generators
        prev_generator_coord = self.instance.generator_coordinates[prev_solution.assigned_generators[device_index]]
        new_generator_coord = self.instance.generator_coordinates[new_solution.assigned_generators[device_index]]
        device_coord = self.instance.device_coordinates[device_index]

        distance_cost_var = 0.0 # cost variation due to the distances between the device and the new/prev generators
        
        # add the new distance
        distance_cost_var += self.instance.get_distance(device_coord[0], device_coord[1], new_generator_coord[0], new_generator_coord[1])

        # remove the previous distance
        distance_cost_var -= self.instance.get_distance(device_coord[0], device_coord[1], prev_generator_coord[0], prev_generator_coord[1])

        # compute the cost variation due to the opening/closing of the affected generators
        opening_cost_var = sum([self.instance.opening_cost[gen_indexes[x]] * gen_states[x] for x in range(len(gen_indexes))])

        return prev_solution.cost + (distance_cost_var + opening_cost_var)

    def solution_is_tabu(self, solution):
        return solution.get_immutable() in self.tabu

    def add_tabu(self, solution):
        if not self.solution_is_tabu(solution):

            self.tabu.add(solution.get_immutable())

            # make sure the tabu list length is limited
            if len(self.tabu) > self.tabu_len:
                self.tabu.pop()


'''
Fonction de validité (L): Toutes les machines ont 1 generateur et tous les generateurs allumes ont au moins 1 machine
Fonction de sélection (H): 
'''
