
from generator_problem import GeneratorProblem
import time
import random
#10717 python main.py --n_generator=25 --n_device=100 --seed=1

class Solve:

    def __init__(self, n_generator, n_device, seed):

        self.n_generator = n_generator
        self.n_device = n_device
        self.seed = seed

        self.instance = GeneratorProblem.generate_random_instance(self.n_generator, self.n_device, self.seed)
        self.init()

    def init(self):
        ''' Find a naive initial solution '''
        assigned_generators, opened_generators, total_cost = self.solve_naive()
        self.best_opened_gens = opened_generators
        self.best_assigned_gens = assigned_generators
        self.total_cost = total_cost

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
        self.instance.plot_solution(assigned_generators, opened_generators)

        #print("[ASSIGNED-GENERATOR]", assigned_generators)
        #print("[OPENED-GENERATOR]", opened_generators)
        #print("[SOLUTION-COST]", total_cost)
        return assigned_generators, opened_generators, total_cost


    def set_best_solution(self, assigned_generators, opened_generators):
        self.best_opened_gens = opened_generators
        self.best_assigned_gens = assigned_generators
        self.total_cost = self.instance.get_solution_cost(assigned_generators, opened_generators)

    def solve_heuristc(self):

        print("Solve with an heuristc algorithm")

        ok_to_stop = False
        iterations = 900
        count = 0

        while not ok_to_stop:
            # N(s)
            start_time = time.time()
            neighbourhood = self.get_neighbourhood()
            print(((time.time() - start_time) * 1000), " ms ==> ", self.total_cost)

            best_neighbour = neighbourhood[0]

            # Q(L(N(s),s),s)
            if self.instance.get_solution_cost(best_neighbour[0], best_neighbour[1]) <= self.total_cost:
                self.set_best_solution(best_neighbour[0], best_neighbour[1])

            count += 1
            if count >= iterations: ok_to_stop = True

            # JUMP TO ANOTHER LOCAL AREA
            if not ok_to_stop: self.shake_dangerously()

        
        self.instance.solution_checker(self.best_assigned_gens, self.best_opened_gens)
        self.total_cost = self.instance.get_solution_cost(self.best_assigned_gens, self.best_opened_gens)
        self.instance.plot_solution(self.best_assigned_gens, self.best_opened_gens)
        
        #print("[ASSIGNED-GENERATOR]", self.best_assigned_gens)
        #print("[OPENED-GENERATOR]", self.best_opened_gens)
        print("[SOLUTION-COST]", self.total_cost)

    def shake_dangerously(self, device_count=1):
        unexplored_devices = list(range(self.n_device))

        for x in range(device_count):
            rand = random.Random()
            device_index = rand.choice(unexplored_devices)
            del unexplored_devices[device_index]
            generator_index = rand.randrange(self.n_generator - 1)

            self.best_assigned_gens[device_index] = generator_index
            self.best_opened_gens[generator_index] = 1


    # returns a neighbourhood from the current solution
    def get_neighbourhood(self):
        ''' N(s) = [ n ∈ S tel que pour tout device i, un generateur j connecte a i, et l'ensemble des generateurs J: j ∈ J]'''
        ''' |N(s)| = G * D avec G = n_generator et D = n_device'''

        neighbourhood = []

        for device_index in range(self.n_device):
            for gen_index in range(self.n_generator):

                if gen_index != self.best_assigned_gens[device_index]:
                    # create a new solution
                    assigned_generators = list(self.best_assigned_gens)
                    opened_generators   = list(self.best_opened_gens)

                    # set new values
                    assigned_generators[device_index] = gen_index

                    # shut down unused generators if any
                    for gen in range(self.n_generator):
                        if gen not in assigned_generators:
                            opened_generators[gen] = 0

                    # append in the neighbourhood
                    if self.solution_is_correct(assigned_generators, opened_generators):
                        neighbourhood.append([assigned_generators, opened_generators])

        return sorted(neighbourhood, key=lambda neighbour: self.instance.get_solution_cost(neighbour[0], neighbour[1]))

    # VALIDATOR
    def solution_is_valid(self, assigned_generators, opened_generators):
        ''' L(N(s),s) = [ n ∈ N(s) tel que f(n) < f(s) et  ] '''
        '''modified_generators = []
        for i in range(len(opened_generators)):

            if self.best_opened_gens[i] == opened_generators[i]:
                # no changes
                modified_generators.append(0)
            elif opened_generators[i] == 1:
                # we opened a new generator
                modified_generators.append(1)
            else:
                # we closed a generator
                modified_generators.append(-1)

        # compute the new assignments (-1 for assignments that haven't changed)
        assigned_generators = [ -1 if self.best_assigned_gens[i] == x else x for i,x in enumerate(assigned_generators)]

        new_cost = self.get_new_cost(assigned_generators, modified_generators)

        solution_is_valid = new_cost <= self.total_cost'''

        #..........
        new_cost = self.instance.get_solution_cost(assigned_generators, opened_generators)
        solution_is_valid = new_cost <= self.total_cost
        return solution_is_valid, new_cost

    # EVALUATOR
    def solution_is_correct(self, assigned_generators, opened_generators):
        ''' f(s) = all_devices_are_assigned and all_opened_generators_are_used '''
        # check if the solution respects the constraints
        try:
            self.instance.solution_checker(assigned_generators, opened_generators)
        except Exception as e:
            return False

        # check if all opened generators are used
        all_opened_gens_are_used = True
        for gen_index in range(self.n_generator):
            if opened_generators[gen_index] == 1:
                all_opened_gens_are_used = all_opened_gens_are_used and gen_index in assigned_generators

        return all_opened_gens_are_used

    def X_get_new_cost(self, assigned_generators, modified_generators):
        '''
        :param assigned_generators: list of changes applied to the assignations. A negative value indicates no changes
        :param modified_generators: integer indicating the number of generators opened or closed. A negative value indicates closing
        '''
        opening_cost_var = sum([cost * modified_gen for cost, modified_gen in zip(self.instance.opening_cost, modified_generators)])
        distance_cost_var = 0

        for i in range(self.n_device):
            
            assignation = assigned_generators[i]
            
            if assignation >= 0:
                prev_generator_coord = self.instance.generator_coordinates[self.best_assigned_gens[i]]
                new_generator_coord = self.instance.generator_coordinates[assigned_generators[i]]
                
                device_coord = self.instance.device_coordinates[i]
                
                # remove the previous distance
                distance_cost_var -= self.instance.get_distance(device_coord[0], device_coord[1], prev_generator_coord[0], prev_generator_coord[1])
                
                # add the new distance
                distance_cost_var += self.instance.get_distance(device_coord[0], device_coord[1], new_generator_coord[0], new_generator_coord[1])

        total_cost_var = distance_cost_var + opening_cost_var
        return self.total_cost + total_cost_var


'''
Fonction de validité (L): Toutes les machines ont 1 generateur et tous les generateurs allumes ont au moins 1 machine
Fonction de sélection (H): 
'''







