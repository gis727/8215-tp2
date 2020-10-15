
from generator_problem import GeneratorProblem
import random


class Solve:

    def __init__(self, n_generator, n_device, seed):

        self.n_generator = n_generator
        self.n_device = n_device
        self.seed = seed

        self.instance = GeneratorProblem.generate_random_instance(self.n_generator, self.n_device, self.seed)

    def solve_naive(self):

        print("Solve with a naive algorithm")
        print("All the generators are opened, and the devices are associated to the closest one")

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

        print("[ASSIGNED-GENERATOR]", assigned_generators)
        print("[OPENED-GENERATOR]", opened_generators)
        print("[SOLUTION-COST]", total_cost)
        
    ###===========================================================================
    #== This function associates generators to devices according proximity with respect 
    #== a list of generators y with 0 or 1 (1: opened, 0: closed)
    
    def distribute(self, y, w):   
        y_temp = list(y)
        
        if w >= 0:
            y_temp[w] = 1 - y_temp[w] 
        
        assigned_generators = [None for _ in range(self.n_device)]
        
        openGen = []
        for i in range(len(y_temp)):
            if y_temp[i] != 0:
                openGen.append(i)
        
        for i in range(self.n_device):
            closest_generator = min(openGen,
                                    key=lambda j: self.instance.get_distance(self.instance.device_coordinates[i][0],
                                                                      self.instance.device_coordinates[i][1],
                                                                      self.instance.generator_coordinates[j][0],
                                                                      self.instance.generator_coordinates[j][1]))

            assigned_generators[i] = closest_generator
            
        
        total_cost = self.instance.get_solution_cost(assigned_generators, y_temp)
        
        return {"assigned_generators": assigned_generators, "total_cost": total_cost}
    
    #####
    # It defines bestGain, bestFlips according the article
    #####
    def best_(self, s_star, complement_tabu, best): 
        
        total_costs = [self.distribute(s_star, w)["total_cost"] for w in complement_tabu]
        bestObj = min(total_costs)
        bestFlips = [complement_tabu[index] for index, value in enumerate(total_costs) if value == bestObj]
        
        return {"bestGain": best - bestObj, "bestFlips": bestFlips}
        
        
    
    def solve_tabu(self):
        
        ## Set up
        #=== - Initial solution: All generators are opened as in naive solution
        #=== - nbStableMax: It is max number of iterations to do if objective function
        #=== has not improved  
        #=== - tLen: length of tabu length
        #=== - best: best objective
         
        y = [1 for _ in range(self.n_generator)] # generators/ y[i] = 1 if opened and 0 otherwise
        s_star = list(y) # initial solution
        nbStableMax = 1000 
        tLen = 10
        obj = self.distribute(s_star, -1)["total_cost"]
        best = obj
        tabu = [0]*len(y)
        nbStable = 0
        it = 0 # for iterations
        y_best = []
        while nbStable < nbStableMax:  
          
            old = best
            
            complement_tabu = []
            for i in range(len(s_star)):
                if tabu[i] == 0 and s_star[i] != 0:
                    complement_tabu.append(i)
                    
            if (len(complement_tabu) > 1): 
                resultTemp = self.best_(s_star, complement_tabu, best)
                if resultTemp["bestGain"] >= 0 and (len(resultTemp["bestFlips"]) > 1):
                    w = random.choice(resultTemp["bestFlips"])
                    y[w] = 1 - y[w]
                    obj = self.distribute(y, -1)["total_cost"]
                    tabu[w] = it + tLen
                    if obj < old and tLen > 2:
                        tLen -= 1
                    if obj >= old and tLen < 10:
                        tLen  += 1
                    it += 1
                else:
                    openY = []
                    for i in range(len(y)):
                        if y[i] != 0:
                            openY.append(i)
                    if len(openY) > 1:                    
                        w = random.choice(openY)
                        y[w] = 0
                        obj = self.distribute(y, -1)["total_cost"]
            
            if obj < best:
                best = obj                
                y_best = list(y)
                s_star = list(y)
                nbStable = 0
            else:
                nbStable += 1
                
            for w in range(len(tabu)):
                if tabu[w] < it and tabu[w] != 0:
                    tabu[w] = 0
        
      
        
        resFinal1 = self.distribute(y_best, -1)
        assigned_generators1 = resFinal1["assigned_generators"]               
                
        self.instance.solution_checker(assigned_generators1, y_best)
        total_cost1 = self.instance.get_solution_cost(assigned_generators1, y_best)
        self.instance.plot_solution(assigned_generators1, y_best)
                
        print("[ASSIGNED-GENERATOR]", assigned_generators1)
        print("[OPENED-GENERATOR]", y_best)  
        print("total cost", total_cost1)
        
        print("[SOLUTION-COSTBesttt]", best)
        

