from solve import Solve
import sys, os
import argparse


# HELPERS
def hide_prints():
    sys.stdout = open(os.devnull, 'w')

def restore_prints():
    sys.stdout = sys.__stdout__

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_prints', type=int, default=1)
    return parser.parse_args()



# SOLVER EXECUTOR
def exec_solver():
    if args().no_prints: hide_prints()
    
    solveur = Solve(n_generator=25, n_device=100, seed=seed)

    # la methode que vous avez codé (solve_heuristc pour moi) doit retourner le meilleur score trouvé
    score = solveur.solve_heuristc()

    restore_prints()

    return score



# MAIN
if __name__ == '__main__':

    seeds = [1,9,19,29,39] # les correcteurs vont choisir des seed au hasard
    scores = []

    for i,seed in enumerate(seeds):
        print("Executing solver on seed", seed, "...")

        
        score = int(exec_solver())
        scores.append(score)

        print("Score =", score)
    
    print("Scores ==> ", scores)