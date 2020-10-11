
from solve import Solve


import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--n_generator', type=int, default=25)
    parser.add_argument('--n_device', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    print("***********************************************************")
    print("[INFO] EVALUATING THE GENERATOR PROBLEM")
    print("[INFO] n_generator: %d" % args.n_generator)
    print("[INFO] n_device': %d" % args.n_device)
    print("[INFO] seed: %s" % args.seed)
    print("***********************************************************")

    solveur = Solve(args.n_generator, args.n_device, args.seed)

    solveur.solve_heuristc()
