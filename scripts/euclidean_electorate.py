from random import gauss, seed
import numpy as np
from datetime import datetime

from policies.policy import Policy
from voters.simple_voter import SimpleVoter
from election_dynamics.electoral_systems import create_simple_electorate

if __name__ == "__main__":
    # seeds for reproducibility
    seed_val = 42
    seed(seed_val)
    np.random.seed(seed_val)
    
    # defining policies
    p1 = Policy([45,50], "Centrism") # more moderate
    p2 = Policy([80,90], "Extremism") # more extreme

    # defining voters
    voters = []
    for i in range(100):
        voters.append(SimpleVoter(Policy(np.array([gauss(50,15),gauss(50,10)]))))

    # defining electorate
    electorate = create_simple_electorate(voters, "Example Issue 1", "Example Issue 2")

    # plotting an election
    # electorate.plot_election_2d(p1, p2, verbose=True)

    # plotting a winset
    # electorate.plot_winset_boundary(p1, n_directions=360, n_halving_iterations=12)

    # plotting a path from the moderate position to the extreme one
    s_time = datetime.now()
    path = electorate.obtain_mckelvey_schofield_path(
        p1, 
        p2, 
        50, 
        step_selection_function="mckelvey_schofield_greedy_with_lookahead", 
        print_verbose=True
    )
    e_time = datetime.now()
    print(f"Path creation completed in {e_time - s_time} seconds.")
    s_time = datetime.now()
    electorate.animate_mckelvey_schofield_path(
        p1,
        p2,
        path,
        filename="euclidean_electorate_animation", 
        plot_verbose=True,
    )
    e_time = datetime.now()
    print(f"Path animation completed in {e_time - s_time} seconds.")
    s_time = datetime.now()
    electorate.plot_mckelvey_schofield_path(p1, p2, path, save_file="output/euclidean_electorate_path.png")
    e_time = datetime.now()
    print(f"Path plot completed in {e_time - s_time} seconds.")
