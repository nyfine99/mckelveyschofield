from random import seed
import numpy as np
from datetime import datetime

from policies.policy import Policy
from voters.simple_voter import SimpleVoter
from voters.taxicab_voter import TaxicabVoter
from election_dynamics.electoral_systems import create_taxicab_electorate

if __name__ == "__main__":
    # seeds for reproducibility
    seed_val = 42
    seed(seed_val)
    np.random.seed(seed_val)

    # defining policies
    p1 = Policy([50,50], "Centrism") # more moderate
    p2 = Policy([90,90], "Extremism") # more extreme

    # defining voters
    voters = []
    voters.append(TaxicabVoter(Policy([40,45])))
    voters.append(TaxicabVoter(Policy([50,60])))
    voters.append(TaxicabVoter(Policy([60,45])))

    # defining electorate
    electorate = create_taxicab_electorate(voters, "Example Issue 1", "Example Issue 2")

    # plotting an election
    # electorate.plot_election_2d(p1, p2, verbose=True)

    # plotting a winset
    # electorate.plot_winset_boundary(p1, n_directions=360, n_halving_iterations=12)

    # plotting a path from the centrist position to the extreme one
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
        filename="small_taxicab_electorate_animation", 
        plot_verbose=True,
    )
    e_time = datetime.now()
    print(f"Path animation completed in {e_time - s_time} seconds.")
    s_time = datetime.now()
    electorate.plot_mckelvey_schofield_path(p1, p2, path, save_file="output/small_taxicab_electorate_path.png")
    e_time = datetime.now()
    print(f"Path plot completed in {e_time - s_time} seconds.")
