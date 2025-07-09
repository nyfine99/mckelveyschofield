from random import gauss, seed
import numpy as np
from datetime import datetime

from policies.policy import Policy
from voters.simple_voter import SimpleVoter
from election_dynamics.electoral_systems import create_simple_electorate

if __name__ == "__main__":
    # defining policies
    p1 = Policy([50,50], "Centrism") # more moderate
    p2 = Policy([80,90], "Extremism") # more extreme

    # defining voters
    seed(42)  # For reproducibility
    voters = []
    for i in range(50):
        voters.append(SimpleVoter(Policy(np.array([gauss(25,5),gauss(55,3)]))))
        voters.append(SimpleVoter(Policy(np.array([gauss(75,5),gauss(45,3)]))))

    # defining electorate
    electorate = create_simple_electorate(voters, "Example Issue 1", "Example Issue 2")

    # plotting a path from the moderate position to the extreme one
    print("Creating McKelvey-Schofield path.")
    s_time = datetime.now()
    path = electorate.obtain_mckelvey_schofield_path(
        p1, p2, 50, step_selection_function="mckelvey_schofield_greedy_with_lookahead", print_verbose=False
    )
    e_time = datetime.now()
    print(f"Path created in {e_time - s_time} seconds.")

    print("Creating path animation.")
    s_time = datetime.now()
    electorate.animate_mckelvey_schofield_path(p1, p2, path, filename="polarized_electorate_animation", plot_verbose=True)
    e_time = datetime.now()
    print(f"Path animation completed in {e_time - s_time} seconds.")

    print("Creating path plot.")
    s_time = datetime.now()
    electorate.plot_mckelvey_schofield_path(p1, p2, path, save_file="output/polarized_electorate_path.png")
    e_time = datetime.now()
    print(f"Path plot completed in {e_time - s_time} seconds.")
