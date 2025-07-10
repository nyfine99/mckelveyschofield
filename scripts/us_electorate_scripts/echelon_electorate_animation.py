from random import gauss, seed
import numpy as np
from datetime import datetime

from policies.policy import Policy
from election_dynamics.electorates import create_us_electorate_echelon_sample


if __name__ == "__main__":
    # seeds for reproducibility
    seed_val = 42
    seed(seed_val)
    np.random.seed(seed_val)

    # creating electorate
    electorate = create_us_electorate_echelon_sample()

    # defining policies
    average_policy_values = [
        np.mean([v.ideal_policy.values[0] for v in electorate.voters]), 
        np.mean([v.ideal_policy.values[1] for v in electorate.voters])
    ]
    average_policy = Policy(average_policy_values, "Average Preferences")
    extreme_libertarianism = Policy([90, 10], "Extreme Libertarianism")

    # plotting a path from the moderate position to the extreme one
    print("Creating McKelvey-Schofield path.")
    s_time = datetime.now()
    path = electorate.obtain_mckelvey_schofield_path(
        average_policy, extreme_libertarianism, 50, step_selection_function="mckelvey_schofield_greedy_with_lookahead", print_verbose=False
    )
    e_time = datetime.now()
    print(f"Path created in {e_time - s_time} seconds.")

    print("Creating path animation.")
    s_time = datetime.now()
    electorate.animate_mckelvey_schofield_path(
        average_policy, extreme_libertarianism, path, filename="echelon_electorate_animation", plot_verbose=True
    )
    e_time = datetime.now()
    print(f"Path animation completed in {e_time - s_time} seconds.")

    print("Creating path plot.")
    s_time = datetime.now()
    electorate.plot_mckelvey_schofield_path(
        average_policy, extreme_libertarianism, path, save_file="output/echelon_electorate_path.png"
    )
    e_time = datetime.now()
    print(f"Path plot completed in {e_time - s_time} seconds.")
