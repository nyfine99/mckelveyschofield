from random import gauss, seed
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from policies.policy import Policy
from voters.simple_voter import SimpleVoter
from election_dynamics.electoral_systems import create_simple_electorate

if __name__ == "__main__":
    
    n_sims = 100  # Number of simulations to run
    max_steps = 50  # Maximum number of steps to take in each simulation
    
    # Initialize counters and data storage
    successful_runs = 0
    input_data = []
    output_data = []

    # setting up directories for output
    multirun_folder = f'output/multirun_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'  # Base folder for multirun outputs
    if not os.path.exists(multirun_folder):
        os.makedirs(multirun_folder)
        os.makedirs(f"{multirun_folder}/animations")
        os.makedirs(f"{multirun_folder}/path_plots")
    
    for run_num in range(1,n_sims+1):
        seed_val = run_num  # Using run_num as the seed for reproducibility
        seed(seed_val)
        print(f"Running simulation {run_num} of {n_sims}...")

        # defining policies
        p1 = Policy([50,50], "Centrism")  # more moderate
        p2 = Policy([80,90], "Extremism")  # more extreme

        # defining voters
        voters = []
        for i in range(100):
            voters.append(SimpleVoter(Policy([gauss(50,15),gauss(50,10)])))

        # defining electorate
        electorate = create_simple_electorate(voters, "Example Issue 1", "Example Issue 2")

        # attempting to create a path from the moderate position to the extreme one
        # animating the path
        path = electorate.animate_mckelvey_schofield(
            p1,
            p2,
            max_steps,
            output_folder=f"{multirun_folder}/animations", 
            filename=f"run_{run_num}",
            fps=1,
            verbose=False
        )
        if len(path) == max_steps + 1 and path[max_steps].values != p2.values:
            print(f"Simulation {run_num} failed to reach the goal policy.")
        else:
            print(f"Simulation {run_num} successfully reached the goal policy in {len(path)-1} steps.")
            successful_runs += 1

        # plotting the path
        save_file = f"{multirun_folder}/path_plots/run_{run_num}.png"
        electorate.plot_mckelvey_schofield_path(p1, p2, path, save_file)

        # data for csvs
        input_data_to_append = {
            'run_num': run_num,
            'seed': seed_val,
            'original_policy_values': p1.values,
            'goal_policy_values': p2.values,
            'voter_count': len(electorate.voters),
            'max_steps': max_steps,
            'mean_voter_preferences': np.mean([voter.ideal_policy.values for voter in electorate.voters], axis=0).tolist(),
            'std_voter_preferences': np.std([voter.ideal_policy.values for voter in electorate.voters], axis=0).tolist(),
        }
        for voter_idx in range(len(electorate.voters)):
            input_data_to_append[f'voter_{voter_idx}_preferences'] = electorate.voters[voter_idx].ideal_policy.values
        input_data.append(input_data_to_append)

        output_data_to_append = {
            'run_num': run_num,
            'num_steps': len(path)-1,  # Exclude the initial policy
            'final_policy_values': path[-1].values,
            'reached_goal_policy': path[-1].values == p2.values,
            'path_taken': [p.values for p in path],
        }
        output_data.append(output_data_to_append)
    
    print(f"Successful runs: {successful_runs} out of {n_sims}")
    print(f"Success rate: {successful_runs / n_sims * 100:.2f}%")

    # saving input data to csv
    input_df = pd.DataFrame(input_data)
    input_df.to_csv(f"{multirun_folder}/input_data.csv", index=False)

    # saving output data to csv
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(f"{multirun_folder}/output_data.csv", index=False)

    # saving the summary data
    summary_data = {
        'total_runs': n_sims,
        'successful_runs': successful_runs,
        'success_rate': successful_runs / n_sims * 100,
        'mean_steps_in_successful_runs': output_df[output_df['reached_goal_policy']]['num_steps'].mean() if successful_runs > 0 else 0,
        'std_steps_in_successful_runs': output_df[output_df['reached_goal_policy']]['num_steps'].std() if successful_runs > 0 else 0,
        'min_steps_in_successful_runs': output_df[output_df['reached_goal_policy']]['num_steps'].min() if successful_runs > 0 else 0,
        'max_steps_in_successful_runs': output_df[output_df['reached_goal_policy']]['num_steps'].max() if successful_runs > 0 else 0,
    }
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(f"{multirun_folder}/summary_data.csv", index=False)
