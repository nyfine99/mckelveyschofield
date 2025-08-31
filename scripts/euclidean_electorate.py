"""
Euclidean Electorate Simulation Script

This script demonstrates the core capabilities of the McKelvey-Schofield project by:
1. Creating a simulated electorate with voters distributed around policy preferences
2. Computing and visualizing a winset boundary (policies that can defeat incumbents)
3. Finding a McKelvey-Schofield path between policy positions
4. Generating animations and plots of electoral dynamics

The script serves as a comprehensive example of how to use the project's
electoral simulation capabilities with Euclidean distance-based voter utilities.
"""

from random import gauss, seed
import numpy as np
from datetime import datetime

from policies.policy import Policy
from voters.simple_voter import SimpleVoter
from election_dynamics.electoral_systems import create_simple_electorate


def main():
    """
    Main demonstration function showcasing electoral dynamics simulation.

    The simulation creates a centrist vs. extremist policy scenario to
    demonstrate how an agenda setter can manipulate electoral outcomes
    through strategic policy sequencing.
    """
    
    # set seeds for reproducible results - important for research and debugging
    seed_val = 42
    seed(seed_val)
    np.random.seed(seed_val)
    print(f"Using seed value: {seed_val} for reproducible results.")
    
    # create policy positions for demonstration
    # these represent different ideological positions in 2D policy space
    p1 = Policy([45, 50], "Centrism")   # moderate position on both issues
    p2 = Policy([80, 90], "Extremism")  # more extreme position on both issues
    
    print(f"Created policies:")
    print(f"  - {p1.name}: {p1.values} (moderate)")
    print(f"  - {p2.name}: {p2.values} (extreme)")
    
    # generate a simple electorate with 100 voters
    # voters are distributed around the center with some variation
    # this simulates a diverse but generally centrist electorate
    voters = []
    for i in range(100):
        # create voters with preferences distributed around the center
        # Issue 1: mean=50, std=15 (economic policy)
        # Issue 2: mean=50, std=10 (social policy)
        voter_preferences = np.array([
            gauss(50, 15),  # economic policy preference
            gauss(50, 10)   # social policy preference
        ])
        voters.append(SimpleVoter(Policy(voter_preferences)))
    
    print(f"Created electorate with {len(voters)} voters.")
    print("Voter preferences distributed around (50,50) with moderate variation.")
    
    # create the electorate simulation object
    # this encapsulates all the electoral dynamics and provides methods
    # for analysis and visualization
    electorate = create_simple_electorate(
        voters, 
        "Economic Policy",  # label for x-axis
        "Social Policy"     # label for y-axis
    )
    
    # note: Election plotting is commented out but available for debugging
    # electorate.plot_election_2d(p1, p2, verbose=True)
    
    # compute and visualize the winset boundary
    # the winset shows all policies that can defeat the centrist incumbent
    # this demonstrates the power of agenda setting in electoral systems
    print("\nComputing winset boundary for centrist policy...")
    electorate.plot_winset_boundary(
        p1,  # status quo policy (centrist)
        n_directions=360,           # number of directions to search (full circle)
        n_halving_iterations=12,    # precision of boundary computation
        output_folder="output",     # where to save the plot
        filename="euclidean_electorate_centrism_winset_boundary.png"
    )
    print("Winset boundary computed and saved.")
    
    # find a McKelvey-Schofield path from centrist to extreme policy
    # this demonstrates how agenda setters can manipulate outcomes through
    # strategic sequencing of binary choices
    print("\nFinding McKelvey-Schofield path from centrist to extreme...")
    
    s_time = datetime.now()
    path = electorate.obtain_mckelvey_schofield_path(
        p1,  # starting policy (centrist)
        p2,  # target policy (extreme)
        50,  # maximum number of steps allowed in path
        step_selection_function="mckelvey_schofield_greedy_with_lookahead", 
        print_verbose=False
    )
    e_time = datetime.now()
    print(f"Path creation completed in {e_time - s_time} seconds.")
    
    if path:
        print(f"Found path with {len(path)} steps.")
        
        # create animation showing the path through policy space
        # this visualizes how the agenda setter moves through the space
        print("\nCreating path animation...")
        s_time = datetime.now()
        electorate.animate_mckelvey_schofield_path(
            p1,  # starting policy
            p2,  # ending policy
            path, # computed path
            filename="euclidean_electorate_animation", 
            plot_verbose=True,
        )
        e_time = datetime.now()
        print(f"Path animation completed in {e_time - s_time} seconds.")
        
        # create static plot of the path
        # this shows the strategic route through policy space
        print("\nCreating static path plot...")
        s_time = datetime.now()
        electorate.plot_mckelvey_schofield_path(
            p1, p2, path, 
            save_file="output/euclidean_electorate_path.png"
        )
        e_time = datetime.now()
        print(f"Path plot completed in {e_time - s_time} seconds.")
        
        print("\nSimulation completed successfully!")
        print("Check the 'output' folder for generated visualizations.")
        
    else:
        print("No path found - this can happen with certain voter distributions")
        print("Try adjusting the voter distribution or policy positions")


if __name__ == "__main__":
    # run the main demonstration when script is executed directly
    main()
