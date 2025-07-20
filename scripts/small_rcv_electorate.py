from random import gauss, seed
import numpy as np
from datetime import datetime

from policies.policy import Policy
from voters.simple_voter import SimpleVoter
from election_dynamics.electoral_systems import create_rcv_electorate

if __name__ == "__main__":
    # seeds for reproducibility
    seed_val = 100
    seed(seed_val)
    np.random.seed(seed_val)
    
    # defining policies
    policies_list = [
        Policy([40,50], "Left-of-center"),
        Policy([60,50], "Right-of-center"),
        Policy([50,50], "Centrism"),
    ]

    # defining voters
    voters = []
    for i in range(25):
        voters.append(SimpleVoter(Policy(np.array([gauss(50,15),gauss(50,10)]))))

    # defining electorate
    rcv_electorate = create_rcv_electorate(voters, "Example Issue 1", "Example Issue 2")

    # animating an election
    rcv_electorate.animate_election(
        policies_list, 
        stop_at_majority=True, 
        output_folder="output",
        filename="small_rcv_electorate_animation",
    )

    # creating a sankey diagram of an election
    rcv_electorate.create_election_sankey_diagram(
        policies_list, 
        stop_at_majority=True, 
        output_folder="output",
        filename="small_rcv_electorate_sankey",
    )
