from random import uniform, seed
import numpy as np

from policies.policy import Policy
from voters.simple_voter import SimpleVoter
from election_dynamics.electoral_systems import create_simple_electorate

if __name__ == "__main__":
    # defining policies
    p1 = Policy([30,50], "Left-wing") # more moderate
    p2 = Policy([70,50], "Right-wing") # more extreme

    # defining voters
    seed(42)  # For reproducibility
    voters = []
    for i in range(250):
        voters.append(SimpleVoter(Policy(np.array([uniform(10,90), uniform(30,70)]))))

    # defining electorate
    electorate = create_simple_electorate(voters, "Example Issue 1", "Example Issue 2")

    # plotting an election
    electorate.plot_election_2d(p1, p2, verbose=True)