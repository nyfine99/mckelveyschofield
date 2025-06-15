from random import gauss

from policies.policy import Policy
from voters.simple_voter import SimpleVoter
from utility_functions.utility_functions import neg_distance
from election_dynamics.electoral_systems import create_simple_electorate

if __name__ == "__main__":
    # defining policies
    p1 = Policy([40,45], "Centrism") # more moderate
    p2 = Policy([80,90], "Extremism") # more extreme

    # defining voters
    voters = []
    for i in range(100):
        voters.append(SimpleVoter(Policy([gauss(50,15),gauss(50,10)])))

    # defining electorate
    electorate = create_simple_electorate(voters, "Example Issue 1", "Example Issue 2")

    # plotting an election
    # electorate.plot_election_2d(p1, p2, verbose=True)

    # plotting a path from the moderate position to the extreme one
    electorate.animate_mckelvey_schofield(p1, p2, 40, filename="example_output", verbose=True)
