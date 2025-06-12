from random import gauss

import math

from policies.policy import Policy
from voters.voter import Voter
from utility_functions.utility_functions import neg_distance_with_limit
from election_dynamics.electoral_systems import create_two_party_status_quo_preference

if __name__ == "__main__":
    # defining policies
    p1 = Policy([55,45], "moderate") # more moderate
    p2 = Policy([30,60], "extreme") # more extreme

    # defining voters
    voters = []
    for i in range(1000):
        voters.append(Voter(Policy([gauss(50,15),gauss(50,10)]), utility_function=neg_distance_with_limit(25)))

    # defining electorate
    electorate = create_two_party_status_quo_preference(voters, "Example Issue 1", "Example Issue 2")

    # plotting results
    electorate.plot_election_2d(p1, p2, verbose=True)
