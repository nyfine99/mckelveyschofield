from random import gauss

import math

from policies.policy import Policy
from voters.voter import Voter
from utility_functions.utility_functions import neg_distance_with_limit
from election_dynamics.electoral_systems import create_two_party_status_quo_preference

# defining policies
p1 = Policy([45,45]) # more moderate
p2 = Policy([70,50]) # more extreme

# defining voters
voters = []
for i in range(1000):
    voters.append(Voter(Policy([gauss(50,15),gauss(50,10)]), utility_function=neg_distance_with_limit(25)))

# defining electorate
electorate = create_two_party_status_quo_preference(voters)

# plotting results
electorate.plot_election_2d(p1, p2)
