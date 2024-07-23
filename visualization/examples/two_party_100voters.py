from random import randint

from policies.policy import Policy
from voters.voter import Voter
from utility_functions.utility_functions import neg_distance
from election_dynamics.electoral_systems import create_two_party_status_quo_preference


p1 = Policy([50,50]) # moderate
p2 = Policy([75,75]) # more extreme

voters = []
for i in range(100):
    voters.append(Voter(Policy([randint(0,100),randint(0,100)]), utility_function=neg_distance))

electorate = create_two_party_status_quo_preference(voters)
electorate.plot_election_2d(p2, p1)