from utility_functions.utility_functions import neg_distance
from policies.policy import Policy
from voters.voter import Voter

"""
A file which defines the SimpleVoter class. This voter evaluates policies based off of Euclidean distance, 
preferring a closer policy by this metric to a further one.
"""

class SimpleVoter(Voter):
    def __init__(self, ideal_policy: Policy):
        """
        Initializes a SimpleVoter.

        Params:
            ideal_policy (Policy): an ordered list of the voter's policy preferences.
        """
        self.ideal_policy = ideal_policy
        self.utility_function = neg_distance
