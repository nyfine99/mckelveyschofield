from utility_functions.utility_functions import neg_taxicab_distance
from policies.policy import Policy
from voters.voter import Voter

"""
A file which defines the TaxicabVoter class. This voter evaluates policies based off of the L1 norm (adding up
differences in each direction), preferring a closer policy by this metric to a further one.
"""

class TaxicabVoter(Voter):
    def __init__(self, ideal_policy: Policy):
        """
        Initializes a TaxicabVoter.

        Params:
            ideal_policy (Policy): an ordered list of the voter's policy preferences.
        """
        self.ideal_policy = ideal_policy
        self.utility_function = neg_taxicab_distance
