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


    def get_utility(self, policy: Policy) -> float:
        """
        Returns the utility of the policy proposed.

        Params:
            policy (Policy): the policy.

        Returns:
            float: the utility of the policy.
        """
        return neg_distance(self.ideal_policy, policy)
    

    def preferred_policy(self, p1: Policy, p2: Policy) -> int:
        """
        Returns 1 if the preferred policy is the first, 2 if the preferred policy is the second, 
        and 0 if the voter is indifferent.

        Params:
            p1 (Policy): the first policy.
            p2 (Policy): the second policy.

        Returns:
            float: the policy chosen by the voter, or 0 if indifferent.
        """
        val = self.get_utility(p1) - self.get_utility(p2)
        if val > 0:
            return 1
        elif val < 0:
            return 2
        return 0
