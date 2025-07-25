from utility_functions.utility_functions import neg_distance
from policies.policy import Policy

"""
A file which defines the Voter class.
"""

class Voter():
    def __init__(self, ideal_policy: Policy, utility_function: callable = neg_distance):
        """
        Initializes a Voter.

        Params:
            ideal_policy (Policy): an ordered list of the voter's policy preferences.
            utility_function (callabe): the voter's utility function; negative distance by default.
        """
        self.ideal_policy = ideal_policy
        self.utility_function = utility_function


    def get_utility(self, policy: Policy) -> float:
        """
        Returns the utility of the policy proposed.

        Params:
            policy (Policy): the policy.

        Returns:
            float: the utility of the policy.
        """
        return self.utility_function(self.ideal_policy, policy)
    

    # I probably need a version of the function below which works for 
    # more than two inputs to output the closest policy (and ignores ties), 
    # and then another version which deals with ties, either randomly or systematically
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

    def rank_policies_by_utility_index_preference(self, policies: list[Policy]) -> list[int]:
        """
        Ranks policies by utility (closer to ideal_policy is better),
        with ties broken by original index order (earlier is better).
        TODO: figure out a more malleable tiebreak system.

        Params:
            policies (list[Policy]): A list of policies being considered by the voter

        Returns:
            list[int]: List of policy indices in ranked order (best to worst)
        """
        scored = [
            (self.get_utility(policy), idx)
            for idx, policy in enumerate(policies)
        ]
        scored.sort(reverse=True)  # sorts by distance, then by index
        return [idx for _, idx in scored]
