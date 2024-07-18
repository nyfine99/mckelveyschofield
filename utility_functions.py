import math
import numpy as np

from policy import Policy

def neg_distance(ideal_policy: Policy, proposed_policy: Policy) -> float:
    """
    A utility function which is maximized when Euclidian distance between the two policies is smallest.

    Params:
            ideal_policy (Policy): the ideal policy.
            p2 (Policy): the proposed policy.

    Returns:
            float: the utility of the proposed policy to the voter given the ideal policy.
    """
    return -1.0 * math.dist(ideal_policy.values, proposed_policy.values)


def neg_taxicab_distance(ideal_policy: Policy, proposed_policy: Policy):
    """
    A utility function which is maximized when taxicab distance between the two policies is smallest.

    Params:
            ideal_policy (Policy): the ideal policy.
            p2 (Policy): the proposed policy.

    Returns:
            float: the utility of the proposed policy to the voter given the ideal policy.
    """
    v1 = np.array(ideal_policy.values)
    v2 = np.array(proposed_policy.values)

    return -1.0 * sum(abs(v2 - v1))
