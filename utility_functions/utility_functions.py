import math
import numpy as np

from policies.policy import Policy

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


def neg_taxicab_distance(ideal_policy: Policy, proposed_policy: Policy) -> float:
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


def neg_distance_with_limit(dist_max: float) -> callable:
    """
    A utility function which is maximized when Euclidian distance between the two policies is smallest.
    However, if p1 is far enough away from p2 (farther than dist_max), the minimum utility value is returned.
    By implication, the voter will be indifferent to policies beyond this distance.
    
    Params:
        dist_max (float): the distance from a policy at which a voter considers all distances the same.

    Returns:
        callable: a utility function which allows a voter to evaluate a policy.
    """

    def inner(a, b):
        return max(neg_distance(a, b), -1*dist_max)
    return inner
