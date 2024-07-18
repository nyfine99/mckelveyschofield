import math
import numpy as np

from policy import Policy

def neg_distance(p1: Policy, p2: Policy):
    """
    A utility function which is maximized when Euclidian distance between the two policies is smallest.
    """
    return -1 * math.dist(p1.values, p2.values)


def neg_taxicab_distance(p1: Policy, p2: Policy):
    """
    A utility function which is maximized when taxicab distance between the two policies is smallest.
    """
    v1 = np.array(p1.values)
    v2 = np.array(p2.values)

    return -1 * sum(abs(v2 - v1))
