"""
Multi-party Electoral Dynamics with Simple Voters Implementation

This module defines the ElectionDynamicsMultiPartySimpleVoters, building slightly off of
ElectionDynamicsMultiParty. Specifically, the tabulate_votes function is overridden with a
more efficient version.
"""


import numpy as np
from typing import Callable

from election_dynamics.election_dynamics_multi_party import ElectionDynamicsMultiParty
from policies.policy import Policy
from voters.simple_voter import SimpleVoter


class ElectionDynamicsMultiPartySimpleVoters(ElectionDynamicsMultiParty):
    def __init__(
        self,
        voters: list[SimpleVoter],
        evaluation_function: Callable,
        issue_1: str = "Issue 1",
        issue_2: str = "Issue 2",
    ):
        self.voters = voters
        self.voter_arr = np.array([voter.ideal_policy.values for voter in self.voters])
        self.evaluation_function = evaluation_function
        self.tiebreak_func = None
        self.issue_1 = issue_1  # Issue 1 name
        self.issue_2 = issue_2  # Issue 2 name


    def tabulate_votes(self, policies: list[Policy]) -> np.ndarray:
        """
        Computes ranked preferences for each voter by distance to policy (closer = higher utility).
        Ties are broken by the index of the policy (lower index = higher utility).
        Returns a 2D np.ndarray where each row is a voter's ranked policy indices (best to worst).

        Args:
            policies (list[Policy]): List of Policy objects.
        
        Returns:
            np.ndarray: shape (num_voters, num_policies)
        """
        policies_arr = np.array([p.values for p in policies])
        dists = np.linalg.norm(self.voter_arr[:, np.newaxis, :] - policies_arr[np.newaxis, :, :], axis=2)
        preferences = np.argsort(dists, axis=1)
        return preferences
