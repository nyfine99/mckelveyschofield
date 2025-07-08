from abc import ABC, abstractmethod

from voters.voter import Voter
from policies.policy import Policy

"""
A file which defines the ElectionDynamics abstract class and extending classes. 
These classes maintain the attributes of Voters across an electorate,
as well as functions useful in determining which policies would succeed among them.
"""


class ElectionDynamics(ABC):
    """
    The ElectionDynamics abstract class. Voters is a list of Voters, while the evaluation_function
    field holds a function which takes in all of the policies and voters, and outputs the successful policy
    or proportional distribution of seats among the policy/parties.
    """

    def __init__(self, voters: list[Voter], evaluation_function: callable):
        self.voters = voters
        self.evaluation_function = evaluation_function

    @abstractmethod
    def tabulate_votes(self):
        pass

    @abstractmethod
    def compare_policies(self):
        pass


class ElectionDyanamicsMultiParty(ElectionDynamics):
    """
    TODO: will allow for implementation of multi-party elections and RCV.
    """

    def __init__(self, voters: list[Voter], evaluation_function: callable):
        self.voters = voters
        self.evaluation_function = evaluation_function

    def tabulate_votes(self):
        pass

    def compare_policies(self, policies: list[Policy]):
        return self.evaluation_function(self.voters, policies)
