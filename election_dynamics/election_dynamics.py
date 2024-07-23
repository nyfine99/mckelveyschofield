import matplotlib.pyplot as plt

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


class ElectionDynamicsTwoParty(ElectionDynamics):
    def __init__(self, voters: list[Voter], evaluation_function: callable, tiebreak_func: callable = None):
        self.voters = voters
        self.evaluation_function = evaluation_function
        self.tiebreak_func = tiebreak_func # used when a voter is ambivalent to distribute their vote

    def tabulate_votes(self, original_policy: Policy, new_policy: Policy):
        counts = [0,0] # index 0 represents the original_policy count, index 1 the new policy
        for voter in self.voters:
            original_utility = voter.get_utility(original_policy)
            new_utility = voter.get_utility(new_policy)
            if original_utility > new_utility:
                counts[0] += 1
            elif original_utility == new_utility and self.tiebreak_func is not None:
                # if == and no tiebreak_func, the voter doesn't vote
                self.tiebreak_func()
            elif original_utility < new_utility:
                counts[1] += 1
        return counts

    def compare_policies(self, original_policy: Policy, new_policy: Policy):
        return self.evaluation_function(self.tabulate_votes(original_policy, new_policy))
    
    def obtain_votes(self, original_policy: Policy, new_policy: Policy):
        votes = len(self.voters)*[-1] # index 0 represents the original_policy count, index 1 the new policy
        for i in range(0,len(self.voters)):
            voter = self.voters[i]
            original_utility = voter.get_utility(original_policy)
            new_utility = voter.get_utility(new_policy)
            if original_utility > new_utility:
                votes[i] = 0
            elif original_utility == new_utility and self.tiebreak_func is not None:
                # if == and no tiebreak_func, the voter doesn't vote
                self.tiebreak_func()
            elif original_utility < new_utility:
                votes[i] = 1
        return votes
    
    def plot_election_2d(self, original_policy: Policy, new_policy: Policy):
        votes = self.obtain_votes(original_policy, new_policy)
        plt.figure(figsize=(8, 6))
        # should probably make the winner have a star at some point or something
        plt.scatter([original_policy.values[0]],[original_policy.values[1]], color='blue', marker='x')
        plt.scatter([new_policy.values[0]],[new_policy.values[1]], color='red', marker='x')
        for i in range(len(self.voters)):
            voter = self.voters[i]
            color = 'yellow'
            if votes[i] == 0:
                color = 'blue'
            elif votes[i] == 1:
                color = 'red'
            plt.scatter([voter.ideal_policy.values[0]], [voter.ideal_policy.values[1]], color=color, marker='o')
        plt.title('Voters\' Distribution')
        # plt.xlabel(issues[0])
        # plt.ylabel(issues[1])
        plt.grid(True)
        plt.show()


class ElectionDyanamicsMultiParty(ElectionDynamics):
    def __init__(self, voters: list[Voter], evaluation_function: callable):
        self.voters = voters
        self.evaluation_function = evaluation_function

    def compare_policies(self, policies: list[Policy]):
        return self.evaluation_function(self.voters, policies)