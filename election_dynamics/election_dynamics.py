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

    def compare_policies(self, original_policy: Policy, new_policy: Policy):
        return self.evaluation_function(self.tabulate_votes(original_policy, new_policy))
    
    def tabulate_votes(self, original_policy: Policy, new_policy: Policy):
        counts = [0,0] # index 0 represents the original_policy count, index 1 the new policy
        votes = self.obtain_individual_votes(original_policy, new_policy)
        counts[0] = sum([1 for vote in votes if vote == 0])
        counts[1] = sum([1 for vote in votes if vote == 1])
        return counts
    
    def obtain_individual_votes(self, original_policy: Policy, new_policy: Policy):
        votes = len(self.voters)*[-1] # index 0 represents the original_policy count, index 1 the new policy
        for i in range(0,len(self.voters)):
            voter = self.voters[i]
            original_utility = voter.get_utility(original_policy)
            new_utility = voter.get_utility(new_policy)
            if original_utility > new_utility:
                votes[i] = 0
            elif original_utility == new_utility and self.tiebreak_func is not None:
                votes[i] = self.tiebreak_func()
            elif original_utility < new_utility:
                votes[i] = 1
            # otherwise, the voter doesn't vote
        return votes
    
    def plot_election_2d(self, original_policy: Policy, new_policy: Policy, verbose: bool = True):
        votes = self.obtain_individual_votes(original_policy, new_policy)
        plt.figure(figsize=(10, 8))
        
        # plotting all voters
        colors = []
        for i in range(len(self.voters)):
            voter = self.voters[i]
            if votes[i] == 0:
                colors.append('blue')
            elif votes[i] == 1:
                colors.append('red')
            else:
                colors.append('yellow')
        
        plt.scatter(
            [voter.ideal_policy.values[0] for voter in self.voters], 
            [voter.ideal_policy.values[1] for voter in self.voters], 
            color=colors, 
            marker='o'
        )
        
        # plotting policies and differentiating winner from loser
        winner = self.compare_policies(original_policy,new_policy)
        blue_marker = '*' if winner == 0 else 'X'
        blue_size = 250 if winner == 0 else 150
        red_marker = '*' if winner == 1 else 'X'
        red_size = 250 if winner == 1 else 150
        winner_text = "original policy" if winner == 0 else "new policy"
        vote_totals = self.tabulate_votes(original_policy, new_policy)
        original_policy_votes = vote_totals[0]
        new_policy_votes = vote_totals[1]
        plt.scatter(
            [original_policy.values[0]],
            [original_policy.values[1]], 
            color='blue', 
            marker=blue_marker, 
            edgecolors='black',
            s=blue_size
        )
        plt.scatter(
            [new_policy.values[0]],
            [new_policy.values[1]], 
            color='red', 
            marker=red_marker, 
            edgecolors='black', 
            s=red_size
        )

        # title and display
        plt.title('Voters\' Distribution')
        plt.xlabel('Issue 1 Position')
        plt.ylabel('Issue 2 Position')
        plt.grid(True)
        plt.show()


class ElectionDyanamicsMultiParty(ElectionDynamics):
    def __init__(self, voters: list[Voter], evaluation_function: callable):
        self.voters = voters
        self.evaluation_function = evaluation_function

    def compare_policies(self, policies: list[Policy]):
        return self.evaluation_function(self.voters, policies)
    