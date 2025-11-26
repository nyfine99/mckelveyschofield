"""
Election Dynamics Abstract Base Class Module

This module defines the core abstract interface for electoral dynamics simulations.
The ElectionDynamics abstract class serves as the foundation for all electoral
system implementations, providing consistent expectations for capabilities in vote tabulation,
policy comparison, and electoral analysis.

The abstract class enforces a common structure across different electoral systems
(two-party, multi-party, different voter types) while allowing specific
implementations to define their own logic for vote counting and policy evaluation.

This design pattern enables:
- Easy extension for new voting methods
- Polymorphic behavior for different voter types
- Standardized electoral analysis tools
"""

from abc import ABC, abstractmethod

from voters.voter import Voter
from policies.policy import Policy


class ElectionDynamics(ABC):
    """
    Abstract base class for electoral dynamics simulations.
    
    This class defines the interface that all electoral systems must implement.
    It maintains a collection of voters and an evaluation function that determines
    electoral outcomes based on voter preferences and policy positions.
    
    The abstract methods ensure that all electoral systems provide consistent
    functionality for vote tabulation and policy comparison, while allowing
    specific implementations to define their own electoral logic.
    
    Attributes:
        voters (list[Voter]): collection of voter objects representing the electorate
        evaluation_function (callable): function that determines electoral outcomes;
            takes preferences as input and returns a winning policy and/or full results by round
    """

    def __init__(self, voters: list[Voter], evaluation_function: callable):
        """
        Initialize the electoral dynamics system.
        
        Args:
            voters (list[Voter]): list of voter objects representing the electorate;
                each voter should have an ideal policy position and utility function for policy evaluation
            evaluation_function (callable): function that determines electoral outcomes;
                should accept preferences as input and return a winning policy and/or full results by round
        
        Raises:
            ValueError: If voters list is empty or evaluation_function is None.
        """
        if not voters:
            raise ValueError("Voters list cannot be empty")
        if evaluation_function is None:
            raise ValueError("Evaluation function cannot be None")
            
        self.voters = voters
        self.evaluation_function = evaluation_function

    @abstractmethod
    def tabulate_votes(self):
        """
        Abstract method for counting votes across all policies.
        
        This method must be implemented by all concrete electoral systems to
        define how votes are counted and aggregated. Different electoral systems
        may use different counting methods (e.g., first-past-the-post, ranked-choice,
        proportional representation). Arguments will differ by implementation, but consist of
        either a pair of policies or a list of policies. Output structure will also differ by implementation.
        """
        pass

    @abstractmethod
    def compare_policies(self):
        """
        Abstract method for comparing policies.
        
        This method must be implemented by all concrete electoral systems to
        define how policies are compared in head-to-head competition.
        Arguments will differ by implementation, but consist of either a pair of policies or a list of policies.
        Output structure will also differ by implementation.
        """
        pass
