"""
Electoral Systems Factory Module

This module provides factory functions for creating different types of electoral systems
and electorates. It serves as a convenient interface for users to instantiate various
electoral dynamics simulations without needing to understand the underlying class
implementations.

The module supports:
- Two-party electoral systems with different voter types
- Multi-party electoral systems with first-past-the-post voting
- Ranked-choice voting systems
- Status quo preference handling for tied elections

Each factory function creates an appropriately configured ElectionDynamics object
with the specified evaluation function and voter type.
"""

from election_dynamics.election_dynamics_multi_party_simple_voters import ElectionDynamicsMultiPartySimpleVoters
from election_dynamics.election_dynamics_two_party import ElectionDynamicsTwoParty
from election_dynamics.election_dynamics_two_party_simple_voters import ElectionDynamicsTwoPartySimpleVoters
from election_dynamics.election_dynamics_two_party_taxicab_voters import ElectionDynamicsTwoPartyTaxicabVoters
from utility_functions.evaluation_functions import first_past_the_post, ranked_choice_preference, status_quo_preference 
from voters.simple_voter import SimpleVoter
from voters.taxicab_voter import TaxicabVoter
from voters.voter import Voter


def create_two_party_status_quo_preference(
    voters: list[Voter],
    issue_1: str = "Issue 1",
    issue_2: str = "Issue 2",
):
    """
    Creates a two-party electoral system with status quo preference for tied elections.
    
    This function creates an ElectionDynamicsTwoParty object where the evaluation
    function gives preference to the original (status quo) policy when total voter
    preferences are tied.
    
    Args:
        voters (list[Voter]): list of voter objects representing the electorate
        issue_1 (str, optional): label for the first policy dimension/issue, defaults to "Issue 1"
        issue_2 (str, optional): label for the second policy dimension/issue, defaults to "Issue 2"
    
    Returns:
        ElectionDynamicsTwoParty: configured two-party electoral system with status quo preference handling
    """
    return ElectionDynamicsTwoParty(
        voters, status_quo_preference, issue_1=issue_1, issue_2=issue_2
    )


def create_simple_electorate(
    voters: list[SimpleVoter],
    issue_1: str = "Issue 1",
    issue_2: str = "Issue 2",
):
    """
    Creates a two-party electoral system optimized for SimpleVoter objects.
    
    This is the most commonly used factory function, creating an
    ElectionDynamicsTwoPartySimpleVoters object. SimpleVoters use Euclidean
    distance-based utility functions, making them computationally efficient
    and suitable for most electoral simulations.
    
    The system automatically handles overall ties by giving preference to the original
    policy, and individual voters abstain when their preferences are tied.
    
    Args:
        voters (list[SimpleVoter]): list of SimpleVoter objects
        issue_1 (str, optional): label for the first policy dimension/issue, defaults to "Issue 1"
        issue_2 (str, optional): label for the second policy dimension/issue, defaults to "Issue 2"
    
    Returns:
        ElectionDynamicsTwoPartySimpleVoters: configured two-party electoral system optimized for SimpleVoters
    """
    return ElectionDynamicsTwoPartySimpleVoters(
        voters, issue_1=issue_1, issue_2=issue_2
    )


def create_taxicab_electorate(
    voters: list[TaxicabVoter],
    issue_1: str = "Issue 1",
    issue_2: str = "Issue 2",
):
    """
    Creates a two-party electoral system for TaxicabVoter objects.
    
    TaxicabVoters use Manhattan distance (L1 norm) instead of Euclidean distance
    for policy evaluation. This creates different preference patterns and can
    be useful for modeling certain types of policy spaces where changes along
    different dimensions are not equally weighted.
    
    Like in create_simple_electorate, ties are resolved in favor of the status quo policy,
    and tied voters abstain.
    
    Args:
        voters (list[TaxicabVoter]): list of TaxicabVoter objects
        issue_1 (str, optional): label for the first policy dimension/issue, defaults to "Issue 1"
        issue_2 (str, optional): label for the second policy dimension/issue, defaults to "Issue 2"
    
    Returns:
        ElectionDynamicsTwoPartyTaxicabVoters: configured two-party electoral system for TaxicabVoters
    """
    return ElectionDynamicsTwoPartyTaxicabVoters(
        voters, issue_1=issue_1, issue_2=issue_2
    )


def create_fptp_multiparty_electorate(
    voters: list[SimpleVoter],
    issue_1: str = "Issue 1",
    issue_2: str = "Issue 2",
):
    """
    Creates a multi-party electoral system using first-past-the-post voting.
    
    This function creates an ElectionDynamicsMultiPartySimpleVoters object
    configured for first-past-the-post (FPTP) voting. In FPTP systems, the
    policy/party/candidate with the most votes wins, regardless of vote share percentage.
    This is the standard voting system used in many countries including the US
    and UK.
    
    The system supports multiple policies competing simultaneously,
    making it more suitable for modeling real-world multi-party democracies.
    
    Args:
        voters (list[SimpleVoter]): list of SimpleVoter objects
        issue_1 (str, optional): label for the first policy dimension/issue, defaults to "Issue 1"
        issue_2 (str, optional): label for the second policy dimension/issue, defaults to "Issue 2"
    
    Returns:
        ElectionDynamicsMultiPartySimpleVoters: configured multi-party electoral system with FPTP voting
    """
    return ElectionDynamicsMultiPartySimpleVoters(
        voters, evaluation_function=first_past_the_post, issue_1=issue_1, issue_2=issue_2,
    )


def create_rcv_electorate(
    voters: list[SimpleVoter],
    issue_1: str = "Issue 1",
    issue_2: str = "Issue 2",
):
    """
    Creates a multi-party electoral system using ranked-choice voting (RCV).
    
    This function creates an ElectionDynamicsMultiPartySimpleVoters object
    configured for ranked-choice voting. RCV allows voters to rank multiple
    candidates/policies/parties in order of preference. If no policy receives a majority
    of first-choice votes, the least popular policy is eliminated and its
    votes are redistributed based on second choices. This process continues
    until one policy receives a majority.
    
    RCV systems can produce different outcomes than FPTP and are used in
    various jurisdictions, including some US cities and countries like Australia.
    
    Args:
        voters (list[SimpleVoter]): list of SimpleVoter objects
        issue_1 (str, optional): label for the first policy dimension/issue, defaults to "Issue 1"
        issue_2 (str, optional): label for the second policy dimension/issue, defaults to "Issue 2"
    
    Returns:
        ElectionDynamicsMultiPartySimpleVoters: configured multi-party electoral system with ranked-choice voting
    """
    return ElectionDynamicsMultiPartySimpleVoters(
        voters, evaluation_function=ranked_choice_preference, issue_1=issue_1, issue_2=issue_2,
    )
