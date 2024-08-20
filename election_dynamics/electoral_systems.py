from election_dynamics.election_dynamics import ElectionDynamicsTwoParty
from voters.voter import Voter


def status_quo_preference(counts: list[int]):
    """
    Returns 0 if the original policy is preferred, and 1 if the new policy is preferred.
    """
    if counts[0] >= counts[1]:
        return 0
    return 1


def create_two_party_status_quo_preference(voters: list[Voter]):
    """
    Creates an ElectionDynamicsTwoParty where the evaluation function gives ties on
    overall preferences to the original policy. 
    If an individual's preferences are tied, that individual does not vote.
    """
    return ElectionDynamicsTwoParty(voters, status_quo_preference)


def create_ranked_choice_voting():
    pass