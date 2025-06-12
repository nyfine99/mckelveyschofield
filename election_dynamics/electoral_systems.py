from election_dynamics.election_dynamics import ElectionDynamicsTwoParty
from voters.voter import Voter


def status_quo_preference(counts: list[int]):
    """
    Returns 0 if the original policy is preferred or tied, and 1 if the new policy is preferred.
    """
    if counts[0] >= counts[1]:
        return 0
    return 1


def tiebreak_status_quo_preference():
    """
    Returns 0 (the original policy) in the event of a tie.
    """
    return 0


def create_two_party_status_quo_preference(
    voters: list[Voter],
    issue_1: str = "Issue 1",
    issue_2: str = "Issue 2",
):
    """
    Creates an ElectionDynamicsTwoParty where the evaluation function gives ties on
    overall preferences to the original policy. 
    If an individual's preferences are tied, that individual does not vote.
    """
    return ElectionDynamicsTwoParty(voters, status_quo_preference, issue_1=issue_1, issue_2=issue_2)


def create_ranked_choice_voting():
    pass