from election_dynamics.election_dynamics import ElectionDynamicsTwoParty
from voters.voter import Voter


def status_quo_preference(voters: list[Voter], original_policy, new_policy):
    """
    Returns 0 if the original policy is preferred, and 1 if the new policy is preferred.
    """
    counts = [0,0] # index 0 represents the original_policy count, index 1 the new policy
    for voter in voters:
        original_utility = voter.get_utility(original_policy)
        new_utility = voter.get_utility(new_policy)
        if original_utility >= new_utility:
            counts[0] += 1
        else:
            counts[1] += 1
    if counts[0] >= counts[1]:
        return 0
    return 1


def create_two_party_status_quo_preference(voters: list[Voter]):
    """
    Creates an ElectionDynamicsTwoParty where the evaluation function gives ties on both individual and
    overall preferences to the original policy.
    """
    return ElectionDynamicsTwoParty(voters, status_quo_preference)


def create_ranked_choice_voting():
    pass