"""
This module defines functions to evaluate a canditate's performance in multiway elections.
For example, a candidate might be rated based on their margin of victory in the final round.
"""

import numpy as np

def min_mov_rcv(vote_matrix):
    """
    Returns the minimum margin of victory of the new policy in each round; only compatible with RCV.
    Assumes that the new policy is the last policy in the vote matrix.

    Example:
        11 10 18
        14  0 25
        gives 8 as the output
    """
    margins = []
    for row, round_votes in enumerate(vote_matrix):
        new_policy_vote_count = round_votes[-1]
        sorted_votes = np.sort(round_votes)[row:]  # removing eliminated policies
        new_policy_performance = new_policy_vote_count - sorted_votes[0]  # votes above eliminated policy
        margins.append(new_policy_performance)
        if new_policy_performance <= 0:
            # the new policy has been eliminated, or tied in votes with an eliminated policy
            # either way, 0 will be the minimum, so can stop here
            break

    return min(margins) if margins else 0

def mov_final_round(vote_matrix):
    """
    Returns the margin of victory of the new policy in the final round; compatible with both FPTP and RCV
    Assumes that the new policy is the last policy in the vote matrix.

    Example:
        11 10 18
        14  0 25
        gives 11 as the output
    """
    final_round_votes = vote_matrix[-1,:]
    if final_round_votes[-1] == 0:
        # new candidate did not make it to the final round
        return 0
    
    # new candidate made it to the final round; make sure they won
    argmax_index = np.argmax(final_round_votes)
    if argmax_index != len(final_round_votes) - 1:
        return 0 

    # new candidate won the final round or tied, return MOV
    sorted_votes = np.sort(final_round_votes)
    return sorted_votes[-1] - sorted_votes[-2]
