import numpy as np

def min_mov(vote_matrix):
    # Example: minimum margin of victory in each round
    margins = []
    for row, round_votes in enumerate(vote_matrix):
        new_policy_vote_count = round_votes[-1]
        sorted_votes = np.sort(round_votes)[row:]  # removing eliminated policies
        new_policy_performance = new_policy_vote_count - sorted_votes[0]  # votes above eliminated policy
        margins.append(new_policy_performance)
        if new_policy_performance <= 0:
            break
    return min(margins) if margins else 0

def mov_final_round(vote_matrix):
    # Example: margin of victory in final round
    final_round_votes = vote_matrix[-1,:]
    if final_round_votes[-1] == 0:
        # new candidate did not make it to the final round
        return 0
    
    # new candidate made it to the final round; make sure they won
    argmax_index = np.argmax(final_round_votes)
    if argmax_index != len(final_round_votes) - 1:
        return 0 

    # new candidate won the final round, return MOV
    sorted_votes = np.sort(final_round_votes)
    return sorted_votes[-1] - sorted_votes[-2]
