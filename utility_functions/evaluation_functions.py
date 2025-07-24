import numpy as np
from numba import njit, int32, boolean
from typing import Union

def status_quo_preference(counts: list[int]):
    """
    Returns 0 if the original policy is preferred or tied, and 1 if the new policy is preferred.
    """
    if counts[0] >= counts[1]:
        return 0
    return 1


def tiebreak_status_quo_preference():
    """
    Returns 0 (the original policy) in the event of a tie in the voter's individual preferences.
    """
    return 0


def first_past_the_post(
    preferences: np.ndarray,
) -> int:
    """
    First-past-the-post (FPTP) from sorted preferences.
    Tie-breaking is done on index (relative incumbency, theoretically).
    Policies are not passed in as Policy objects, but are instead represented as integers (i.e. 
    policy 0, 1, ...).

    Params:
        preferences (np.ndarray): the voters' relative policy preferences; 
            preferences[i] is a list representing where the first element is voter i's most prefered policy, and so on

    Returns:
        int: the index of winning policy
    """
    _, num_policies = preferences.shape

    # get policy index of top choice for each voter
    first_choices = preferences[:, 0]
    # get the current count for each policy
    counts = np.bincount(first_choices, minlength=num_policies)

    return counts.argmax()


def ranked_choice_preference(
    preferences: np.ndarray,
    stop_at_majority: bool = True,
    output_vote_counts: bool = False
) -> Union[int, np.ndarray]:
    """
    Ranked Choice Voting (RCV) from sorted preferences.
    Tie-breaking on original first-round support, then index (relative incumbency, theoretically).
    Policies are not passed in as Policy objects, but are instead represented as integers (i.e. 
    policy 0, 1, ...).
    The user may not want to stop vote counting at majority to see how votes would have been distributed 
    if the election had been run to completion.
    If stop_at_majority is True, the function will return results as soon as more than half of the votes go to one policy.
    If stop_at_majority is False, the function will continue until only two policies are left.

    Params:
        preferences (np.ndarray): the voters' relative policy preferences; 
            preferences[i] is a list representing where the first element is voter i's most prefered policy, and so on
        stop_at_majority (bool): If True, the function will return the index of the winning policy once more 
            than half of the votes have been counted.
        output_vote_counts (bool): If True, return a 2D np.ndarray with vote counts for each policy by round.

    Returns:
        int | np.ndarray: the index of winning policy (if output_vote_counts is False), or
            vote counts by policy by round (if output_vote_counts is True)
    """
    num_voters, num_policies = preferences.shape
    active = np.ones(num_policies, dtype=bool)  # maintaining which policies are active

    # original first-round votes to use as a tie-breaker
    original_first_choices = preferences[:, 0]
    original_first_round_counts = np.bincount(original_first_choices, minlength=num_policies)

    vote_counts_by_round = []

    while True:
        # mask eliminated policies from preferences
        mask = active[preferences]  
        # for each voter, index of their highest-ranked active policy
        top_choice_indices = mask.argmax(axis=1)
        # get actual policy index of top choice for each vote
        first_choices = preferences[np.arange(num_voters), top_choice_indices]
        # get the current count for each policy
        counts = np.bincount(first_choices, minlength=num_policies)
        # get the total number of active votes
        total_active_votes = counts[active].sum()

        # store vote counts for this round
        if output_vote_counts:
            vote_counts_by_round.append(counts.copy())

        # determining if any policy can/must be declared the winner
        if (
            stop_at_majority and max(counts) > total_active_votes / 2
        ) or (
            len(active[active == True]) == 2
        ):
            if output_vote_counts:
                return np.array(vote_counts_by_round)
            
            # we need to determine which policy is the winner
            initial_winner = counts.argmax()
            if counts[initial_winner] > total_active_votes / 2:
                return initial_winner

            # otherwise, we have a tie, and need to apply tiebreaks
            max_votes = counts[active].max()
            greatest = np.flatnonzero((counts == max_votes) & active)

            # applying tiebreak 1 (original first-round support)
            orig_support = original_first_round_counts[greatest]
            max_orig = orig_support.max()
            greatest = greatest[orig_support == max_orig]

            # applying tiebreak 2 (smallest index wins) if needed
            return greatest.min()


        # finding which policy(s) received the least number of votes
        min_votes = counts[active].min()
        lowest = np.flatnonzero((counts == min_votes) & active)

        if len(lowest) > 1:
            # applying tiebreak 1 (original first-round support)
            orig_support = original_first_round_counts[lowest]
            min_orig = orig_support.min()
            lowest = lowest[orig_support == min_orig]

        # applying tiebreak 2 (smallest index wins) if needed
        to_eliminate = lowest.max()

        # eliminate worst-performing policy
        active[to_eliminate] = False

@njit
def fast_rcv_many_voters(preferences: np.ndarray):
    """
    Ranked Choice Voting (RCV) from sorted preferences.
    Tie-breaking on original first-round support, then index.
    Significantly faster than the other RCV function in cases of ~ millions of voters.
    """
    num_voters, num_policies = preferences.shape

    eliminated = np.zeros(num_policies, dtype=boolean)
    original_first_choice_counts = np.zeros(num_policies, dtype=int32)

    # Round 1 counts for tiebreaking
    for i in range(num_voters):
        first = preferences[i, 0]
        original_first_choice_counts[first] += 1

    # Each voter's index into their ranked preferences
    pointer = np.zeros(num_voters, dtype=int32)

    while True:
        # Count votes
        counts = np.zeros(num_policies, dtype=int32)
        for v in range(num_voters):
            p = pointer[v]
            while eliminated[preferences[v, p]]:
                p += 1
            pointer[v] = p
            counts[preferences[v, p]] += 1

        # Total votes
        total_votes = 0
        for i in range(num_policies):
            total_votes += counts[i]

        # Check for majority
        for i in range(num_policies):
            if counts[i] * 2 > total_votes:
                return i

        # Find lowest policy with tiebreaking
        min_votes = 2147483647  # max int32
        to_eliminate = -1

        for i in range(num_policies):
            if eliminated[i]:
                continue
            if counts[i] < min_votes:
                min_votes = counts[i]
                to_eliminate = i
            elif counts[i] == min_votes:
                oc_i = original_first_choice_counts[i]
                oc_te = original_first_choice_counts[to_eliminate]
                if oc_i < oc_te:
                    to_eliminate = i
                elif oc_i == oc_te:
                    to_eliminate = i if i < to_eliminate else to_eliminate

        eliminated[to_eliminate] = True
