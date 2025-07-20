import numpy as np
from numba import njit, int32, boolean
from collections import Counter

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


def ranked_choice_preference(preferences: np.ndarray) -> int:
    """
    Optimized Ranked Choice Voting (RCV) from sorted preferences.
    Tie-breaking on original first-round support, then index (relative incumbency, theoretically).
    Policies are not passed in as Policy objects, but are instead represented as integers (i.e. 
    policy 0, 1, ...).

    Params:
        preferences (np.ndarray): the voters' relative policy preferences; 
            preferences[i] is a list representing where the first element is voter i's most prefered policy, and so on

    Returns:
        int: the index of winning policy
    """
    num_voters, num_policies = preferences.shape
    active = np.ones(num_policies, dtype=bool)  # maintaining which policies are active

    # original first-round votes to use as a tie-breaker
    original_first_choices = preferences[:, 0]
    original_first_round_counts = np.bincount(original_first_choices, minlength=num_policies)

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

        # if more than half of active votes went to one policy, that policy is the winner
        for i in np.flatnonzero(active):
            if counts[i] > total_active_votes / 2:
                return i

        # finding which policy(s) received the least number of votes
        min_votes = counts[active].min()
        lowest = np.flatnonzero((counts == min_votes) & active)

        if len(lowest) > 1:
            # applying tiebreak 1 (original first-round support)
            orig_support = original_first_round_counts[lowest]
            min_orig = orig_support.min()
            lowest = lowest[orig_support == min_orig]

        # due to how argmax works, tiebreak 2 (smallest index) will have already been applied if needed
        to_eliminate = lowest.min()

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

        # Find lowest candidate with tiebreaking
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
