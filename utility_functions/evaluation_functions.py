import numpy as np
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


def ranked_choice_preference(ballots: list[list[int]]) -> int:
    """
    Performs Ranked Choice Voting with specified tie-breaking:
    - Majority wins
    - Lowest first-choice votes eliminated
    - Ties broken by original first-round support, then by index

    Params:
        ballots: List of lists. Each sublist is a voter's ranked list of policy indices.

    Returns:
        int: Index of winning policy
    """

    ballots = np.array(ballots)

    # Original first-round votes to use as a tie-breaker
    original_first_choices = ballots[:, 0]
    original_first_round_counts = Counter(original_first_choices)

    eliminated = set()

    while True:
        # Mask eliminated candidates from each ballot
        masked_ballots = np.array([
            [c for c in ballot if c not in eliminated]
            for ballot in ballots
        ])
        
        # Get current first-choice votes
        first_choices = [ballot[0] for ballot in masked_ballots if len(ballot) > 0]
        tally = Counter(first_choices)
        total_active = sum(tally.values())

        # Check for majority
        # print(tally)  # for debug
        for candidate, count in tally.items():
            if count > total_active / 2:
                return candidate

        # Find candidate(s) with fewest first-choice votes
        min_votes = min(tally.values())
        lowest = [c for c in tally if tally[c] == min_votes]

        # Tiebreak 1: original first-round support
        if len(lowest) > 1:
            original_counts = {c: original_first_round_counts[c] for c in lowest}
            min_original = min(original_counts.values())
            lowest = [c for c in lowest if original_counts[c] == min_original)]

        # Tiebreak 2: smallest index
        to_eliminate = min(lowest)

        eliminated.add(to_eliminate)
