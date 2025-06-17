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
