from election_dynamics.election_dynamics_multi_party_simple_voters import ElectionDynamicsMultiPartySimpleVoters
from election_dynamics.election_dynamics_two_party import ElectionDynamicsTwoParty
from election_dynamics.election_dynamics_two_party_simple_voters import ElectionDynamicsTwoPartySimpleVoters
from election_dynamics.election_dynamics_two_party_taxicab_voters import ElectionDynamicsTwoPartyTaxicabVoters
from utility_functions.evaluation_functions import status_quo_preference
from voters.simple_voter import SimpleVoter
from voters.taxicab_voter import TaxicabVoter
from voters.voter import Voter


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
    return ElectionDynamicsTwoParty(
        voters, status_quo_preference, issue_1=issue_1, issue_2=issue_2
    )


def create_simple_electorate(
    voters: list[SimpleVoter],
    issue_1: str = "Issue 1",
    issue_2: str = "Issue 2",
):
    """
    Creates an ElectionDynamicsTwoPartySimpleVoters where the evaluation function gives ties on
    overall preferences to the original policy.
    If an individual's preferences are tied, that individual does not vote.
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
    Creates an ElectionDynamicsTwoPartyTaxicabVoters where the evaluation function gives ties on
    overall preferences to the original policy.
    If an individual's preferences are tied, that individual does not vote.
    """
    return ElectionDynamicsTwoPartyTaxicabVoters(
        voters, issue_1=issue_1, issue_2=issue_2
    )


def create_rcv_electorate(
    voters: list[SimpleVoter],
    issue_1: str = "Issue 1",
    issue_2: str = "Issue 2",
):
    return ElectionDynamicsMultiPartySimpleVoters(
        voters, issue_1=issue_1, issue_2=issue_2,
    )
