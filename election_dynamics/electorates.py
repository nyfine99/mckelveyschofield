import pandas as pd
import numpy as np

from election_dynamics.election_dynamics_two_party import ElectionDynamicsTwoParty
from election_dynamics.electoral_systems import create_simple_electorate
from utility_functions.evaluation_functions import status_quo_preference
from voters.simple_voter import SimpleVoter
from voters.voter import Policy


def create_us_electorate_echelon_sample():
    """
    Creates a US electorate corresponding to the sample data from Echelon Insights, populated with Euclidean voters.
    This electorate does not account for the voter weights present in Echelon's data, nor for other attributes in the
    data, outside of the Econ Scores and Social Scores of the voters.

    Returns the created electorate.
    """
    # loading voter data
    voter_data = pd.read_csv("data/us_electorate_echelon_sample_2025.csv")

    # transforming and cleaning data
    voter_data['Econ Score'] = pd.to_numeric(voter_data['Econ Score'], errors='coerce')
    voter_data['Social Score'] = pd.to_numeric(voter_data['Social Score'], errors='coerce')
    voter_data.dropna(subset=['Econ Score', 'Social Score'], inplace=True)

    # defining voters
    voters = [SimpleVoter(Policy(
        np.array([row['Econ Score'], row['Social Score']])
    )) for _, row in voter_data.iterrows()]

    # defining electorate
    electorate = create_simple_electorate(voters, "Economic Conservatism", "Social Conservatism")
    return electorate
