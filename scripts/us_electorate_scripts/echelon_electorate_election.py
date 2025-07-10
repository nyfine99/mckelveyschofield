from policies.policy import Policy
from election_dynamics.electorates import create_us_electorate_echelon_sample


if __name__ == "__main__":
    # creating electorate
    electorate = create_us_electorate_echelon_sample()

    # defining policies
    libertarianism = Policy([70, 30], "Libertarianism")
    populism = Policy(
        [30, 70], "Populism"
    )  # at least, by Echelon's definition of populism

    # conducting election
    electorate.plot_election_2d(libertarianism, populism, verbose=True)
