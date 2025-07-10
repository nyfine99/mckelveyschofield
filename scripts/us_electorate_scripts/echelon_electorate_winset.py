from policies.policy import Policy
from election_dynamics.electorates import create_us_electorate_echelon_sample


if __name__ == "__main__":
    # creating electorate
    electorate = create_us_electorate_echelon_sample()

    # defining policies
    centrism = Policy([50, 50], "Centrism")

    # plotting a winset
    electorate.plot_winset_boundary(
        centrism, n_directions=1080, n_halving_iterations=15
    )
