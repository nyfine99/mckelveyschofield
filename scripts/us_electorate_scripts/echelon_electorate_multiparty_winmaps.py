from election_dynamics.electorates import create_us_electorate_echelon_multiparty_fptp, create_us_electorate_multiparty_rcv
from policies.policy import Policy

if __name__ == "__main__":
    # defining policies across the ideological spectrum
    mod_left = Policy([30,30], "Moderate Liberalism")
    center = Policy([50,50], "Centrism")
    mod_right = Policy([70,70], "Moderate Conservatism")
    populist = Policy([30,70], "Populism")
    libertarian = Policy([70,30], "Libertarianism")
    extreme_left = Policy([10,10], "Extreme Leftism")
    extreme_right = Policy([90,90], "Extreme Rightism")

    all_policies_list = [
        mod_left,
        mod_right,
        center,
        populist,
        libertarian,
        extreme_left,
        extreme_right,
    ]

    # defining electorates
    rcv_electorate = create_us_electorate_multiparty_rcv()
    fptp_electorate = create_us_electorate_echelon_multiparty_fptp()

    # obtaining winmaps
    rcv_electorate.gridsearch_policy_winmap(all_policies_list, output_filename="output/echelon_electorate_rcv_winmap.png")
    fptp_electorate.gridsearch_policy_winmap(all_policies_list, output_filename="output/echelon_electorate_fptp_winmap.png")
