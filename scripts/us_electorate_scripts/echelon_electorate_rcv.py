from numpy import squeeze
from election_dynamics.electorates import create_us_electorate_multiparty_rcv
from policies.policy import Policy
from voters.simple_voter import SimpleVoter

if __name__ == "__main__":
    # defining policies across the ideological spectrum
    mod_left = Policy([30,30], "Moderate Liberalism")
    center = Policy([50,50], "Centrism")
    mod_right = Policy([70,70], "Moderate Conservatism")
    populist = Policy([30,70], "Populism")
    libertarian = Policy([70,30], "Libertarianism")
    extreme_left = Policy([10,10], "Extreme Leftism")
    extreme_right = Policy([90,90], "Extreme Rightism")
    lean_right = Policy([60,60], "Slight Conservatism")
    lean_left = Policy([40,40], "Slight Liberalism")

    # for basic rcv elections
    basic_policies_list = [
        mod_left,
        mod_right,
        center,
    ]

    # to see centrism get squeezed out
    centrist_policies_list = [
        lean_left,
        lean_right,
        center,
    ]

    # huge competition
    all_policies_list = [
        mod_left,
        mod_right,
        center,
        populist,
        libertarian,
        extreme_left,
        extreme_right,
        lean_left,
        lean_right,
    ]

    # defining electorate
    rcv_electorate = create_us_electorate_multiparty_rcv()

    # animating the elections
    rcv_electorate.animate_election(
        basic_policies_list, 
        stop_at_majority=True, 
        output_folder="output",
        filename="echelon_electorate_small_rcv_animation",
    )
    rcv_electorate.animate_election(
        centrist_policies_list, 
        stop_at_majority=True, 
        output_folder="output",
        filename="echelon_electorate_squeeze_rcv_animation",
    )
    rcv_electorate.animate_election(
        all_policies_list, 
        stop_at_majority=True, 
        output_folder="output",
        filename="echelon_electorate_big_rcv_animation",
    )

    # creating sankey diagrams of the elections
    rcv_electorate.create_election_sankey_diagram(
        basic_policies_list, 
        stop_at_majority=True, 
        output_folder="output",
        filename="echelon_electorate_small_rcv_sankey",
    )
    rcv_electorate.create_election_sankey_diagram(
        centrist_policies_list, 
        stop_at_majority=True, 
        output_folder="output",
        filename="echelon_electorate_squeeze_rcv_sankey",
    )
    rcv_electorate.create_election_sankey_diagram(
        all_policies_list, 
        stop_at_majority=True, 
        output_folder="output",
        filename="echelon_electorate_big_rcv_sankey",
    )
