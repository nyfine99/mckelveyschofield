import numpy as np
from datetime import datetime

from election_dynamics.electorates import create_us_electorate_echelon_sample_rcv
from utility_functions.evaluation_functions import ranked_choice_preference
import matplotlib.colors as mcolors
from policies.policy import Policy

us_rcv = create_us_electorate_echelon_sample_rcv()

# for i in range(8):
#     us_rcv.voters = us_rcv.voters + us_rcv.voters
#     us_rcv.voter_arr = np.array([voter.ideal_policy.values for voter in us_rcv.voters])

populist = Policy([30,70], "Populism")
mod_left = Policy([30,30], "Mod Left")
center = Policy([50,50], "Centrism")
mod_right = Policy([70,70], "Mod Right")
libertarian = Policy([70,30], "Libertarianism")
extreme_left = Policy([10,10], "Extreme Left")
extreme_populism = Policy([10,90], "Extreme Populism")
lean_right = Policy([60,60], "Lean Right")
lean_left = Policy([40,40], "Lean Left")
policies_list = [mod_left, mod_right, center, populist, libertarian, extreme_left]
policies_arr = np.array([p.values for p in policies_list])
# print(us_rcv.compare_policies(policies_list))
# us_rcv.plot_voters_first_choices(policies_list)

s_time = datetime.now()
rankings = us_rcv.tabulate_votes(policies_list)
e_time = datetime.now()
print(e_time - s_time)

s_time = datetime.now()
rankings = us_rcv.compare_policies(policies_list)
e_time = datetime.now()
print(e_time - s_time)

rankings = us_rcv.tabulate_votes(policies_list)

s_time = datetime.now()
results = us_rcv.evaluation_function(rankings)  # much faster performance
e_time = datetime.now()
print(e_time - s_time)

print(us_rcv.voters[0].ideal_policy.values)
print(us_rcv.voters[-1].ideal_policy.values)
print(results)


# for i in range(13):
#     us_rcv.voters = us_rcv.voters + us_rcv.voters
#     us_rcv.voter_arr = np.array([voter.ideal_policy.values for voter in us_rcv.voters])

# rankings = us_rcv.compute_preferences(policies_arr)
# s_time = datetime.now()
# print(fast_rcv(rankings))
# e_time = datetime.now()
# print(e_time - s_time)

# us_rcv.animate_election(policies_list)

tiny_policies_list = [mod_left, mod_right]
small_policies_list = [extreme_left, extreme_populism, center]
huge_policies_list = [mod_left, mod_right, center, populist, libertarian, extreme_left, extreme_populism, lean_right, lean_left]
big_us_rcv = create_us_electorate_echelon_sample_rcv()
for i in range(4):
    big_us_rcv.voters = big_us_rcv.voters + big_us_rcv.voters
    big_us_rcv.voter_arr = np.array([voter.ideal_policy.values for voter in big_us_rcv.voters])\

# us_rcv.animate_election(huge_policies_list)
# us_rcv.animate_election(huge_policies_list, stop_at_majority=False, output_folder="output", filename="rcv_election_animation_no_stop")

# us_rcv.create_election_sankey_diagram(small_policies_list)
# us_rcv.create_election_sankey_diagram(small_policies_list, stop_at_majority=False)

# big_us_rcv.create_election_sankey_diagram(tiny_policies_list)
# big_us_rcv.create_election_sankey_diagram(small_policies_list)
# big_us_rcv.create_election_sankey_diagram(policies_list)
# big_us_rcv.create_election_sankey_diagram(huge_policies_list)

us_rcv.gridsearch_policy_winmap([extreme_left, extreme_populism])