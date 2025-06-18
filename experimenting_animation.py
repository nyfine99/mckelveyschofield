
from random import randint, gauss

import math
import numpy as np
import random
import matplotlib.pyplot as plt

from policies.policy import Policy
from voters.simple_voter import SimpleVoter
from utility_functions.utility_functions import neg_distance, capped_neg_distance
from election_dynamics.electoral_systems import create_simple_electorate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# hi
SEED = 11
NUM_VOTERS = 100
random.seed(SEED)  # 8 is lopsided, as is 9, though in a different direction

voters = []
for i in range(NUM_VOTERS):
    voters.append(
        SimpleVoter(
            Policy(
                [
                    random.sample([i for i in range(100)], 1)[0], 
                    random.sample([i for i in range(100)], 1)[0]
                ]
            )
        )
    )

electorate = create_simple_electorate(voters, "Derekkkkk", "BONIUK")
average_policy = Policy(
    [
        sum([voter.ideal_policy.values[0] for voter in voters])/len(voters), 
        sum([voter.ideal_policy.values[1] for voter in voters])/len(voters)
    ],
    "Moderation"
)
GOAL_POLICY = Policy([100,100], "Terrorism")
electorate.animate_mckelvey_schofield(average_policy, GOAL_POLICY, filename="output_boniuk", fps=1)

# # Create figure outside so FuncAnimation can reuse it
# fig = plt.figure()

# policies_at_step = {0: Policy(average_policy)}

# def make_frame(f_num):
#     plt.clf()  # Clear the current axes/figure
#     print(f"Starting to create frame {f_num}")
#     current_policy = policies_at_step[f_num]

#     # avg_dist = sum([math.dist(current_policy.values, voter.ideal_policy.values) for voter in voters])/len(voters)
#     # print(f"Average distance: {avg_dist}")
#     max_dist = max([math.dist(current_policy.values, voter.ideal_policy.values) for voter in voters])
#     max_r = 2.0 * max_dist
#     inner_bounds = []
#     outer_bounds = []
#     number_of_points = 360
#     angles = [math.pi * 2.0 * n/(float(number_of_points)) for n in range(number_of_points)]
#     for angle in angles:
#         inner_bounds.append(current_policy.values)
#         outer_bounds.append([current_policy.values[0] + max_r * math.sin(angle), current_policy.values[1] + max_r * math.cos(angle)])

#     num_halving_iterations = 12
#     for i in range(len(inner_bounds)):
#         for j in range(num_halving_iterations):
#             curr_mid_policy_values = [
#                 (inner_bounds[i][0] + outer_bounds[i][0])/2, (inner_bounds[i][1] + outer_bounds[i][1])/2
#             ]
#             curr_mid_policy = Policy(curr_mid_policy_values)
#             if electorate.compare_policies(current_policy, curr_mid_policy):
#                 # curr_mid policy won
#                 inner_bounds[i] = curr_mid_policy_values
#             else:
#                 outer_bounds[i] = curr_mid_policy_values

#     max_avg_dist = -1
#     max_ind = -1
#     for i in range(len(inner_bounds)):
#         curr_avg_dist = sum([math.dist(inner_bounds[i], voter.ideal_policy.values) for voter in voters])/len(voters)
#         if curr_avg_dist > max_avg_dist:
#             max_avg_dist = curr_avg_dist
#             max_ind = i

#     # plotting all voters
#     original_color = "green" if f_num % 2 == 0 else "orange"
#     new_color = "orange" if f_num % 2 == 0 else "green"
#     new_policy = Policy(inner_bounds[max_ind])
#     votes = electorate.obtain_individual_votes(current_policy, new_policy)
#     colors = []
#     for i in range(len(voters)):
#         voter = voters[i]
#         if votes[i] == 0:
#             colors.append(original_color)
#         elif votes[i] == 1:
#             colors.append(new_color)
#         else:
#             colors.append('yellow')
#     plt.scatter(
#         [voter.ideal_policy.values[0] for voter in voters], 
#         [voter.ideal_policy.values[1] for voter in voters], 
#         color=colors, 
#         marker='o'
#     )

#     # plotting policies
#     plt.scatter(
#         [current_policy.values[0]],
#         [current_policy.values[1]],
#         color=original_color, 
#         # marker=blue_marker, 
#         edgecolors='black',
#         s=200,
#     )

#     plt.scatter(
#         [new_policy.values[0]],
#         [new_policy.values[1]], 
#         color=new_color, 
#         # marker=blue_marker, 
#         edgecolors='black',
#         s=200,
#     )

#     policies_at_step[f_num+1] = new_policy
#     plt.title(f"Frame {f_num}")
#     print(f"Frame {f_num} created")

# def frame_gen():
#     i = 0
#     while True:
#         if policies_at_step[i].values[0] > 100 or policies_at_step[i].values[1] > 100:
#             break
#         yield i
#         i += 1

# ani = animation.FuncAnimation(fig, make_frame, frames=frame_gen())

# # Save to mp4 (requires ffmpeg installed and available)
# ani.save("output.mp4", writer='ffmpeg', fps=1)