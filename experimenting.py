from random import randint, gauss

import math
import numpy as np
import random
import matplotlib.pyplot as plt

from policies.policy import Policy
from voters.voter import Voter
from utility_functions.utility_functions import neg_distance, neg_distance_with_limit
from election_dynamics.electoral_systems import create_two_party_status_quo_preference

# p1 = Policy([1,6,7])
# p2 = Policy([4,9,4])
# p3 = Policy([-1,-6,-7])

# v = Voter(Policy([0,0,0]))

# print(v.preferred_policy(p1,p2))
# print(v.preferred_policy(p1,p3))

p1 = Policy([1,6])
p2 = Policy([4,9])
p3 = Policy([-6,-7])

v1 = Voter(Policy([0,0]), utility_function=neg_distance)
v2 = Voter(Policy([1,1]), utility_function=neg_distance)

# electorate = create_two_party_status_quo_preference([v1,v2])
# electorate.plot_election_2d(p2, p1)

# p1 = Policy([50,50], "Derek sometimes, Boniuk sometimes") # moderate
# p2 = Policy([70,100], "Derek often, Boniuk always") # extreme

# voters = []
# for i in range(250):
#     voters.append(
#         Voter(
#             Policy(
#                 [
#                     random.sample([i + 70 for i in range(30)] + [i for i in range(100)], 1)[0], 
#                     random.sample([i + 70 for i in range(30)] + [2], 1)[0]
#                 ]
#             ), 
#             utility_function=neg_distance_with_limit(100)
#         )
#     )
#     # voters.append(Voter(Policy([randint(0,100),randint(0,100)]), utility_function=neg_distance_with_limit(25)))
#     # voters.append(Voter(Policy([gauss(50,15),gauss(50,15)]), utility_function=neg_distance_with_limit(25)))

# electorate = create_two_party_status_quo_preference(voters, "Derekkkkk", "BONIUK")
# electorate.plot_election_2d(p1, p2)




# voters = [
#     Voter(Policy([43, 30])),
#     Voter(Policy([46, 31])),
#     Voter(Policy([40, 40])),
#     Voter(Policy([46, 53])),
#     Voter(Policy([48, 53])),
#     Voter(Policy([44, 53])),
#     Voter(Policy([50, 40])),
# ]

SEED = 53
NUM_VOTERS = 100
GOAL_POLICY = [100, 100]
random.seed(SEED)  # 8 is lopsided, as is 9, though in a different direction

voters = []
for i in range(100):
    voters.append(Voter(Policy([gauss(50,15),gauss(50,10)]), utility_function=neg_distance))

electorate = create_two_party_status_quo_preference(voters, "Cats", "Dogs")
# electorate.plot_election_2d(p1, p2)

average_policy = [
    sum([voter.ideal_policy.values[0] for voter in voters])/len(voters), 
    sum([voter.ideal_policy.values[1] for voter in voters])/len(voters)
]
p1, p2 = Policy(average_policy), Policy(GOAL_POLICY)

max_dist = max([math.dist(p1.values, voter.ideal_policy.values) for voter in voters])
avg_dist = sum([math.dist(p1.values, voter.ideal_policy.values) for voter in voters])/len(voters)
print(max_dist)
print(avg_dist)
max_r = 2.0 * max_dist

original_policy = p1
new_policy = p2
inner_bounds = []
outer_bounds = []
number_of_points = 360
angles = [math.pi * 2.0 * n/(float(number_of_points)) for n in range(number_of_points)]
for angle in angles:
    inner_bounds.append(p1.values)
    outer_bounds.append([p1.values[0] + max_r * math.sin(angle), p1.values[1] + max_r * math.cos(angle)])

num_halving_iterations = 10
for i in range(len(inner_bounds)):
    for j in range(num_halving_iterations):
        curr_mid_policy_values = [
            (inner_bounds[i][0] + outer_bounds[i][0])/2, (inner_bounds[i][1] + outer_bounds[i][1])/2
        ]
        curr_mid_policy = Policy(curr_mid_policy_values)
        if electorate.compare_policies(original_policy, curr_mid_policy):
            # curr_mid policy won
            inner_bounds[i] = curr_mid_policy_values
        else:
            outer_bounds[i] = curr_mid_policy_values

votes = electorate.obtain_individual_votes(original_policy, new_policy)
fig = plt.figure(figsize=(6, 4))
ax = fig.add_axes([0.2, 0.3, 0.6, 0.6])  # Shrink plot inside the figure

# heatmap
heatmap = []
heatmap_colors = []
for i in range(0,100,1):
    for j in range(0,100,1):
        heatmap.append([i,j])
        heatmap_colors.append(sum([math.dist([i,j], voter.ideal_policy.values) for voter in voters])/len(voters))

ax.scatter(
    [heat[0] for heat in heatmap], 
    [heat[1] for heat in heatmap], 
    c=heatmap_colors, 
    marker='o'
)

# plotting all voters
colors = []
for i in range(len(electorate.voters)):
    voter = electorate.voters[i]
    colors.append('black')
    # if votes[i] == 0:
    #     colors.append('blue')
    # elif votes[i] == 1:
    #     colors.append('red')
    # else:
    #     colors.append('yellow')

ax.scatter(
    [voter.ideal_policy.values[0] for voter in electorate.voters], 
    [voter.ideal_policy.values[1] for voter in electorate.voters], 
    color=colors, 
    marker='o'
)

# plotting policies and differentiating winner from loser
winner = electorate.compare_policies(original_policy,new_policy)
blue_marker = '*' if winner == 0 else 'X'
blue_size = 250 if winner == 0 else 150
red_marker = '*' if winner == 1 else 'X'
red_size = 250 if winner == 1 else 150
ax.scatter(
    [original_policy.values[0]],
    [original_policy.values[1]], 
    color='blue', 
    # marker=blue_marker, 
    edgecolors='black',
    # s=blue_size
)
ax.scatter(
    [new_policy.values[0]],
    [new_policy.values[1]], 
    color='red', 
    # marker=red_marker, 
    edgecolors='black', 
    # s=red_size
)

# plotting bounds
# this bound is the furthest found solution, in each direction, which is preferable to the current policy
ax.plot(
    [bound[0] for bound in inner_bounds] + [inner_bounds[0][0]],
    [bound[1] for bound in inner_bounds] + [inner_bounds[0][1]],
    color='green'
)
# ax.scatter(
#     [bound[0] for bound in inner_bounds],
#     [bound[1] for bound in inner_bounds],
#     color='green'
# )
# ax.scatter(
#     [bound[0] for bound in outer_bounds],
#     [bound[1] for bound in outer_bounds],
#     color='black'
# )

max_avg_dist = -1
max_ind = -1
for i in range(len(inner_bounds)):
    curr_avg_dist = sum([math.dist(inner_bounds[i], voter.ideal_policy.values) for voter in voters])/len(voters)
    if curr_avg_dist > max_avg_dist:
        max_avg_dist = curr_avg_dist
        max_ind = i

current_policy = Policy(inner_bounds[max_ind])
ax.scatter(
    [current_policy.values[0]],
    [current_policy.values[1]], 
    color='orange', 
    # marker=blue_marker, 
    edgecolors='black',
    # s=blue_size
)


# def select_next_policy_max_avg_dist() - define the policy for selecting the next point so it can be swapped out


n_steps = 20
for k in range(1,n_steps):
    if electorate.compare_policies(current_policy, new_policy):
        print("The new policy has won!")
        break

    print(f"Current step: {k}")
    avg_dist = sum([math.dist(current_policy.values, voter.ideal_policy.values) for voter in voters])/len(voters)
    print(f"Average distance: {avg_dist}")
    max_dist = max([math.dist(current_policy.values, voter.ideal_policy.values) for voter in voters])
    max_r = 2.0 * max_dist
    inner_bounds = []
    outer_bounds = []
    number_of_points = 360
    angles = [math.pi * 2.0 * n/(float(number_of_points)) for n in range(number_of_points)]
    for angle in angles:
        inner_bounds.append(current_policy.values)
        outer_bounds.append([current_policy.values[0] + max_r * math.sin(angle), current_policy.values[1] + max_r * math.cos(angle)])

    num_halving_iterations = 12
    for i in range(len(inner_bounds)):
        for j in range(num_halving_iterations):
            curr_mid_policy_values = [
                (inner_bounds[i][0] + outer_bounds[i][0])/2, (inner_bounds[i][1] + outer_bounds[i][1])/2
            ]
            curr_mid_policy = Policy(curr_mid_policy_values)
            if electorate.compare_policies(current_policy, curr_mid_policy):
                # curr_mid policy won
                inner_bounds[i] = curr_mid_policy_values
            else:
                outer_bounds[i] = curr_mid_policy_values

    green_intensity = k/n_steps
    ax.plot(
        [bound[0] for bound in inner_bounds] + [inner_bounds[0][0]],
        [bound[1] for bound in inner_bounds] + [inner_bounds[0][1]],
        color=(0, green_intensity, 0)
    )

    avg_dists = [sum([math.dist(inner_bounds[i], voter.ideal_policy.values) for voter in voters])/len(voters) for i in range(len(inner_bounds))]
    sorted(zip(inner_bounds, avg_dists))
    # idea - take the point with the maximum average distance from the voters' ideal policies,
    # at a 120 degree angle or more from the current policy

    # max_avg_dist = -1
    # max_ind = -1
    # for i in range(len(inner_bounds)):
    #     curr_avg_dist = sum([math.dist(inner_bounds[i], voter.ideal_policy.values) for voter in voters])/len(voters)
    #     if curr_avg_dist > max_avg_dist:
    #         max_avg_dist = curr_avg_dist
    #         max_ind = i

    current_policy = Policy(inner_bounds[max_ind])
    ax.scatter(
        [current_policy.values[0]],
        [current_policy.values[1]], 
        color='orange', 
        # marker=blue_marker, 
        edgecolors='black',
        # s=blue_size
    )
    # if max_avg_dist > 18.57259:
    #     import pdb; pdb.set_trace()




# title and display
plt.title(f'Perferred Policy Boundary, Random Seed {SEED}')
plt.xlabel(f'Position on {electorate.issue_1}')
plt.ylabel(f'Position on {electorate.issue_2}')

plt.grid(True)
plt.show()