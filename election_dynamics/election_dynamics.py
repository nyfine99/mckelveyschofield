import matplotlib.animation as animation
import matplotlib.pyplot as plt
from datetime import datetime
import math
from ordered_set import OrderedSet
import numpy as np
import random

from abc import ABC, abstractmethod

from utility_functions.evaluation_functions import status_quo_preference
from voters.simple_voter import SimpleVoter
from voters.voter import Voter
from policies.policy import Policy

"""
A file which defines the ElectionDynamics abstract class and extending classes. 
These classes maintain the attributes of Voters across an electorate,
as well as functions useful in determining which policies would succeed among them.
"""

class ElectionDynamics(ABC):
    """
    The ElectionDynamics abstract class. Voters is a list of Voters, while the evaluation_function
    field holds a function which takes in all of the policies and voters, and outputs the successful policy
    or proportional distribution of seats among the policy/parties.
    """
    def __init__(self, voters: list[Voter], evaluation_function: callable):
        self.voters = voters
        self.evaluation_function = evaluation_function


    @abstractmethod
    def tabulate_votes(self):
        pass


    @abstractmethod
    def compare_policies(self):
        pass


class ElectionDynamicsTwoParty(ElectionDynamics):
    def __init__(
        self, 
        voters: list[Voter], 
        evaluation_function: callable, 
        tiebreak_func: callable = None,
        issue_1: str = "Issue 1",
        issue_2: str = "Issue 2",
    ):
        self.voters = voters
        self.evaluation_function = evaluation_function
        self.tiebreak_func = tiebreak_func  # used when a voter is ambivalent to distribute their vote
        self.issue_1 = issue_1  # Issue 1 name
        self.issue_2 = issue_2  # Issue 2 name

    def compare_policies(self, original_policy: Policy, new_policy: Policy):
        return self.evaluation_function(self.tabulate_votes(original_policy, new_policy))
    
    def tabulate_votes(self, original_policy: Policy, new_policy: Policy):
        counts = [0,0] # index 0 represents the original_policy count, index 1 the new policy
        votes = self.obtain_individual_votes(original_policy, new_policy)
        counts = [np.sum(votes == 0), np.sum(votes == 1)]
        return counts
    
    def obtain_individual_votes(self, original_policy: Policy, new_policy: Policy) -> np.array:
        # it could be more performant to take this out of numpy
        # but numpy should certainly be used for the overriding functions
        original_utilities = np.array([voter.get_utility(original_policy) for voter in self.voters])
        new_utilities = np.array([voter.get_utility(new_policy) for voter in self.voters])
        votes = np.full(len(self.voters), -1)
        votes[original_utilities > new_utilities] = 0
        votes[original_utilities < new_utilities] = 1
        if self.tiebreak_func is not None:
            ties = (original_utilities == new_utilities)
            votes[ties] = [self.tiebreak_func() for _ in range(np.sum(ties))]
        return votes
    
    def plot_election_2d(self, original_policy: Policy, new_policy: Policy, verbose: bool = True):
        # initialize the figure and axes
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.3, 0.55, 0.55])  # Shrink plot inside the figure

        # some settings
        original_policy_name = "Original Policy" if original_policy.name is None else original_policy.name
        new_policy_name = "New Policy" if original_policy.name is None else new_policy.name
        undecided_name = "Undecided"
        original_color = "blue"
        new_color = "red"
        undecided_color = "yellow"

        # plotting all voters
        votes = self.obtain_individual_votes(original_policy, new_policy)
        voters_by_vote = {"original": [], "new": [], "undecided": []}
        policy_colors = {"original": original_color, "new": new_color, "undecided": undecided_color}
        policy_names = {"original": original_policy_name, "new": new_policy_name, "undecided": undecided_name}
        for i, vote in enumerate(votes):
            if vote == 0:
                voters_by_vote["original"].append(self.voters[i])
            elif vote == 1:
                voters_by_vote["new"].append(self.voters[i])
            else:
                voters_by_vote["undecided"].append(self.voters[i])
        
        for k in voters_by_vote.keys():
            if voters_by_vote[k]:
                arr = np.array([voter.ideal_policy.values for voter in voters_by_vote[k]])
                voters_plot = ax.scatter(
                    arr[:, 0], 
                    arr[:, 1], 
                    c=policy_colors[k], 
                    marker='o',
                )
                curr_name = policy_names[k]
                voters_plot.set_label(f"{curr_name} Voters")
        
        # plotting policies and differentiating winner from loser
        winner = self.compare_policies(original_policy,new_policy)
        original_marker = '*' if winner == 0 else 'X'
        original_size = 250 if winner == 0 else 150
        new_marker = '*' if winner == 1 else 'X'
        new_size = 250 if winner == 1 else 150
        original_policy_plot = ax.scatter(
            [original_policy.values[0]],
            [original_policy.values[1]], 
            color=original_color, 
            marker=original_marker, 
            edgecolors='black',
            s=original_size
        )
        original_policy_plot.set_label(original_policy_name)
        new_policy_plot = ax.scatter(
            [new_policy.values[0]],
            [new_policy.values[1]], 
            color=new_color, 
            marker=new_marker, 
            edgecolors='black', 
            s=new_size
        )
        new_policy_plot.set_label(new_policy_name)

        # title, labels, legend
        desired_order = [
            original_policy_name,
            new_policy_name,
            f'{original_policy_name} Voters',  
            f'{new_policy_name} Voters',  
            f'{undecided_name} Voters',  
        ]
        handles, labels = plt.gca().get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        ordered_handles = [label_to_handle[label] for label in desired_order if label in label_to_handle]
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., handles=ordered_handles, labels=desired_order)
        plt.title(f'{original_policy_name} vs {new_policy_name}: Election Results')
        plt.xlabel(f'Position on {self.issue_1}')
        plt.ylabel(f'Position on {self.issue_2}')

        # details
        if verbose:
            winner_text = original_policy_name if winner == 0 else new_policy_name
            vote_totals = self.tabulate_votes(original_policy, new_policy)
            original_policy_votes = vote_totals[0]
            new_policy_votes = vote_totals[1]
            fig.text(0, 0.05, f"""
                        {original_policy_name} (blue) received {original_policy_votes} votes.
                        {new_policy_name} (red) received {new_policy_votes} votes.
                        So, {winner_text} wins!""",
                        fontsize=9, color='black')

        plt.grid(True)
        plt.show()

    def plot_mckelvey_schofield_path(
        self, 
        original_policy: Policy, 
        goal_policy: Policy, 
        path: list[Policy],
        save_file: str = None,
    ):
        # plotting the path
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.3, 0.55, 0.55])  # Shrink plot inside the figure

        # some settings
        original_policy_name = "Original Policy" if original_policy.name is None else original_policy.name
        goal_policy_name = "New Policy" if original_policy.name is None else goal_policy.name
        original_color = "blue"
        goal_color = "red"

        # plotting all voters
        votes = self.obtain_individual_votes(original_policy, goal_policy)
        
        voters_plot = ax.scatter(
            [voter.ideal_policy.values[0] for voter in self.voters], 
            [voter.ideal_policy.values[1] for voter in self.voters], 
            c='black', 
            marker='o'
        )
        voters_plot.set_label(f"Voters")
        
        # plotting path
        ax.quiver(
            [p.values[0] for p in path[:-1]], 
            [p.values[1] for p in path[:-1]], 
            [path[i+1].values[0] - path[i].values[0] for i in range(len(path)-1)], 
            [path[i+1].values[1] - path[i].values[1] for i in range(len(path)-1)],
            angles='xy', scale_units='xy', scale=1, color='green', alpha=0.7
        )
        

        # plotting policies
        intermediate_plot = ax.scatter(
            [p.values[0] for p in path[1:-1]],
            [p.values[1] for p in path[1:-1]], 
            color="green",
            marker='o',
            s=200,
            alpha=0.7,
        )
        if path[-1].values != goal_policy.values:
            # if the last policy in the path is not the goal policy, plot it as well
            ax.scatter(
                [path[-1].values[0]],
                [path[-1].values[1]], 
                color="green",
                marker='o',
                s=200,
                alpha=0.7,
            )
        intermediate_plot.set_label("Intermediate Policies")

        original_policy_plot = ax.scatter(
            [original_policy.values[0]],
            [original_policy.values[1]], 
            color=original_color,
            edgecolors='black',
            marker='X',
            s=200,
        )
        original_policy_plot.set_label(original_policy_name)

        goal_policy_plot = ax.scatter(
            [goal_policy.values[0]],
            [goal_policy.values[1]], 
            color=goal_color,
            edgecolors='black',
            marker='*',
            s=200,
        )
        goal_policy_plot.set_label(goal_policy_name)

        # title, labels, legend
        desired_order = [
            original_policy_name,
            goal_policy_name,
            'Intermediate Policies',
            'Voters'
        ]
        handles, labels = plt.gca().get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        ordered_handles = [label_to_handle[label] for label in desired_order]
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., handles=ordered_handles, labels=desired_order)
        title = f'Path from {original_policy_name} to {goal_policy_name}'
        if not np.allclose(path[-1].values, goal_policy.values):
            # the goal policy was not reached, and the title should reflect this
            title = f'Attempted Path from {original_policy_name} to {goal_policy_name}'
        plt.title(title)
        plt.xlabel(f'Position on {self.issue_1}')
        plt.ylabel(f'Position on {self.issue_2}')

        plt.grid(True)
        if save_file is not None:
            plt.savefig(save_file, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)


class ElectionDynamicsTwoPartySimpleVoters(ElectionDynamicsTwoParty):
    def __init__(
        self, 
        voters: list[SimpleVoter],
        issue_1: str = "Issue 1",
        issue_2: str = "Issue 2",
    ):
        self.voters = voters
        self.voter_arr = np.array([voter.ideal_policy.values for voter in self.voters])
        self.evaluation_function = status_quo_preference
        self.tiebreak_func = None  # used when a voter is ambivalent to distribute their vote
        self.issue_1 = issue_1  # Issue 1 name
        self.issue_2 = issue_2  # Issue 2 name

    def obtain_individual_votes(self, original_policy: Policy, new_policy: Policy) -> np.array:
        original_utilities = -np.linalg.norm(self.voter_arr - original_policy.values, axis=1)
        new_utilities = -np.linalg.norm(self.voter_arr - new_policy.values, axis=1)
        votes = np.full(len(self.voters), -1)
        votes[original_utilities > new_utilities] = 0
        votes[original_utilities < new_utilities] = 1
        if self.tiebreak_func is not None:
            ties = (original_utilities == new_utilities)
            votes[ties] = [self.tiebreak_func() for _ in range(np.sum(ties))]
        return votes

    def mckelvey_schofield_greedy_avg_dist(
        self,
        current_policy
    ) -> Policy:
        """
        Select the next policy using a greedy algorithm - choose the policy with the highest average distance
        from all voters which beats the current policy.
        """
        dists = np.linalg.norm(self.voter_arr - current_policy.values, axis=1)
        max_dist = np.max(dists)
        max_r = 2.0 * max_dist
        number_of_points = 360
        angles = np.linspace(0, 2 * np.pi, number_of_points, endpoint=False)
        inner_bounds = [current_policy.values.copy() for _ in angles]
        outer_bounds = [current_policy.values + max_r * np.array([np.sin(angle), np.cos(angle)]) for angle in angles]

        num_halving_iterations = 12
        for i in range(len(inner_bounds)):
            for _ in range(num_halving_iterations):
                curr_mid_policy_values = [
                    (inner_bounds[i][0] + outer_bounds[i][0])/2, 
                    (inner_bounds[i][1] + outer_bounds[i][1])/2
                ]
                curr_mid_policy = Policy(curr_mid_policy_values)
                if self.compare_policies(current_policy, curr_mid_policy):
                    # curr_mid policy won
                    inner_bounds[i] = curr_mid_policy_values
                else:
                    outer_bounds[i] = curr_mid_policy_values

        max_avg_dist = -1
        max_ind = -1
        for i, bound in enumerate(inner_bounds):
            if np.allclose(bound, current_policy.values):
                # This policy is not actually on the indifference curve, but rather is the current policy;
                # so, skip it
                continue
            avg_dist = np.mean(np.linalg.norm(self.voter_arr - bound, axis=1))
            if avg_dist > max_avg_dist:
                max_avg_dist = avg_dist
                max_ind = i

        return inner_bounds[max_ind]
    

    def mckelvey_schofield_greedy_with_adjustment_avg_dist(
        self,
        current_policy,
        policy_path
    ) -> Policy:
        """
        Select the next policy using a greedy algorithm - choose the policy with the highest average distance
        from all voters which beats the current policy.
        """
        max_dist = max([math.dist(current_policy.values, voter.ideal_policy.values) for voter in self.voters])
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
                if self.compare_policies(current_policy, curr_mid_policy):
                    # curr_mid policy won
                    inner_bounds[i] = curr_mid_policy_values
                else:
                    outer_bounds[i] = curr_mid_policy_values

        max_avg_dist = -1
        max_ind = -1
        for i in range(len(inner_bounds)):
            if inner_bounds[i] == current_policy.values:
                # This policy is not actually on the indifference curve, so skip it
                continue
            curr_avg_dist = sum([math.dist(inner_bounds[i], voter.ideal_policy.values) for voter in self.voters])/len(self.voters)
            if curr_avg_dist > max_avg_dist:
                max_avg_dist = curr_avg_dist
                max_ind = i

        poss_next_policy = inner_bounds[max_ind]
        if len(policy_path) > 1:
            average_policy_gap = np.mean([math.dist(policy_path[i].values, policy_path[i-1].values) for i in range(1,len(policy_path))])
            gap_tolerance = 0.1
            forced_movement_factor = 0.5
            if math.dist(poss_next_policy, current_policy.values) < gap_tolerance * average_policy_gap:  # this will likely cycle, which we want to avoid
                # minimum distance allowed between the current policy and the next policy; doing this to create significant movement
                minimum_dist = average_policy_gap * forced_movement_factor
                greater_than_minimum = [p for p in inner_bounds if math.dist(p, current_policy.values) >= minimum_dist]
                if greater_than_minimum != []:
                    poss_next_policy = random.choice(greater_than_minimum)
                else:
                    poss_next_policy = random.choice(inner_bounds)

        return poss_next_policy
    
    def animate_mckelvey_schofield(
        self, 
        original_policy, 
        goal_policy, 
        max_steps=50, 
        step_selection_function="mckelvey_schofield_greedy_with_adjustment_avg_dist",
        output_folder="output",
        filename=f"output",
        verbose=True,
        fps=0.5,
    ):
        policy_path = [original_policy]  # Initialize the path with the original policy
        fig = plt.figure()

        original_policy_color = "blue"
        original_policy_name = original_policy.name if original_policy.name is not None else "Original Policy"
        goal_policy_color = "red"
        goal_policy_name = goal_policy.name if goal_policy.name is not None else "Goal Policy"

        def make_frame(f_num):
            if verbose:
                print(f"Starting to create frame {f_num+1}")
                overall_s_time = datetime.now()

            plt.clf()  # Clear the current axes/figure
            fig.add_axes([0.1, 0.3, 0.55, 0.55])
            current_policy = policy_path[f_num]
            s_time = datetime.now()
            if step_selection_function == "mckelvey_schofield_greedy_with_adjustment_avg_dist":
                new_policy = goal_policy if self.compare_policies(current_policy, goal_policy) == 1 else Policy(self.mckelvey_schofield_greedy_with_adjustment_avg_dist(current_policy, policy_path))
            elif step_selection_function == "mckelvey_schofield_greedy_avg_dist":
                new_policy = goal_policy if self.compare_policies(current_policy, goal_policy) == 1 else Policy(self.mckelvey_schofield_greedy_avg_dist(current_policy))
            else:
                raise ValueError(f"Unknown step selection function: {step_selection_function}")
            e_time = datetime.now()
            print(e_time - s_time)

            # initial plot settings
            current_color = "green" if f_num % 2 == 0 else "orange"
            new_color = "orange" if f_num % 2 == 0 else "green"
            undecided_color = "yellow"
            current_policy_name = f"Policy {f_num}"
            new_policy_name = f"Policy {f_num+1}"
            undecided_name = "Undecided"

            original_policy_opacity = 0.5
            goal_policy_opacity = 0.5

            if f_num == 0:
                current_color = original_policy_color
                current_policy_name = original_policy_name
                original_policy_opacity = 1.0
            if new_policy == goal_policy:
                new_color = goal_policy_color
                new_policy_name = goal_policy_name
                goal_policy_opacity = 1.0

            # plotting all voters
            votes = self.obtain_individual_votes(current_policy, new_policy)
            voters_by_vote = {"current": [], "new": [], "undecided": []}
            policy_colors = {"current": current_color, "new": new_color, "undecided": undecided_color}
            policy_names = {"current": current_policy_name, "new": new_policy_name, "undecided": undecided_name}
            for i in range(len(self.voters)):
                if votes[i] == 0:
                    voters_by_vote["current"].append(self.voters[i])
                elif votes[i] == 1:
                    voters_by_vote["new"].append(self.voters[i])
                else:
                    voters_by_vote["undecided"].append(self.voters[i])
            
            for k in voters_by_vote.keys():
                voters_plot = plt.scatter(
                    [voter.ideal_policy.values[0] for voter in voters_by_vote[k]], 
                    [voter.ideal_policy.values[1] for voter in voters_by_vote[k]], 
                    c=policy_colors[k], 
                    marker='o'
                )
                curr_name = policy_names[k]
                voters_plot.set_label(f"{curr_name} Voters")

            # plotting policies
            original_policy_plot = plt.scatter(
                [original_policy.values[0]],
                [original_policy.values[1]], 
                color=original_policy_color,
                edgecolors='black',
                marker='X',
                s=200,
                alpha=original_policy_opacity,
            )
            original_policy_plot.set_label(original_policy_name)

            goal_policy_plot = plt.scatter(
                [goal_policy.values[0]],
                [goal_policy.values[1]], 
                color=goal_policy_color,
                edgecolors='black',
                marker='*',
                s=200,
                alpha=goal_policy_opacity,
            )
            goal_policy_plot.set_label(goal_policy_name)

            if f_num != 0:
                current_policy_plot = plt.scatter(
                    [current_policy.values[0]],
                    [current_policy.values[1]],
                    color=current_color, 
                    edgecolors='black',
                    s=200,
                )
                current_policy_plot.set_label(current_policy_name)

            if new_policy != goal_policy:
                new_policy_plot = plt.scatter(
                    [new_policy.values[0]],
                    [new_policy.values[1]], 
                    color=new_color, 
                    edgecolors='black',
                    s=200,
                )
                new_policy_plot.set_label(new_policy_name)

            policy_path.append(new_policy)

            # title, labels, legend
            desired_order = [
                original_policy_name,
                goal_policy_name,
                current_policy_name,
                new_policy_name,
                f'{current_policy_name} Voters',  
                f'{new_policy_name} Voters',  
                f'{undecided_name} Voters',  
            ]
            desired_order = OrderedSet(desired_order)  # removing duplicates on first and last frames
            handles, labels = plt.gca().get_legend_handles_labels()
            label_to_handle = dict(zip(labels, handles))
            ordered_handles = [label_to_handle[label] for label in desired_order]
            plt.title(f'Path from {original_policy_name} to {goal_policy_name},\n Step {f_num+1}')
            plt.xlabel(f'Position on {self.issue_1}')
            plt.ylabel(f'Position on {self.issue_2}')
            plt.legend(
                loc='upper left', 
                bbox_to_anchor=(1.05, 1), 
                borderaxespad=0., 
                handles=ordered_handles, 
                labels=desired_order
            )

            # details
            if verbose:
                vote_totals = self.tabulate_votes(current_policy, new_policy)
                current_policy_votes = vote_totals[0]
                new_policy_votes = vote_totals[1]
                fig.text(
                    0, 
                    0.05, 
                    f"""
                    The agenda setter pits {current_policy_name} (the current policy) against {new_policy_name}.
                    {current_policy_name} receives {current_policy_votes} votes.
                    {new_policy_name} receives {new_policy_votes} votes.
                    So, {new_policy_name} defeats {current_policy_name}, and {new_policy_name} is adopted.
                    """,
                    fontsize=9, color='black'
                )
                overall_e_time = datetime.now()
                print(overall_e_time - overall_s_time)
                print(f"Frame {f_num+1} created")

        def frame_gen():
            f_num = 0
            while True:
                # TODO: check if off by one at all in max_steps condition
                if policy_path[f_num] == goal_policy or f_num >= max_steps:
                    if verbose:
                        if f_num >= max_steps:
                            print(f"Could not reach the goal policy after {max_steps} steps.")
                        else:
                            print("Reached the goal policy!")
                    break
                yield f_num
                f_num += 1

        def init():
            return None

        ani = animation.FuncAnimation(fig, make_frame, frames=frame_gen(), init_func=init)
        # Save to mp4
        ani.save(f"{output_folder}/{filename}.mp4", writer='ffmpeg', fps=fps)
        plt.close(fig)
        return policy_path
    
    def plot_path_average_distances(
        self, 
        path: list[Policy],
        max_steps: int,
        output_folder="output",
        filename=f"output",
    ):
        # plot settings
        fig = plt.figure(figsize=(12, 8))

        # obtaining average values
        distances = []
        for p in path:
            curr_avg_dist = sum([math.dist(p.values, voter.ideal_policy.values) for voter in self.voters])/len(self.voters)
            distances.append(curr_avg_dist)

        # creating plot
        plt.plot(
            [i for i in range(len(distances))],
            distances, 
            color='black', 
            linestyle='-', 
            marker='o'
        )

        title = 'Average Distance Values for Policies Along the Path'
        plt.title(title)
        plt.xlim(right=max_steps)  # adjust the right leaving left unchanged
        plt.xlabel(f'Policy')
        plt.ylabel(f'Average Distance from Voter Preferences')
        plt.savefig(f"{output_folder}/{filename}", bbox_inches='tight')
        plt.close(fig)

class ElectionDyanamicsMultiParty(ElectionDynamics):
    def __init__(self, voters: list[Voter], evaluation_function: callable):
        self.voters = voters
        self.evaluation_function = evaluation_function

    def compare_policies(self, policies: list[Policy]):
        return self.evaluation_function(self.voters, policies)
    