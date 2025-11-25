"""
Two-Party Electoral Dynamics with Simple Voters Implementation

This module implements a two-party electoral system with simple (Euclidean) voters specifically.
Since the utility function of the voters is known, it can be built into the various functions
of this module for greater efficiency.

Key Features:
- More efficient implementation of some functions from the parent class, optimized for taxicab voters
- Winset boundary generation, specific to taxicab voters
- McKelvey-Schofield path creation
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import numpy as np
import random


from election_dynamics.election_dynamics_two_party import ElectionDynamicsTwoParty
from utility_functions.evaluation_functions import status_quo_preference
from voters.simple_voter import SimpleVoter
from policies.policy import Policy


class ElectionDynamicsTwoPartySimpleVoters(ElectionDynamicsTwoParty):
    """
    The ElectionDynamicsTwoPartySimpleVoters class. Voters is a list of SimpleVoters, while the evaluation_function
    is fixed as status_quo_preference. An additional field, voter_arr, is a numpy array of the ideal policies of the
    voters, and is used to speed up calculations.
    """

    def __init__(
        self,
        voters: list[SimpleVoter],
        issue_1: str = "Issue 1",
        issue_2: str = "Issue 2",
    ):
        """
        Initialize the two-party taxicab voters electoral dynamics system.
        
        Note that the voter_arr and calculated_utilities fields are added on top of those provided
        by the base class. This enables more efficient calculation later on.
        """
        self.voters = voters
        self.voter_arr = np.array([voter.ideal_policy.values for voter in self.voters])
        self.calculated_utilities = {}
        self.evaluation_function = status_quo_preference
        self.tiebreak_func = (
            None  # used when a voter is ambivalent to distribute their vote
        )
        self.issue_1 = issue_1  # Issue 1 name
        self.issue_2 = issue_2  # Issue 2 name

    def obtain_individual_votes(
        self, original_policy: Policy, new_policy: Policy
    ) -> np.array:
        """
        Calculate individual voter preferences between two policies.
        
        This method computes how each individual voter would vote in a
        binary choice between two policies. It overrides the version provided
        by the parent class in favor of increased efficiency.
        
        Args:
            original_policy (Policy): The incumbent or status quo policy.
            new_policy (Policy): The challenger or proposed policy.
        
        Returns:
            np.array: Array of individual votes where:
                      - 0: voter prefers original_policy
                      - 1: voter prefers new_policy
                      - -1: voter is indifferent (abstains)
                      - Length equals the number of voters
        """
        original_utilities = self.get_policy_utilities(original_policy)
        new_utilities = self.get_policy_utilities(new_policy)
        votes = np.full(len(self.voters), -1)
        votes[original_utilities > new_utilities] = 0
        votes[original_utilities < new_utilities] = 1
        return votes

    def get_policy_utilities(self, policy: Policy) -> np.array:
        """
        Obtains the utility of the policy to each voter.

        Args:
            policy (Policy): the policy.

        Returns:
            np.array: an array where each element indicates the utility of the policy to the
                corresponding voter.
        """
        key = policy.id
        if key in self.calculated_utilities:
            utilities = self.calculated_utilities[key]
        else:
            utilities = -np.linalg.norm(self.voter_arr - policy.values, axis=1)
            self.calculated_utilities[key] = utilities
        return utilities

    def generate_winset_boundary(
        self,
        current_policy: Policy,
        n_directions=360,
        n_halving_iterations=12,
        angle_offset: float=0.0,
    ) -> np.ndarray:
        """
        Obtains the boundary of the policies which could beat the current_policy.
        An approximation of the boundary is obtained via binary searches in the specified number
        of directions.

        Args:
            current_policy (Policy): the current policy.
            n_directions (int): the number of directions in which to search for boundary points.
            n_halving_iterations (int): the number of iterations of binary search to run.
            angle_offset (float): the angle at which to offset the first point.
        
        Returns:
            np.ndarray: a matrix containing the boundary points.
        """

        voters_policy_values = self.voter_arr
        current_policy_values = current_policy.values  # shape (2,)
        current_policy_voter_dists = np.linalg.norm(
            voters_policy_values - current_policy_values, axis=1
        )

        # set up directions (360 vectors in circle)
        angles = np.linspace(
            angle_offset, 2 * np.pi + angle_offset, n_directions, endpoint=False
        )
        directions = np.stack(
            [np.sin(angles), np.cos(angles)], axis=1
        )  # shape (360, 2)

        # set up initial inner and outer bounds for winset boundary in each direction (shape: (360, 2))
        inner_bounds = np.tile(
            current_policy_values, (n_directions, 1)
        )  # each row is the current policy
        max_dist = np.max(current_policy_voter_dists)
        # outer bounds of preferable policies will be at most twice the maximum distance
        # from the current policy to any voter's ideal policy
        max_r = 2.0 * max_dist
        outer_bounds = inner_bounds + max_r * directions

        # vectorized binary search
        for _ in range(n_halving_iterations):
            mid_points = (inner_bounds + outer_bounds) / 2  # shape (360, 2)

            # broadcasted distance computation: (360, V), where V is the number of voters
            dists_to_mid = np.linalg.norm(
                mid_points[:, None, :] - voters_policy_values[None, :, :], axis=2
            )

            # compare who wins per direction
            wins = (dists_to_mid < current_policy_voter_dists).sum(axis=1) > (
                voters_policy_values.shape[0] // 2
            )  # shape (360,)

            # update bounds
            inner_bounds[wins] = mid_points[wins]
            outer_bounds[~wins] = mid_points[~wins]

        # inner bounds now contains points near the winset boundary which beat the current policy...
        # ... but must filter out the current policy itself
        mask = np.all(np.isclose(inner_bounds, current_policy.values), axis=1)
        filtered_boundary_points = inner_bounds[~mask]
        if len(filtered_boundary_points) == 0:
            raise ValueError("The winset boundary could not be found, and the current policy appears optimal!")
        return filtered_boundary_points

    def mckelvey_schofield_greedy_avg_dist(self, current_policy: Policy) -> Policy:
        """
        Select the next policy in the sequence using a greedy algorithm:
        choose the policy with the highest average distance
        from all voters which beats the current policy. 
        This will make it highly likely that, over time, the voters choose policies which are
        further away from any of their preferences, and thus that the goal policy will be able
        to win an election.

        Args:
            current_policy (Policy): the current policy enacted by the voters.

        Returns:
            Policy: a policy which will beat current_policy in an election.
        """
        boundary_points = self.generate_winset_boundary(current_policy)
        boundary_points_voters_deltas = (
            boundary_points[:, None, :] - self.voter_arr[None, :, :]
        )  # shape (360, V, 2)
        boundary_points_voters_dists = np.linalg.norm(
            boundary_points_voters_deltas, axis=2
        )  # shape (360, V)
        avg_boundary_points_voters_dists = boundary_points_voters_dists.mean(axis=1)
        arg_max = np.argmax(
            avg_boundary_points_voters_dists
        )  # index of the maximum average distance
        return Policy(
            boundary_points[arg_max]
        )  # return the policy with the maximum average distance

    def mckelvey_schofield_greedy_with_adjustment_avg_dist(
        self, current_policy: Policy, policy_path: list[Policy]
    ) -> Policy:
        """
        Select the next policy in the sequence using a mostly greedy algorithm:
        choose the policy with the highest average distance
        from all voters which beats the current policy.
        However, if that policy does not differ sufficiently from the current_policy,
        we choose a different policy on the winset boundary.
        
        This process makes getting stuck at a fixed point/local optimum less likely
        than the pure greedy algorithm.

        Args:
            current_policy (Policy): the current policy enacted by the voters.
            policy_path (list[Policy]): the path of policies that has been taken to reach the current_policy.

        Returns:
            Policy: a policy which will beat current_policy in an election.
        """
        boundary_points = self.generate_winset_boundary(current_policy)
        boundary_points_voters_deltas = (
            boundary_points[:, None, :] - self.voter_arr[None, :, :]
        )  # shape (360, V, 2)
        boundary_points_voters_dists = np.linalg.norm(
            boundary_points_voters_deltas, axis=2
        )  # shape (360, V)
        avg_boundary_points_voters_dists = boundary_points_voters_dists.mean(axis=1)
        arg_max = np.argmax(
            avg_boundary_points_voters_dists
        )  # index of the maximum average distance

        poss_next_policy_values = boundary_points[
            arg_max
        ]  # the policy with the maximum average distance
        if len(policy_path) > 1:
            # policy_path_arr = np.stack([p.values for p in policy_path])
            policy_path_arr = np.array([p.values for p in policy_path])  # shape (N, D)
            gaps = np.linalg.norm(policy_path_arr[1:] - policy_path_arr[:-1], axis=1)
            average_policy_gap = gaps.mean()
            gap_tolerance = 0.1
            forced_movement_factor = 0.5
            if (
                math.dist(poss_next_policy_values, current_policy.values)
                < gap_tolerance * average_policy_gap
            ):  # this will likely cycle, which we want to avoid
                # minimum distance allowed between the current policy and the next policy; doing this to create significant movement
                minimum_dist = average_policy_gap * forced_movement_factor
                dists_to_current = np.linalg.norm(
                    boundary_points - current_policy.values, axis=1
                )
                greater_than_minimum = boundary_points[dists_to_current >= minimum_dist]
                if greater_than_minimum.size != 0:
                    poss_next_policy_values = random.choice(greater_than_minimum)
                else:
                    poss_next_policy_values = random.choice(boundary_points)

        return Policy(
            poss_next_policy_values
        )  # return the policy with the maximum average distance or a forced movement policy

    def mckelvey_schofield_greedy_with_lookahead(self, current_policy: Policy) -> Policy:
        """
        Select the next policy using a greedy algorithm with a lookahead:
        - Choose the policy on the winset boundary with the maximum average distance from
            all voters.
        - Choose several other policies on the winset boundary at random.
        - From this set, choose whichever policy has the policy with the maximum average 
            distance from all voters on its winset boundary.

        This process makes getting stuck at a fixed point/local optimum less likely
        than the pure greedy algorithm.

        Args:
            current_policy (Policy): the current policy enacted by the voters.

        Returns:
            Policy: a policy which will beat current_policy in an election.
        """
        boundary_points = self.generate_winset_boundary(current_policy)
        boundary_points_voters_deltas = (
            boundary_points[:, None, :] - self.voter_arr[None, :, :]
        )  # shape (360, V, 2)
        boundary_points_voters_dists = np.linalg.norm(
            boundary_points_voters_deltas, axis=2
        )  # shape (360, V)
        avg_boundary_points_voters_dists = boundary_points_voters_dists.mean(axis=1)
        arg_max = np.argmax(
            avg_boundary_points_voters_dists
        )  # index of the maximum average distance
        curr_max = boundary_points[arg_max]

        # performing lookahead
        n_random_points_to_choose = 3
        lookahead_directions = 60
        lookahead_halving_iterations = 12  # should probably be the value used in first generate_winset_boundary call
        random_indices = np.random.choice(
            boundary_points.shape[0], size=n_random_points_to_choose, replace=False
        )
        sampled_points = boundary_points[random_indices]
        points_to_examine = np.vstack([sampled_points, curr_max])
        new_max_point = curr_max
        new_max_val = avg_boundary_points_voters_dists[arg_max]
        for point in points_to_examine:
            new_boundary_points = self.generate_winset_boundary(
                Policy(point), lookahead_directions, lookahead_halving_iterations
            )
            new_boundary_points_voters_deltas = (
                new_boundary_points[:, None, :] - self.voter_arr[None, :, :]
            )  # shape (360, V, 2)
            new_boundary_points_voters_dists = np.linalg.norm(
                new_boundary_points_voters_deltas, axis=2
            )  # shape (360, V)
            new_avg_boundary_points_voters_dists = (
                new_boundary_points_voters_dists.mean(axis=1)
            )
            new_arg_max = np.argmax(
                new_avg_boundary_points_voters_dists
            )  # index of the maximum average distance
            if new_avg_boundary_points_voters_dists[new_arg_max] > new_max_val:
                new_max_point = point
                new_max_val = new_avg_boundary_points_voters_dists[new_arg_max]

        return Policy(
            new_max_point
        )  # return the policy with the maximum average distance
    
    def obtain_mckelvey_schofield_path(
        self,
        original_policy: Policy,
        goal_policy: Policy,
        max_steps: int = 50,
        step_selection_function="mckelvey_schofield_greedy_with_lookahead",
        print_verbose=False,
    ) -> list[Policy]:
        """
        Obtains the McKelvey-Schofield path from the original_policy to the goal_policy.

        Args:
            original_policy (Policy): the original policy.
            goal_policy (Policy): the goal policy.
            max_steps (int): the maximum number of steps for which to run the algorithm.
            step_selection_function (str): the function to use at each step to select the next policy.
            print_verbose (bool): whether to provide the user with updates as the path is obtained.

        Returns:
            list[Policy]: a list of policies starting with the original_policy and ending with the
                goal_policy (if possible) such that policy i+1 will defeat policy i in an election.
        """
        
        policy_path = [original_policy]  # Initialize the path with the original policy

        for policy_num in range(1,max_steps+1):
            if print_verbose:
                print(f"Obtaining policy {policy_num} in path.")

            current_policy = policy_path[-1]
            if self.compare_policies(current_policy, goal_policy) == 1:
                new_policy = goal_policy
            else:
                try:
                    if (
                        step_selection_function
                        == "mckelvey_schofield_greedy_with_adjustment_avg_dist"
                    ):
                        new_policy = (
                            self.mckelvey_schofield_greedy_with_adjustment_avg_dist(
                                current_policy, policy_path
                            )
                        )
                    elif step_selection_function == "mckelvey_schofield_greedy_avg_dist":
                        new_policy = self.mckelvey_schofield_greedy_avg_dist(current_policy)
                    elif (
                        step_selection_function
                        == "mckelvey_schofield_greedy_with_lookahead"
                    ):
                        new_policy = self.mckelvey_schofield_greedy_with_lookahead(
                            current_policy
                        )
                    else:
                        raise ValueError(
                            f"Unknown step selection function: {step_selection_function}"
                        )
                except ValueError as e:
                    print(f"ValueError({e}) encountered with policy {current_policy.values} at step {policy_num}.")
                    print("Returning current path.")
                    break
                
            policy_path.append(new_policy)
            if print_verbose:
                print(f"Policy {policy_num} obtained.")
            if new_policy == goal_policy:
                break

        if print_verbose:
            if policy_num >= max_steps and not np.allclose(
                policy_path[-1].values, goal_policy.values
            ):
                print(
                    f"Could not reach the goal policy after {max_steps} steps."
                )
            else:
                print("Reached the goal policy!")

        return policy_path  # Return the path of policies from original to goal policy

    def plot_path_average_distances(
        self,
        path: list[Policy],
        max_steps: int,
        output_folder="output",
        filename="output",
    ):
        """
        Given a McKelvey-Schofield path, plots the average distance of each policy from the voters.
        The plot is output to the specified location.

        Args:
            path (list[Policy]): a McKelvey-Schofield path.
            max_steps (int): the max_steps parameter used in calculating that path.
            output_folder (str): the output folder.
            filename (str): the output file name to use (should end in .png).
        """
        # plot settings
        fig = plt.figure(figsize=(12, 8))

        # obtaining average values
        distances = []
        for p in path:
            curr_avg_dist = sum(
                [
                    math.dist(p.values, voter.ideal_policy.values)
                    for voter in self.voters
                ]
            ) / len(self.voters)
            distances.append(curr_avg_dist)

        # creating plot
        plt.plot(
            [i for i in range(len(distances))],
            distances,
            color="black",
            linestyle="-",
            marker="o",
        )

        title = "Average Distance Values for Policies Along the Path"
        plt.title(title)
        plt.xlim(right=max_steps)  # adjust the right leaving left unchanged
        plt.xlabel("Policy")
        plt.ylabel("Average Distance from Voter Preferences")
        plt.savefig(f"{output_folder}/{filename}", bbox_inches="tight")
        plt.close(fig)

    def plot_winset_boundary(
        self,
        current_policy,
        n_directions=360,
        n_halving_iterations=12,
        angle_offset=0,
        output_folder="output",
        filename=None,
    ):
        """
        Plots (approximately) the set of policies capable of beating the current_policy.
        The plot is visualized.

        Args:
            current_policy (Policy): the current policy.
            n_directions (int): the number of directions in which to search for boundary points.
            n_halving_iterations (int): the number of iterations of binary search to run.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.3, 0.55, 0.55])  # Shrink plot inside the figure

        # some settings
        current_policy_name = (
            "Current Policy" if current_policy.name is None else current_policy.name
        )
        current_color = "orange"
        boundary_color = "green"

        # plotting all voters
        voters_plot = ax.scatter(
            self.voter_arr[:, 0],
            self.voter_arr[:, 1],
            c="black",
            marker="o",
        )
        voters_plot.set_label("Voters")

        # plotting policies
        winset_boundary = self.generate_winset_boundary(
            current_policy, n_directions, n_halving_iterations, angle_offset
        )
        winset_boundary_loop = np.vstack([winset_boundary, winset_boundary[0]])
        ax.plot(
            winset_boundary_loop[:, 0],
            winset_boundary_loop[:, 1],
            color=boundary_color,
            label="Winset Boundary",
            marker="o",
            markersize=4,
            linestyle="--",
            linewidth=3,
        )
        poly = Polygon(winset_boundary_loop, closed=True, facecolor=boundary_color, alpha=0.25)
        ax.add_patch(poly)

        current_policy_plot = ax.scatter(
            [current_policy.values[0]],
            [current_policy.values[1]],
            color=current_color,
            marker="o",
            edgecolors="black",
            s=100,
        )
        current_policy_plot.set_label(current_policy_name)

        # title, labels, legend
        desired_order = [
            current_policy_name,
            "Winset Boundary",
            "Voters",
        ]
        handles, labels = plt.gca().get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        ordered_handles = [
            label_to_handle[label]
            for label in desired_order
            if label in label_to_handle
        ]
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.0,
            handles=ordered_handles,
            labels=desired_order,
        )
        plt.title(
            f"Approximate Set of Policies that Beat {current_policy_name}"
        )
        plt.xlabel(f"Position on {self.issue_1}")
        plt.ylabel(f"Position on {self.issue_2}")
        plt.grid(True)

        # saving or showing the plot
        if output_folder is not None and filename is not None:
            plt.savefig(f"{output_folder}/{filename}", bbox_inches="tight")
        else:
            plt.show()
        plt.close()
