import matplotlib.pyplot as plt
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
        original_utilities = self.get_policy_utilities(original_policy)
        new_utilities = self.get_policy_utilities(new_policy)
        votes = np.full(len(self.voters), -1)
        votes[original_utilities > new_utilities] = 0
        votes[original_utilities < new_utilities] = 1
        return votes

    def get_policy_utilities(self, policy: Policy) -> np.array:
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
        angle_offset: float = 0.0,
    ) -> np.ndarray:
        # TODO: add a function to plot the winset boundary
        # setting up recurring values
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
        return filtered_boundary_points

    def mckelvey_schofield_greedy_avg_dist(self, current_policy) -> Policy:
        """
        Select the next policy using a greedy algorithm - choose the policy with the highest average distance
        from all voters which beats the current policy.
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
        self, current_policy, policy_path
    ) -> Policy:
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

    def mckelvey_schofield_greedy_with_lookahead(self, current_policy) -> Policy:
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
        
        policy_path = [original_policy]  # Initialize the path with the original policy

        for policy_num in range(1,max_steps+1):
            if print_verbose:
                print(f"Obtaining policy {policy_num} in path.")

            current_policy = policy_path[-1]
            if self.compare_policies(current_policy, goal_policy) == 1:
                new_policy = goal_policy
            else:
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
        self, current_policy, n_directions=360, n_halving_iterations=12, angle_offset=0
    ):
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
            markersize=2,
            linestyle="--",
        )

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
            f"Approximate Boundary of the Set of Policies that Beats {current_policy_name}"
        )
        plt.xlabel(f"Position on {self.issue_1}")
        plt.ylabel(f"Position on {self.issue_2}")
        plt.grid(True)
        plt.show()
        plt.close()
