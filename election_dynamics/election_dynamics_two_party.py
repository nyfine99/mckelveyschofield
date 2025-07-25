import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from ordered_set import OrderedSet


from election_dynamics.election_dynamics import ElectionDynamics
from voters.voter import Voter
from policies.policy import Policy


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
        self.tiebreak_func = (
            tiebreak_func  # used when a voter is ambivalent to distribute their vote
        )
        self.issue_1 = issue_1  # Issue 1 name
        self.issue_2 = issue_2  # Issue 2 name

    def compare_policies(self, original_policy: Policy, new_policy: Policy):
        return self.evaluation_function(
            self.tabulate_votes(original_policy, new_policy)
        )

    def tabulate_votes(self, original_policy: Policy, new_policy: Policy):
        counts = [
            0,
            0,
        ]  # index 0 represents the original_policy count, index 1 the new policy
        votes = self.obtain_individual_votes(original_policy, new_policy)
        counts = [np.sum(votes == 0), np.sum(votes == 1)]
        return counts

    def obtain_individual_votes(
        self, original_policy: Policy, new_policy: Policy
    ) -> np.array:
        # it could be more performant to take this out of numpy
        # but numpy should certainly be used for the overriding functions
        original_utilities = np.array(
            [voter.get_utility(original_policy) for voter in self.voters]
        )
        new_utilities = np.array(
            [voter.get_utility(new_policy) for voter in self.voters]
        )
        votes = np.full(len(self.voters), -1)
        votes[original_utilities > new_utilities] = 0
        votes[original_utilities < new_utilities] = 1
        if self.tiebreak_func is not None:
            ties = original_utilities == new_utilities
            votes[ties] = [self.tiebreak_func() for _ in range(np.sum(ties))]
        return votes

    def plot_election_2d(
        self, original_policy: Policy, new_policy: Policy, verbose: bool = True
    ):
        # initialize the figure and axes
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.3, 0.55, 0.55])  # Shrink plot inside the figure

        # some settings
        original_policy_name = (
            "Original Policy" if original_policy.name is None else original_policy.name
        )
        new_policy_name = (
            "New Policy" if original_policy.name is None else new_policy.name
        )
        undecided_name = "Undecided"
        original_color = "blue"
        new_color = "red"
        undecided_color = "yellow"

        # plotting all voters
        votes = self.obtain_individual_votes(original_policy, new_policy)
        voters_by_vote = {"original": [], "new": [], "undecided": []}
        policy_colors = {
            "original": original_color,
            "new": new_color,
            "undecided": undecided_color,
        }
        policy_names = {
            "original": original_policy_name,
            "new": new_policy_name,
            "undecided": undecided_name,
        }
        for i, vote in enumerate(votes):
            if vote == 0:
                voters_by_vote["original"].append(self.voters[i])
            elif vote == 1:
                voters_by_vote["new"].append(self.voters[i])
            else:
                voters_by_vote["undecided"].append(self.voters[i])

        for k in voters_by_vote.keys():
            if voters_by_vote[k]:
                arr = np.array(
                    [voter.ideal_policy.values for voter in voters_by_vote[k]]
                )
                voters_plot = ax.scatter(
                    arr[:, 0],
                    arr[:, 1],
                    c=policy_colors[k],
                    marker="o",
                )
                curr_name = policy_names[k]
                voters_plot.set_label(f"{curr_name} Voters")

        # plotting policies and differentiating winner from loser
        winner = self.compare_policies(original_policy, new_policy)
        original_marker = "*" if winner == 0 else "X"
        original_size = 250 if winner == 0 else 150
        new_marker = "*" if winner == 1 else "X"
        new_size = 250 if winner == 1 else 150
        original_policy_plot = ax.scatter(
            [original_policy.values[0]],
            [original_policy.values[1]],
            color=original_color,
            marker=original_marker,
            edgecolors="black",
            s=original_size,
        )
        original_policy_plot.set_label(original_policy_name)
        new_policy_plot = ax.scatter(
            [new_policy.values[0]],
            [new_policy.values[1]],
            color=new_color,
            marker=new_marker,
            edgecolors="black",
            s=new_size,
        )
        new_policy_plot.set_label(new_policy_name)

        # title, labels, legend
        desired_order = [
            original_policy_name,
            new_policy_name,
            f"{original_policy_name} Voters",
            f"{new_policy_name} Voters",
            f"{undecided_name} Voters",
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
        plt.title(f"{original_policy_name} vs {new_policy_name}: Election Results")
        plt.xlabel(f"Position on {self.issue_1}")
        plt.ylabel(f"Position on {self.issue_2}")

        # details
        if verbose:
            winner_text = original_policy_name if winner == 0 else new_policy_name
            vote_totals = self.tabulate_votes(original_policy, new_policy)
            original_policy_votes = vote_totals[0]
            new_policy_votes = vote_totals[1]
            fig.text(
                0,
                0.05,
                f"""
                        {original_policy_name} (blue) received {original_policy_votes} votes.
                        {new_policy_name} (red) received {new_policy_votes} votes.
                        So, {winner_text} wins!""",
                fontsize=9,
                color="black",
            )

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
        original_policy_name = (
            "Original Policy" if original_policy.name is None else original_policy.name
        )
        goal_policy_name = (
            "New Policy" if original_policy.name is None else goal_policy.name
        )
        original_color = "blue"
        goal_color = "red"

        # plotting all voters
        voters_plot = ax.scatter(
            [voter.ideal_policy.values[0] for voter in self.voters],
            [voter.ideal_policy.values[1] for voter in self.voters],
            c="black",
            marker="o",
        )
        voters_plot.set_label("Voters")

        # plotting path
        ax.quiver(
            [p.values[0] for p in path[:-1]],
            [p.values[1] for p in path[:-1]],
            [path[i + 1].values[0] - path[i].values[0] for i in range(len(path) - 1)],
            [path[i + 1].values[1] - path[i].values[1] for i in range(len(path) - 1)],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="green",
            alpha=0.7,
        )

        # plotting policies
        intermediate_plot = ax.scatter(
            [p.values[0] for p in path[1:-1]],
            [p.values[1] for p in path[1:-1]],
            color="green",
            marker="o",
            s=200,
            alpha=0.7,
        )
        if not np.allclose(path[-1].values, goal_policy.values):
            # if the last policy in the path is not the goal policy, plot it as well
            ax.scatter(
                [path[-1].values[0]],
                [path[-1].values[1]],
                color="green",
                marker="o",
                s=200,
                alpha=0.7,
            )
        intermediate_plot.set_label("Intermediate Policies")

        original_policy_plot = ax.scatter(
            [original_policy.values[0]],
            [original_policy.values[1]],
            color=original_color,
            edgecolors="black",
            marker="X",
            s=200,
        )
        original_policy_plot.set_label(original_policy_name)

        goal_policy_plot = ax.scatter(
            [goal_policy.values[0]],
            [goal_policy.values[1]],
            color=goal_color,
            edgecolors="black",
            marker="*",
            s=200,
        )
        goal_policy_plot.set_label(goal_policy_name)

        # title, labels, legend
        desired_order = [
            original_policy_name,
            goal_policy_name,
            "Intermediate Policies",
            "Voters",
        ]
        handles, labels = plt.gca().get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        ordered_handles = [label_to_handle[label] for label in desired_order]
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.0,
            handles=ordered_handles,
            labels=desired_order,
        )
        title = f"Path from {original_policy_name} to {goal_policy_name}"
        if not np.allclose(path[-1].values, goal_policy.values):
            # the goal policy was not reached, and the title should reflect this
            title = f"Attempted Path from {original_policy_name} to {goal_policy_name}"
        plt.title(title)
        plt.xlabel(f"Position on {self.issue_1}")
        plt.ylabel(f"Position on {self.issue_2}")

        plt.grid(True)
        if save_file is not None:
            plt.savefig(save_file, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)

    def animate_mckelvey_schofield_path(
        self,
        original_policy,
        goal_policy,
        policy_path,
        max_frames=1000,
        output_folder="output",
        filename="output",
        plot_verbose=True,
        fps=0.5,
    ):
        fig = plt.figure()

        original_policy_color = "blue"
        original_policy_name = (
            original_policy.name
            if original_policy.name is not None
            else "Original Policy"
        )
        goal_policy_color = "red"
        goal_policy_name = (
            goal_policy.name if goal_policy.name is not None else "Goal Policy"
        )

        def make_frame(f_num):
            plt.clf()  # Clear the current axes/figure
            fig.add_axes([0.1, 0.3, 0.55, 0.55])

            current_policy = policy_path[f_num]
            new_policy = policy_path[f_num + 1]  # frame_gen avoids cases where f_num + 1 is out of bounds
            
            # initial plot settings
            current_color = "green" if f_num % 2 == 0 else "orange"
            new_color = "orange" if f_num % 2 == 0 else "green"
            undecided_color = "yellow"
            current_policy_name = f"Policy {f_num}"
            new_policy_name = f"Policy {f_num+1}"
            undecided_name = "Undecided"

            original_policy_opacity = 0.5
            goal_policy_opacity = 0.5

            if f_num == 0:  # this assumes that policy_path starts with the original policy
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
            policy_colors = {
                "current": current_color,
                "new": new_color,
                "undecided": undecided_color,
            }
            policy_names = {
                "current": current_policy_name,
                "new": new_policy_name,
                "undecided": undecided_name,
            }
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
                    marker="o",
                )
                curr_name = policy_names[k]
                voters_plot.set_label(f"{curr_name} Voters")

            # plotting policies
            original_policy_plot = plt.scatter(
                [original_policy.values[0]],
                [original_policy.values[1]],
                color=original_policy_color,
                edgecolors="black",
                marker="X",
                s=200,
                alpha=original_policy_opacity,
            )
            original_policy_plot.set_label(original_policy_name)

            goal_policy_plot = plt.scatter(
                [goal_policy.values[0]],
                [goal_policy.values[1]],
                color=goal_policy_color,
                edgecolors="black",
                marker="*",
                s=200,
                alpha=goal_policy_opacity,
            )
            goal_policy_plot.set_label(goal_policy_name)

            if f_num != 0:
                current_policy_plot = plt.scatter(
                    [current_policy.values[0]],
                    [current_policy.values[1]],
                    color=current_color,
                    edgecolors="black",
                    s=200,
                )
                current_policy_plot.set_label(current_policy_name)

            if new_policy != goal_policy:
                new_policy_plot = plt.scatter(
                    [new_policy.values[0]],
                    [new_policy.values[1]],
                    color=new_color,
                    edgecolors="black",
                    s=200,
                )
                new_policy_plot.set_label(new_policy_name)

            # title, labels, legend
            desired_order = [
                original_policy_name,
                goal_policy_name,
                current_policy_name,
                new_policy_name,
                f"{current_policy_name} Voters",
                f"{new_policy_name} Voters",
                f"{undecided_name} Voters",
            ]
            desired_order = OrderedSet(
                desired_order
            )  # removing duplicates on first and last frames
            handles, labels = plt.gca().get_legend_handles_labels()
            label_to_handle = dict(zip(labels, handles))
            ordered_handles = [label_to_handle[label] for label in desired_order]
            plt.title(
                f"Path from {original_policy_name} to {goal_policy_name},\n Step {f_num+1}"
            )
            plt.xlabel(f"Position on {self.issue_1}")
            plt.ylabel(f"Position on {self.issue_2}")
            plt.legend(
                loc="upper left",
                bbox_to_anchor=(1.05, 1),
                borderaxespad=0.0,
                handles=ordered_handles,
                labels=desired_order,
            )

            # details
            if plot_verbose:
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
                    fontsize=9,
                    color="black",
                )

        def frame_gen():
            f_num = 0
            while True:
                if (
                    np.allclose(policy_path[f_num].values, goal_policy.values)
                    or f_num >= max_frames
                    or f_num >= len(policy_path) - 1
                ):
                    break
                yield f_num
                f_num += 1

        def init():
            return None

        ani = animation.FuncAnimation(
            fig, make_frame, frames=frame_gen(), init_func=init, save_count=max_frames+5  # leaving some extra buffer
        )
        # Save to mp4
        ani.save(f"{output_folder}/{filename}.mp4", writer="ffmpeg", fps=fps)
        plt.close(fig)
