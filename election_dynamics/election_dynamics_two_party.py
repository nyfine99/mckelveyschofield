"""
Two-party Electoral Dynamics Implementation

This module implements the core two-party electoral system, serving as the foundation
for electoral simulations where voters choose between exactly two policy alternatives.
The ElectionDynamicsTwoParty class extends the abstract base class to provide
concrete implementations of vote tabulation, policy comparison, and electoral analysis.

Key Features:
- Binary choice electoral system (two policies compete)
- Individual voter preference calculation, aggregation, and visualization
- McKelvey-Schofield path visualization

The implementation serves as the base class for more specialized two-party systems
(e.g., SimpleVoter-specific implementations) while providing the core electoral logic.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from ordered_set import OrderedSet


from election_dynamics.election_dynamics import ElectionDynamics
from voters.voter import Voter
from policies.policy import Policy


class ElectionDynamicsTwoParty(ElectionDynamics):
    """
    Concrete implementation of two-party electoral dynamics.
    
    This class implements the core electoral logic for binary choice elections,
    where voters must choose between exactly two policy alternatives. It provides
    the foundation for understanding how individual voter preferences aggregate
    into electoral outcomes through the lens of spatial voting theory.
    
    The class handles:
    - Individual vote calculation based on utility comparisons
    - Aggregate vote counting and electoral outcome determination
    - Tie-breaking for indifferent voters
    - Policy space visualization and analysis
    - Efficient utility computation and caching
    
    This implementation is designed to be extended by more specialized classes
    (e.g., ElectionDynamicsTwoPartySimpleVoters) while providing the core
    electoral mechanics that all two-party systems share.
    
    Attributes:
        voters (list[Voter]): collection of voter objects representing the electorate.
                              each voter must implement the voter interface with
                              get_utility() method.
        evaluation_function (callable): function that determines electoral outcomes
                                       from vote counts. Typically returns the
                                       winning policy based on vote totals.
        tiebreak_func (callable, optional): function to handle voter indifference.
                                            Called when a voter's utilities for
                                            both policies are equal. Defaults to None.
        issue_1 (str): Label for the first policy dimension (e.g., "Economic Policy").
        issue_2 (str): Label for the second policy dimension (e.g., "Social Policy").
    """

    def __init__(
        self,
        voters: list[Voter],
        evaluation_function: callable,
        tiebreak_func: callable = None,
        issue_1: str = "Issue 1",
        issue_2: str = "Issue 2",
    ):
        """
        Initialize the two-party electoral dynamics system.
        
        Creates a new electoral system configured for binary choice elections.
        The system is designed to handle any voter type that implements the
        Voter interface, making it flexible for different preference models.
        
        Args:
            voters (list[Voter]): List of voter objects. Each voter must have
                                  a get_utility(policy) method that returns a
                                  numeric utility value for any given policy.
            evaluation_function (callable): Function that determines electoral
                                           outcomes from vote counts. Should
                                           accept a list of vote counts and
                                           return the winning policy or result.
            tiebreak_func (callable, optional): Function to resolve voter
                                               indifference. Called when a
                                               voter's utilities for both
                                               policies are equal. If None,
                                               indifferent voters abstain.
            issue_1 (str, optional): Human-readable label for the first
                                     policy dimension. Used for plot labels
                                     and documentation. Defaults to "Issue 1".
            issue_2 (str, optional): Human-readable label for the second
                                     policy dimension. Used for plot labels
                                     and documentation. Defaults to "Issue 2".
        
        Raises:
            ValueError: If voters list is empty or evaluation_function is None.
            TypeError: If voters contains objects that don't implement Voter interface.
        """
        super().__init__(voters, evaluation_function)
        
        # validate inputs
        if not voters:
            raise ValueError("Voters list cannot be empty")
        if evaluation_function is None:
            raise ValueError("Evaluation function cannot be None")
        if not all(hasattr(voter, 'get_utility') for voter in voters):
            raise TypeError("All voters must implement get_utility method")
        
        self.tiebreak_func = tiebreak_func  # used when a voter is ambivalent to distribute their vote
        self.issue_1 = issue_1  # Issue 1 name
        self.issue_2 = issue_2  # Issue 2 name

    def compare_policies(self, original_policy: Policy, new_policy: Policy):
        """
        Compare two policies to determine which would win in an election.
        
        This method implements the core electoral logic by tabulating votes
        between two policies and then applying the evaluation function to
        determine the winner. It's the primary method for understanding
        electoral competition between policy alternatives.
        
        Args:
            original_policy (Policy): The incumbent or status quo policy.
                                      Represents the current policy position
                                      that the new policy is challenging.
            new_policy (Policy): The challenger or proposed new policy.
                                 Represents the alternative policy that
                                 voters are considering.
        
        Returns:
            The result depends on the evaluation_function:
            - For simple majority: usually returns the winning policy
            - For other systems: may return vote counts, margins, or
              other electoral outcome measures
        """
        vote_counts = self.tabulate_votes(original_policy, new_policy)
        return self.evaluation_function(vote_counts)

    def tabulate_votes(self, original_policy: Policy, new_policy: Policy):
        """
        Count votes for each policy in a binary choice election.
        
        This method aggregates individual voter preferences into vote counts
        for each policy. It handles the core electoral mechanics including
        utility comparison, vote assignment, and tie-breaking for indifferent
        voters. The method is designed to be efficient for large electorates.
        
        Args:
            original_policy (Policy): The incumbent or status quo policy.
            new_policy (Policy): The challenger or proposed policy.
        
        Returns:
            list[int]: Vote counts where:
                       - index 0: votes for original_policy
                       - index 1: votes for new_policy
                       - Total votes may be less than electorate size
                         if some voters abstain due to indifference
        """
        votes = self.obtain_individual_votes(original_policy, new_policy)
        counts = [np.sum(votes == 0), np.sum(votes == 1)]
        return counts

    def obtain_individual_votes(
        self, original_policy: Policy, new_policy: Policy
    ) -> np.array:
        """
        Calculate individual voter preferences between two policies.
        
        This method computes how each individual voter would vote in a
        binary choice between two policies. It handles utility comparison,
        vote assignment, and tie-breaking for indifferent voters. The
        method is designed to be efficient and vectorizable for large
        electorates.
        
        Args:
            original_policy (Policy): The incumbent or status quo policy.
            new_policy (Policy): The challenger or proposed policy.
        
        Returns:
            np.array: Array of individual votes where:
                      - 0: voter prefers original_policy
                      - 1: voter prefers new_policy
                      - -1: voter is indifferent (abstains)
                      - Length equals the number of voters
        
        Note:
            The method could be optimized by moving some operations out of
            numpy loops, but numpy is used for the overriding functions
            in derived classes for performance reasons.
        """
        # calculate utilities for each voter for each policy
        original_utilities = np.array(
            [voter.get_utility(original_policy) for voter in self.voters]
        )
        new_utilities = np.array(
            [voter.get_utility(new_policy) for voter in self.voters]
        )
        
        # initialize votes array with -1 (indifferent/abstain)
        votes = np.full(len(self.voters), -1)
        
        # assign votes based on utility comparison
        votes[original_utilities > new_utilities] = 0  # Prefer original
        votes[original_utilities < new_utilities] = 1  # Prefer new
        
        # handle ties using tiebreak function if provided
        if self.tiebreak_func is not None:
            ties = original_utilities == new_utilities
            votes[ties] = [self.tiebreak_func() for _ in range(np.sum(ties))]
        
        return votes

    def plot_election_2d(
        self, original_policy: Policy, new_policy: Policy, verbose: bool = True
    ):
        """
        Create a 2D visualization of electoral competition between two policies.
        
        This method generates a comprehensive visualization showing how voters
        are distributed in policy space and how they voted in the binary choice
        between two policies. The plot includes voter positions, policy positions,
        and color-coding to show voting patterns. This visualization is crucial
        for understanding spatial voting theory and electoral dynamics.
        
        Args:
            original_policy (Policy): The incumbent or status quo policy.
                                      Will be plotted in blue.
            new_policy (Policy): The challenger or proposed policy.
                                 Will be plotted in red.
            verbose (bool, optional): Whether to print detailed information
                                     about the election results in the plot.
                                     Defaults to True.
        
        Returns:
            None: Creates and displays a matplotlib figure.
        
        Note:
            The plot is designed to show the relationship between policy
            positions and voter preferences, making it useful for understanding
            electoral competition and spatial voting theory.
        """
        # initialize the figure and axes with appropriate sizing
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.3, 0.55, 0.55])  # Shrink plot inside the figure

        # configure policy names and colors for visualization
        original_policy_name = (
            "Original Policy" if original_policy.name is None else original_policy.name
        )
        new_policy_name = (
            "New Policy" if new_policy.name is None else new_policy.name
        )
        undecided_name = "Undecided"
        
        # define color scheme for different voting groups
        original_color = "blue"
        new_color = "red"
        undecided_color = "yellow"

        # get individual votes and organize voters by their choice
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
        
        # categorize voters by their voting choice
        for i, vote in enumerate(votes):
            if vote == 0:
                voters_by_vote["original"].append(self.voters[i])
            elif vote == 1:
                voters_by_vote["new"].append(self.voters[i])
            else:
                voters_by_vote["undecided"].append(self.voters[i])

        # plot voters grouped by their voting preference
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

        # plot policies and differentiate winner from loser
        winner = self.compare_policies(original_policy, new_policy)
        # use star (*) for winner, X for loser, with larger size for winner
        original_marker = "*" if winner == 0 else "X"
        original_size = 250 if winner == 0 else 150
        new_marker = "*" if winner == 1 else "X"
        new_size = 250 if winner == 1 else 150
        
        # plot original policy with appropriate marker and size
        original_policy_plot = ax.scatter(
            [original_policy.values[0]],
            [original_policy.values[1]],
            color=original_color,
            marker=original_marker,
            edgecolors="black",
            s=original_size,
        )
        original_policy_plot.set_label(original_policy_name)
        
        # plot new policy with appropriate marker and size
        new_policy_plot = ax.scatter(
            [new_policy.values[0]],
            [new_policy.values[1]],
            color=new_color,
            marker=new_marker,
            edgecolors="black",
            s=new_size,
        )
        new_policy_plot.set_label(new_policy_name)

        # configure legend with desired order and positioning
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
        
        # set title and axis labels
        plt.title(f"{original_policy_name} vs {new_policy_name}: Election Results")
        plt.xlabel(f"Position on {self.issue_1}")
        plt.ylabel(f"Position on {self.issue_2}")

        # display detailed election results if verbose mode is enabled
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
        """
        Create a static visualization of a McKelvey-Schofield path through policy space.
        
        The McKelvey-Schofield Chaos Theorem demonstrates that agenda setters can
        manipulate outcomes by controlling the sequence of binary choices, even
        when the final policy would lose in a direct comparison with the original.

        This method generates a comprehensive plot showing how an agenda setter
        can manipulate electoral outcomes by strategically sequencing binary choices.
        The visualization displays the path from an original policy to a goal policy,
        showing intermediate policy positions and the strategic route taken.
        
        Args:
            original_policy (Policy): The starting policy position (incumbent).
                                      Plotted in blue with an X marker.
            goal_policy (Policy): The target policy position (desired outcome).
                                  Plotted in red with a star marker.
            path (list[Policy]): Sequence of policies representing the strategic
                                 path from original to goal. Each policy in the
                                 sequence must be able to defeat the previous one.
            save_file (str, optional): File path to save the plot. If None,
                                       the plot is displayed interactively.
                                       Defaults to None.
        
        Returns:
            None: Creates and displays/saves a matplotlib figure.
        
        Note:
            The path visualization is crucial for understanding how agenda setting
            can manipulate electoral outcomes through strategic policy sequencing.
            Green arrows show the direction of policy movement, and intermediate
            policies are shown as green circles.
        """
        # plotting the path
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.3, 0.55, 0.55])  # Shrink plot inside the figure

        # some settings
        original_policy_name = (
            "Original Policy" if original_policy.name is None else original_policy.name
        )
        goal_policy_name = (
            "New Policy" if goal_policy.name is None else goal_policy.name
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
        original_policy: Policy,
        goal_policy: Policy,
        policy_path: list[Policy],
        max_frames: int = 1000,
        output_folder: str = "output",
        filename: str = "output",
        plot_verbose: bool = True,
        fps: float = 0.5,
    ):
        """
        Create an animated visualization of a McKelvey-Schofield path through policy space.
        
        This method generates an MP4 animation showing how an agenda setter
        manipulates electoral outcomes step-by-step through strategic policy
        sequencing. Each frame shows a binary choice between consecutive policies
        in the path, with voters color-coded by their preference and vote counts
        displayed below the plot.
        
        The animation demonstrates the McKelvey-Schofield theorem by showing
        how each policy in the sequence can defeat the previous one, even when
        the final goal policy would lose in a direct comparison with the original.
        
        Args:
            original_policy (Policy): The starting policy position (incumbent).
                                      Plotted in blue with an X marker.
            goal_policy (Policy): The target policy position (desired outcome).
                                  Plotted in red with a star marker.
            policy_path (list[Policy]): Sequence of policies representing the
                                        strategic path from original to goal.
                                        Each policy must defeat the previous one.
            max_frames (int, optional): Maximum number of animation frames to
                                        generate. Prevents infinite loops.
                                        Defaults to 1000.
            output_folder (str, optional): Directory to save the output MP4 file.
                                           Defaults to "output".
            filename (str, optional): Base filename for the output MP4 file
                                      (without extension). Defaults to "output".
            plot_verbose (bool, optional): Whether to display vote counts and
                                           explanatory text below the plot.
                                           Defaults to True.
            fps (float, optional): Frames per second for the animation.
                                   Lower values create slower animations
                                   for easier viewing. Defaults to 0.5.
        
        Returns:
            None: Creates and saves an MP4 animation file.
        
        Note:
            The animation requires FFmpeg to be installed for MP4 output.
            Each frame shows the electoral competition between consecutive
            policies, with voters color-coded by their voting preference.
            The animation automatically stops when the goal is reached or
            the maximum frames are exceeded.
        """
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
