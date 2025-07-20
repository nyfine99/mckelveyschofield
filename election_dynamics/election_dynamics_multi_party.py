import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from ordered_set import OrderedSet


from election_dynamics.election_dynamics import ElectionDynamics
from voters.voter import Voter
from policies.policy import Policy


class ElectionDynamicsMultiParty(ElectionDynamics):
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

    def compare_policies(self, policies: list[Policy]) -> int:
        """
        Compares policies using the evaluation function. Takes a list of Policy objects, tabulates votes (returns ndarray), and passes to evaluation function.
        Params:
            policies (list[Policy]): List of Policy objects.
        Returns:
            int: Index of winning policy
        """
        preferences = self.tabulate_votes(policies)
        return self.evaluation_function(preferences)

    def tabulate_votes(self, policies: list[Policy]) -> np.ndarray:
        """
        Returns a 2D np.ndarray where each row represents an individual voter's preferences (policy indices, ranked best to worst).
        For example, array([[1,2,3], [2,1,3]]) means voter 0 prefers 1>2>3, voter 1 prefers 2>1>3.
        Params:
            policies (list[Policy]): List of Policy objects.
        Returns:
            np.ndarray: shape (num_voters, num_policies)
        """
        return np.array([voter.rank_policies_by_utility_index_preference(policies) for voter in self.voters])

    def plot_voters_first_choices(
        self, policies: list[Policy], verbose: bool = True
    ):
        # initialize the figure and axes
        if len(policies) < 1:
            print("Not enough policies to hold an election!")
            return
        if len(policies) > 10:
            print("Currently not enough colors to adequately plot!")
            # TODO: change this
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.3, 0.55, 0.55])  # Shrink plot inside the figure

        # color and name vis settings
        mcolors_dict = mcolors.TABLEAU_COLORS

        # ensuring blue and red are the first two colors used
        del mcolors_dict['tab:blue']
        del mcolors_dict['tab:red']
        all_colors = ['blue', 'red'] + list(mcolors_dict.values())
        policy_names = [policy.name for policy in policies]
        policy_colors = all_colors[0:len(policies)]

        # plotting all voters and policies
        ballots = self.tabulate_votes(policies)
        voters_by_vote = {}
        for i in range(len(policies)):
            voters_by_vote[i] = []

        for i, ballot in enumerate(ballots):
            voters_by_vote[ballot[0]].append(self.voters[i])

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
                
                policy_plot = ax.scatter(
                    [policies[k].values[0]],
                    [policies[k].values[1]],
                    color=policy_colors[k],
                    marker="o",
                    edgecolors="black",
                    s=200,
                )
                policy_plot.set_label(curr_name)

        # title, labels, legend
        desired_order = policy_names + [f"{policy_name} Voters" for policy_name in policy_names]
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
        title = f"Voters' First Choices:\n" + " vs. ".join(policy_names[:5])
        if len(policy_names) > 5:
            title += " vs. ..."
        plt.title(title)
        plt.xlabel(f"Position on {self.issue_1}")
        plt.ylabel(f"Position on {self.issue_2}")

        if verbose:
            # sorting voters_by_vote by first round vote count
            description_string = "In first-choice selections"
            sorted_items = sorted(voters_by_vote.items(), key=lambda item: len(item[1]), reverse=True)
            top_policy_idx = sorted_items[0][0]
            top_policy_name = policy_names[top_policy_idx]
            top_policy_votes = len(sorted_items[0][1])
            description_string = f"\n{top_policy_name} leads, as the first choice of {top_policy_votes} voters."

            if len(sorted_items) > 1:
                second_policy_idx = sorted_items[1][0]
                second_policy_name = policy_names[second_policy_idx]
                second_policy_votes = len(sorted_items[1][1])
                if len(sorted_items) > 2:
                    description_string += f"\n{second_policy_name} is in second place, as the first choice of {second_policy_votes} voters."
                else:
                    # this is the last place policy, phrase accordingly
                    description_string += f"\n{second_policy_name} trails, as the first choice of {second_policy_votes} voters."

            if len(sorted_items) > 2:
                last_policy_idx = sorted_items[-1][0]
                last_policy_name = policy_names[last_policy_idx]
                last_policy_votes = len(sorted_items[-1][1])
                description_string += "\n..."
                description_string += f"\n{last_policy_name} is in last place, as the first choice of {last_policy_votes} voters."

            fig.text(
                0.1,
                0.1,
                description_string,
                fontsize=12,
                color="black",
            )

        plt.grid(True)
        plt.show()
