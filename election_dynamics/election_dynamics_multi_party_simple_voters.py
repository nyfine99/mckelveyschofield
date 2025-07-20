import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from ordered_set import OrderedSet


from election_dynamics.election_dynamics_multi_party import ElectionDynamicsMultiParty
from policies.policy import Policy
from utility_functions.evaluation_functions import ranked_choice_preference
from voters.simple_voter import SimpleVoter


class ElectionDynamicsMultiPartySimpleVoters(ElectionDynamicsMultiParty):
    def __init__(
        self,
        voters: list[SimpleVoter],
        issue_1: str = "Issue 1",
        issue_2: str = "Issue 2",
    ):
        self.voters = voters
        self.voter_arr = np.array([voter.ideal_policy.values for voter in self.voters])
        self.evaluation_function = ranked_choice_preference
        self.tiebreak_func = None
        self.issue_1 = issue_1  # Issue 1 name
        self.issue_2 = issue_2  # Issue 2 name


    def tabulate_votes(self, policies: list[Policy]) -> np.ndarray:
        """
        Computes ranked preferences for each voter by distance to policy (closer = higher utility).
        Ties are broken by the index of the policy (lower index = higher utility).
        Returns a 2D np.ndarray where each row is a voter's ranked policy indices (best to worst).
        Params:
            policies (list[Policy]): List of Policy objects.
        Returns:
            np.ndarray: shape (num_voters, num_policies)
        """
        policies_arr = np.array([p.values for p in policies])
        dists = np.linalg.norm(self.voter_arr[:, np.newaxis, :] - policies_arr[np.newaxis, :, :], axis=2)
        preferences = np.argsort(dists, axis=1)
        return preferences


    def animate_election(self, policies: list[Policy]):
        # TODO: include option for stopping at majority or at final two
        policies_arr = np.array([p.values for p in policies])
        preferences = self.tabulate_votes(policies)
        num_voters, num_candidates = preferences.shape
        active = np.ones(num_candidates, dtype=bool)

        original_first_choices = preferences[:, 0]
        original_first_round_counts = np.bincount(original_first_choices, minlength=num_candidates)

    
        # color and name vis settings
        mcolors_dict = mcolors.TABLEAU_COLORS

        # ensuring blue and red are the first two colors used
        del mcolors_dict['tab:blue']
        del mcolors_dict['tab:red']
        all_colors = ['blue', 'red'] + list(mcolors_dict.values())
        policy_names = [policy.name for policy in policies]
        policy_colors = all_colors[0:len(policies)]

        # plot
        # initialize the figure and axes
        if len(policies) < 1:
            print("Not enough policies to hold an election!")
            return
        if len(policies) > 10:
            print("Currently not enough colors to adequately plot!")
            # TODO: change this
            return
        fig = plt.figure()

        def make_frame(f_num):
            # TODO: should this be subplots?
            plt.clf()  # Clear the current axes/figure

            mask = active[preferences]
            top_choice_indices = mask.argmax(axis=1)
            first_choices = preferences[np.arange(num_voters), top_choice_indices]

            counts = np.bincount(first_choices, minlength=num_candidates)
            total_active_votes = counts[active].sum()
            ax = fig.add_axes([0.1, 0.3, 0.55, 0.55])  # Shrink plot inside the figure

            # plotting all voters and policies
            voters_by_vote = {}
            for i in range(len(active)):
                if active[i]:
                    voters_by_vote[i] = []

            for i, choice in enumerate(first_choices):
                voters_by_vote[choice].append(self.voters[i])

            for k in voters_by_vote.keys():
                if active[k] and voters_by_vote[k]:
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
                        policies_arr[k,0],
                        policies_arr[k,1],
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
            ordered_labels = [
                label
                for label in desired_order
                if label in label_to_handle
            ]
            plt.legend(
                loc="upper left",
                bbox_to_anchor=(1.05, 1),
                borderaxespad=0.0,
                handles=ordered_handles,
                labels=ordered_labels,
            )
            title = f"Ranked-choice Voting Round {f_num+1}\n" + " vs. ".join(policy_names[:3])
            if len(policy_names) > 3:
                title += " vs. ..."
            plt.title(title)
            plt.xlabel(f"Position on {self.issue_1}")
            plt.ylabel(f"Position on {self.issue_2}")

            if True:
                # sorting voters_by_vote by first round vote count
                description_string = "In first-choice selections"
                sorted_items = sorted(voters_by_vote.items(), key=lambda item: len(item[1]), reverse=True)
                for i in range(len(sorted_items)):
                    sorted_items[i] = list(sorted_items[i])
                    sorted_items[i][1] = len(sorted_items[i][1])
                top_policy_idx = sorted_items[0][0]
                top_policy_name = policy_names[top_policy_idx]
                top_policy_votes = sorted_items[0][1]
                description_string = f"\n{top_policy_name} leads, as the first choice of {top_policy_votes} voters."

                if len(sorted_items) > 1:
                    second_policy_idx = sorted_items[1][0]
                    second_policy_name = policy_names[second_policy_idx]
                    second_policy_votes = sorted_items[1][1]
                    if len(sorted_items) > 2:
                        description_string += f"\n{second_policy_name} is in second place, as the first choice of {second_policy_votes} voters."
                    else:
                        # this is the last place candidate, phrase accordingly
                        description_string += f"\n{second_policy_name} trails, as the first choice of {second_policy_votes} voters."
                        description_string += f"\nSo, {top_policy_name} wins!"

                if len(sorted_items) > 2:
                    last_policy_idx = sorted_items[-1][0]
                    last_policy_name = policy_names[last_policy_idx]
                    last_policy_votes = sorted_items[-1][1]
                    description_string += "\n..."
                    description_string += f"\n{last_policy_name} is in last place, as the first choice of {last_policy_votes} voters."
                    description_string += f"\nSo, {last_policy_name} is eliminated."

                fig.text(
                    0.1,
                    0.05,
                    description_string,
                    fontsize=8,
                    color="black",
                )

            # eliminating losers
            for i in np.flatnonzero(active):
                if counts[i] > total_active_votes / 2:
                    # TODO: implement an option that allows for stopping after a candidate
                    # receives over 50%
                    pass

            min_votes = counts[active].min()
            lowest = np.flatnonzero((counts == min_votes) & active)

            if len(lowest) > 1:
                orig_support = original_first_round_counts[lowest]
                min_orig = orig_support.min()
                lowest = lowest[orig_support == min_orig]

            to_eliminate = lowest.min()
            active[to_eliminate] = False

        def init():
            return None
        
        def frame_gen():
            f_num = 0
            while True:
                if f_num >= len(policies) - 1:
                    break
                yield f_num
                f_num += 1

        ani = animation.FuncAnimation(
            fig, make_frame, frames=frame_gen(), init_func=init, save_count=len(policies)+5  # leaving some extra buffer
        )
        # Save to mp4
        ani.save(f"output/rcv.mp4", writer="ffmpeg", fps=.5)
        plt.close(fig)
