import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
        if len(policies) < 2:
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
    

    def create_election_sankey_diagram(self, policies: list[Policy], output_filepath: str = None):
        """
        Creates a round-by-round alluvial (Sankey-like) diagram using matplotlib, showing how votes are 
        redistributed as candidates are eliminated. 
        Each round is a vertical column (labeled at the top), each candidate is a horizontal row,
        each node is labeled as policy_name and sized by vote count, and flows are drawn between nodes
        in adjacent rounds.
        Note: the sankey library is not used here, as its capabilities do not allow for the desired visualization.

        Params:
            policies (list[Policy]): list of Policy objects.
            output_filepath (str): if provided, saves the plot to this filepath (e.g., 'output/rcv_alluvial.png');
                if not provided, the plot is displayed in a window
        """
        if len(policies) < 2:
            print("Not enough policies to hold an election!")
            return
        if len(policies) > 10:
            print("Currently not enough colors to adequately plot!")
            # TODO: change this
            return
        
        preferences = self.tabulate_votes(policies)
        num_voters, num_candidates = preferences.shape
        policy_names = [p.name for p in policies]

        # Run RCV and collect round-by-round vote counts
        vote_counts_by_round = self.evaluation_function(preferences, output_vote_counts=True)
        vote_counts_by_round = vote_counts_by_round
        num_rounds = vote_counts_by_round.shape[0]

        # For each round, build a mapping from policy to y-position (for stacking)
        y_positions = []
        for r in range(num_rounds):
            y = 0
            pos = {}
            for c in range(num_candidates):
                if vote_counts_by_round[r, c] > 0:
                    pos[c] = y
                    y += vote_counts_by_round[r, c]
            y_positions.append(pos)
        total_votes = np.sum(vote_counts_by_round[0])

        # Prepare figure
        fig, ax = plt.subplots(figsize=(2*num_rounds+2, 1.2*num_candidates+2))
        ax.set_xlim(-0.5, num_rounds-0.5)
        ax.set_ylim(0, total_votes)
        ax.axis('off')

        # Draw round labels at the top
        for r in range(num_rounds):
            ax.text(r, total_votes + total_votes*0.04, f"Round {r+1}", ha='center', va='bottom', fontsize=14, fontweight='bold')

        # Draw candidate row labels on the left
        for c in range(num_candidates):
            # Find the first round where candidate is present
            for r in range(num_rounds):
                if vote_counts_by_round[r, c] > 0:
                    y = y_positions[r][c] + vote_counts_by_round[r, c]/2
                    ax.text(-0.7, y, policy_names[c], ha='right', va='center', fontsize=12, fontweight='bold')
                    break

        # Draw nodes (rectangles for each policy in each round)
        node_width = 0.3
        node_rects = {}
        for r in range(num_rounds):
            for c in range(num_candidates):
                count = vote_counts_by_round[r, c]
                if count > 0:
                    y = y_positions[r][c]
                    rect = patches.Rectangle((r-node_width/2, y), node_width, count, facecolor=f'C{c}', edgecolor='k', alpha=0.8)
                    ax.add_patch(rect)
                    node_rects[(c, r)] = (r-node_width/2, y, node_width, count)
                    # Label
                    ax.text(r, y+count/2, f"{count}", ha='center', va='center', fontsize=10, color='white' if count > total_votes/10 else 'black')

        # Track which candidates are active each round
        active = np.ones(num_candidates, dtype=bool)
        current_preferences = preferences.copy()
        for r in range(num_rounds-1):
            counts = vote_counts_by_round[r]
            next_counts = vote_counts_by_round[r+1]
            # Find eliminated candidate (active in this round, but not in next)
            eliminated = np.where((counts > 0) & (next_counts == 0))[0]
            if len(eliminated) == 0:
                break  # No more eliminations
            elim = eliminated[0]
            # For each voter whose top active choice is elim, find their next active choice
            mask = active[current_preferences]
            top_choice_indices = mask.argmax(axis=1)
            first_choices = current_preferences[np.arange(num_voters), top_choice_indices]
            affected_voters = np.where(first_choices == elim)[0]
            # For each affected voter, find their next preferred active candidate
            transfer_targets = []
            for v in affected_voters:
                for pref in current_preferences[v]:
                    if active[pref] and pref != elim:
                        transfer_targets.append(pref)
                        break
            # Count transfers
            from collections import Counter
            transfer_counter = Counter(transfer_targets)
            # Draw flows from elim in round r to each target in round r+1
            y0 = y_positions[r][elim]
            y1s = {c: y_positions[r+1][c] for c in transfer_counter}
            offset0 = 0
            offsets1 = {c: 0 for c in transfer_counter}
            for c, n in transfer_counter.items():
                # Source rectangle (elim, r): (r-node_width/2, y0+offset0, node_width, n)
                # Target rectangle (c, r+1): (r+1-node_width/2, y1s[c]+offsets1[c], node_width, n)
                verts = [
                    (r+node_width/2, y0+offset0),
                    (r+1-node_width/2, y1s[c]+offsets1[c]),
                    (r+1-node_width/2, y1s[c]+offsets1[c]+n),
                    (r+node_width/2, y0+offset0+n)
                ]
                polygon = patches.Polygon(verts, closed=True, facecolor=f'C{elim}', edgecolor='none', alpha=0.4)
                ax.add_patch(polygon)
                offset0 += n
                offsets1[c] += n
            # Eliminate the candidate
            active[elim] = False
        plt.title('RCV Vote Transfers (Sankey Diagram)', fontsize=16)
        plt.tight_layout()
        if output_filepath:
            plt.savefig(output_filepath, bbox_inches='tight')
            print(f"Alluvial diagram saved to {output_filepath}")
        else:
            plt.show()
        plt.close(fig)
