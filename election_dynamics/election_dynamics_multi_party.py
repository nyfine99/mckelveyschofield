from collections import Counter
import copy
import matplotlib.animation as animation
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


from election_dynamics.election_dynamics import ElectionDynamics
from voters.voter import Voter
from policies.policy import Policy
from utility_functions.evaluation_functions import ranked_choice_preference

COLORS_FOR_PLOTTING = ['blue', 'red', 'orange', 'green', 'purple', 'yellow', 'brown', 'pink', 'gray']

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
        if len(policies) > 9:
            print("Currently not enough colors to adequately plot!")
            # TODO: change this
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.3, 0.55, 0.55])  # Shrink plot inside the figure

        all_colors = COLORS_FOR_PLOTTING
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
    
    def animate_election(
        self, 
        policies: list[Policy],
        stop_at_majority: bool = True,
        output_folder="output",
        filename="rcv_election_animation",
        plot_verbose=True,
        fps=0.5,
    ):
        # setup
        if self.evaluation_function != ranked_choice_preference:
            print("Warning: this animation is only supported for RCV elections.")
            return
        if len(policies) < 2:
            print("Not enough policies to hold an election!")
            return
        if len(policies) > 9:
            print("Not enough colors to adequately plot!")
            return
        
        policies_arr = np.array([p.values for p in policies])
        preferences = self.tabulate_votes(policies)
        num_voters, num_candidates = preferences.shape
        active = np.ones(num_candidates, dtype=bool)
        done = {"stop": False}

        original_first_choices = preferences[:, 0]
        original_first_round_counts = np.bincount(original_first_choices, minlength=num_candidates)

        all_colors = COLORS_FOR_PLOTTING
        policy_names = [policy.name for policy in policies]
        policy_colors = all_colors[0:len(policies)]

        # plot
        # initialize the figure and axes
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

             # determining if the stopping condition has been met
            for i in np.flatnonzero(active):
                if (
                    stop_at_majority and counts[i] > total_active_votes / 2
                ) or (
                    not stop_at_majority and len(active) == 2  and counts[i] > total_active_votes / 2
                ):
                    done["stop"] = True

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
            # reformatting ordered_labels to be shorter if too long
            formatted_ordered_labels = []
            for label in ordered_labels:
                if len(label) > 18:
                    formatted_ordered_labels.append(label[:18] + "...")
                else:
                    formatted_ordered_labels.append(label)

            plt.legend(
                loc="upper left",
                bbox_to_anchor=(1.05, 1),
                borderaxespad=0.0,
                handles=ordered_handles,
                labels=formatted_ordered_labels,
            )

            title = f"Ranked-choice Voting Round {f_num+1}\n"
            too_many_policies = False
            policy_added = False
            for policy_name in policy_names:
                if len(title) + len(str(policy_name)) > 80:
                    too_many_policies = True
                    break
                else:
                    title += f"{policy_name} vs. "
                    policy_added = True
            if too_many_policies and policy_added:
                title += "..."
            elif policy_added:
                title = title[:-5]
            plt.title(title)

            plt.xlabel(f"Position on {self.issue_1}")
            plt.ylabel(f"Position on {self.issue_2}")

            if plot_verbose:
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
                    if len(sorted_items) > 3:
                            description_string += "\n..."
                    description_string += f"\n{last_policy_name} is in last place, as the first choice of {last_policy_votes} voters."
                    if not done["stop"]:
                        description_string += f"\nSo, {last_policy_name} is eliminated."
                    else:
                        # stopping condition has been met, declare winner
                        description_string += f"\n{top_policy_name} has a majority of votes, so {top_policy_name} wins!"

                fig.text(
                    0.1,
                    0.05,
                    description_string,
                    fontsize=8,
                    color="black",
                )
            
            # eliminating losers
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
            while not done["stop"]:
                if f_num >= len(policies) - 1:
                    break
                yield f_num
                f_num += 1

        ani = animation.FuncAnimation(
            fig, make_frame, frames=frame_gen(), init_func=init, save_count=len(policies)+5  # leaving some extra buffer
        )
        # Save to mp4
        ani.save(f"{output_folder}/{filename}.mp4", writer="ffmpeg", fps=fps)
        print(f"Animation saved to {output_folder}/{filename}.png")
        plt.close(fig)
    

    def create_election_sankey_diagram(
        self, 
        policies: list[Policy], 
        stop_at_majority: bool = True,
        output_folder="output",
        filename=None
    ):
        """
        Creates a round-by-round alluvial (Sankey-like) diagram using matplotlib, showing how votes are 
        redistributed as candidates are eliminated. 
        Each round is a vertical column (labeled at the top), each candidate is a horizontal row,
        each node is labeled as policy_name and sized by vote count, and flows are drawn between nodes
        in adjacent rounds.
        Note: the sankey library is not used here, as its capabilities do not allow for the desired visualization.
        """
        if self.evaluation_function != ranked_choice_preference:
            print("Warning: this animation is only supported for RCV elections.")
            return
        if len(policies) < 2:
            print("Not enough policies to hold an election!")
            return
        if len(policies) > 9:
            print("Too many policies to plot properly!")
            return
        
        preferences = self.tabulate_votes(policies)
        num_voters, num_candidates = preferences.shape
        policy_names = [p.name for p in policies]

        # run RCV and collect round-by-round vote counts
        vote_counts_by_round = self.evaluation_function(preferences, stop_at_majority=stop_at_majority, output_vote_counts=True)
        num_rounds = vote_counts_by_round.shape[0]

        # for each round, build a mapping from policy to y-position (for stacking)
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

        # prepare figure
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_xlim(-0.5, num_rounds-0.5)
        ax.set_ylim(0, total_votes)
        ax.axis('off')

        # draw round labels at the top
        for r in range(num_rounds):
            ax.text(r, total_votes + total_votes*0.04, f"Round {r+1}", ha='center', va='top', fontsize=11, fontweight='bold')

        # draw candidate row labels on the left
        for c in range(num_candidates):
            # find the first round where candidate is present
            for r in range(num_rounds):
                if vote_counts_by_round[r, c] > 0:
                    y = y_positions[r][c] + vote_counts_by_round[r, c]/2
                    ax.text(-0.3, y, policy_names[c], ha='right', va='center', fontsize=11, fontweight='bold')
                    break

        # draw nodes (rectangles for each policy in each round)
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
                    ax.text(r, y+count/2, f"{count}", ha='center', va='center', fontsize=10, color='black')

        # track which candidates are active each round
        active = np.ones(num_candidates, dtype=bool)
        current_preferences = preferences.copy()
        for r in range(num_rounds-1):
            counts = vote_counts_by_round[r]
            next_counts = vote_counts_by_round[r+1]
            # find eliminated candidate (active in this round, but not in next)
            eliminated = np.where((counts > 0) & (next_counts == 0))[0]
            if len(eliminated) == 0:
                break  # No more eliminations
            elim = eliminated[0]
            # for each voter whose top active choice is elim, find their next active choice
            mask = active[current_preferences]
            top_choice_indices = mask.argmax(axis=1)
            first_choices = current_preferences[np.arange(num_voters), top_choice_indices]
            affected_voters = np.where(first_choices == elim)[0]
            # for each affected voter, find their next preferred active candidate
            transfer_targets = []
            for v in affected_voters:
                for pref in current_preferences[v]:
                    if active[pref] and pref != elim:
                        transfer_targets.append(pref)
                        break
            # count transfers
            transfer_counter = Counter(transfer_targets)
            # draw flows from elim in round r to each target in round r+1
            y0 = y_positions[r][elim]
            y1s = {c: y_positions[r+1][c] for c in transfer_counter}
            offset0 = 0
            offsets1 = {c: 0 for c in transfer_counter}
            for c, n in transfer_counter.items():
                # dource rectangle (elim, r): (r-node_width/2, y0+offset0, node_width, n)
                # target rectangle (c, r+1): (r+1-node_width/2, y1s[c]+offsets1[c], node_width, n)
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
            # eliminate the candidate
            active[elim] = False
        plt.title('RCV Vote Transfers (Sankey Diagram)', fontsize=16, pad=30, fontweight='bold')
        plt.tight_layout()
        if output_folder is not None and filename is not None:
            plt.savefig(f"{output_folder}/{filename}.png", bbox_inches='tight')
            print(f"Sankey diagram saved to {output_folder}/{filename}.png")
        else:
            plt.show()
        plt.close(fig)


    def gridsearch_policy_winmap(self, policies, new_policy_name="New Policy", x_min=None, x_max=None, y_min=None, y_max=None, x_step=1, y_step=1, output_filename=None):
        """
        For each point on a 2D grid, adds a new policy at that point to the list of policies, runs compare_policies,
        and records which policy would win. Plots a soft/decayed heatmap showing the winner at each gridpoint.
        Voters are plotted as black dots. The color intensity decays with distance from each grid point.
        Params:
            policies (list[Policy]): List of existing Policy objects
            x_min, x_max, y_min, y_max (float): bounds of the grid
            x_step, y_step (float): step size for the grid
            output_filename (str): if provided, saves the plot to this file
        """
        # input validation
        if len(policies) < 2:
            print("Not enough policies to hold an election!")
            return
        if len(policies) > 9:
            print("Too many policies to plot properly!")
            return

        # Original grid for computation
        if x_min is None:
            x_min = min([v.ideal_policy.values[0] for v in self.voters]) - 1
        if x_max is None:
            x_max = max([v.ideal_policy.values[0] for v in self.voters]) + 1
        if y_min is None:
            y_min = min([v.ideal_policy.values[1] for v in self.voters]) - 1
        if y_max is None:
            y_max = max([v.ideal_policy.values[1] for v in self.voters]) + 1
        x_vals = np.arange(x_min, x_max + x_step, x_step)
        y_vals = np.arange(y_min, y_max + y_step, y_step)
        win_map = np.zeros((len(y_vals), len(x_vals)), dtype=int)
        policy_names = [p.name for p in policies]
        n_policies = len(policies)
        
        # Compute win map at original grid resolution
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                new_policy = Policy(np.array([x, y]), name="GridPolicy")
                test_policies = policies + [new_policy]
                winner_idx = self.compare_policies(test_policies)
                win_map[i, j] = winner_idx

        # Apply Gaussian smoothing to create decay within each pixel
        # Create finer display grid (10x finer resolution)
        display_factor = 10
        x_display = np.linspace(x_min, x_max, len(x_vals) * display_factor)
        y_display = np.linspace(y_min, y_max, len(y_vals) * display_factor)
        X_display, Y_display = np.meshgrid(x_display, y_display)
        
        # Interpolate win map to display resolution
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        values = win_map.ravel()
        win_map_display = griddata(points, values, (X_display, Y_display), method='nearest')
        
        # Create intensity maps at display resolution
        intensity_maps = np.zeros((n_policies + 1, len(y_display), len(x_display)))
        for policy_idx in range(n_policies + 1):
            mask = (win_map_display == policy_idx)
            intensity_maps[policy_idx] = mask.astype(float)
        
        # Apply Gaussian smoothing to each intensity map
        sigma = 2  # Smoothing parameter - increased for more decay within each pixel
        for i in range(n_policies + 1):
            intensity_maps[i] = gaussian_filter(intensity_maps[i], sigma=sigma)
        
        # Normalize intensity maps
        total_intensity = np.sum(intensity_maps, axis=0, keepdims=True)
        total_intensity[total_intensity == 0] = 1
        norm_intensity = intensity_maps / total_intensity
        
        # color and name vis settings
        all_colors = COLORS_FOR_PLOTTING
        policy_colors = all_colors[0:len(policies)+1]

        policy_rgbs = np.array([to_rgb(policy_colors[i]) for i in range(n_policies + 1)])
        
        # Compose RGB image
        rgb_img = np.tensordot(norm_intensity.transpose(1, 2, 0), policy_rgbs, axes=([2], [0]))
        # Plot voters as black dots
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_axes([0.1, 0.15, 0.5, 0.75])  # Shrink plot inside the figure
        voter_arr = np.array([v.ideal_policy.values for v in self.voters])
        ax.scatter(voter_arr[:, 0], voter_arr[:, 1], c='k', s=10, label='Voters', zorder=10)
        
        # Plotting heatmap
        ax.imshow(rgb_img, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='auto')
        # Optionally, plot policy locations
        for idx, p in enumerate(policies):
            ax.scatter(p.values[0], p.values[1], c=[policy_rgbs[idx]], s=200, edgecolor='k', marker='o', label=policy_names[idx], zorder=11)
        ax.set_xlabel(self.issue_1)
        ax.set_ylabel(self.issue_2)
        ax.set_title(f'Winning Policy after Insertion of {new_policy_name} at each Grid Point')
        
        # Create legend with policy locations and background color mapping
        legend_elements = []
        
        # Add policy location markers
        for idx, p in enumerate(policies):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=policy_rgbs[idx], 
                                            markeredgecolor='k', markersize=10, 
                                            label=f'{policy_names[idx]} (existing)'))
        
        # Add background color mapping
        for idx in range(n_policies + 1):
            if idx < len(policies):
                policy_name = policy_names[idx]
            else:
                policy_name = "New Policy"
            
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=policy_rgbs[idx], 
                                               alpha=0.7, label=f'Background: {policy_name} wins'))
        
        # Add voter marker
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='k', markersize=6, 
                                        label='Voters'))
        
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.05, 1), 
                borderaxespad=0.0, fontsize=9)
        if output_filename:
            plt.savefig(output_filename, bbox_inches='tight')
            print(f"Gridsearch winmap saved to {output_filename}")
        else:
            plt.show()
        plt.close(fig)
