import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


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
        mcolors_dict = mcolors.TABLEAU_COLORS

        # ensuring blue and red are the first two colors used
        del mcolors_dict['tab:blue']
        del mcolors_dict['tab:red']
        all_colors = ['blue', 'red'] + list(mcolors_dict.values())
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
