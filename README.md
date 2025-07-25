# mckelveyschofield

![run_1](https://github.com/user-attachments/assets/02606108-7236-42b9-b1e4-841a8c6247be)

## Description

The [McKelvey-Schofield Chaos Theorem](https://en.wikipedia.org/wiki/McKelvey%E2%80%93Schofield_chaos_theorem) guarantees that, provided certain conditions are met, a set of voters with multidimensional preferences can be made to choose any policy over any other through a series of binary choices. This repository contains scripts capable of demostrating this visually through several different types of plots and animations, as well as providing the infrastructure for future expansions of this project towards exploring other facets of electoral weirdness.

Additionally, this project contains scripts capable of simulating ranked-choice voting, and visualizing the results of such elections through by-round animations and Sankey diagrams. And again, the infrastructure is provided to enable future expansions to explore other aspects of ranked-choice voting.

## Features

Assuming:
- Voters and candidates/policies are represented as points in an n-dimensional Euclidean policy space (typically 2D).
- Voters have full knowledge of their own preferences and the positions of all candidates/policies.
- Voters vote sincerely for the option that gives them the highest utility (or, in the case of ranked-choice voting, rank their options by greatest utility).

This project supports:
- Modeling and visualizing two-party and ranked-choice voting elections with customizable voter types (currently assuming rational voters with simple Manhattan or Euclidean distance-based utility).
- Identifying the set of policies that can defeat a given or incumbent policy via majority rule.
- Finding a path, through a sequence of forced binary choices presented to voters, between any two policy positions in a two-dimensional plane (as McKelvey-Schofield guarantees possible, in most cases).
- Applying these tools to real-world voter data (though not necessarily for real-world conclusions, given the simplified assumptions of the model; rather, to explore its implications and structure).

**See the `docs` folder for a deeper dive into the capabilities offered in this repository!**

## Installation

1. Clone the repository:
   ```
   https://github.com/nyfine99/mckelveyschofield.git
   ```
2. Navigate to the project directory:
   ```
   cd mckelveyschofield
   ```
3. Install dependencies
   ```
   pip install deap
   pip install matplotlib
   pip install numba
   pip install numpy
   pip install ordered_set
   pip install pandas
   pip install scipy
   ```
   
   Separately, ffmpeg will also require installation to get any of the animations working.

## Usage

All of the scripts in the `scripts` folder provide a good outline for usage. To run a script, for example, `euclidean_electorate.py`, run

`python -m scripts.euclidean_electorate` 

in your terminal.

To run many animations with saved results, use one of the multirun files. For example, to run `multirun_euclidean.py`, run

`python -m scripts.multirun_euclidean`

in your terminal.

Note: for scripts which output images and/or animations, you will need to create an `output` folder in advance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Planned Additions

- McKelvey-Schofield pathfinding capabilities for voters with any weighted L1 or L2 norm utility functions.
- A justification as to the validity of the winset boundary algorithm.
- Exploration and possible implementation of non-greedy McKelvey-Schofield pathfinding.
- Exploration of ranked-choice voting with limits on how many policies voters can put on their ballots, and finding the optimal location to insert a new policy.

![pets_election](https://github.com/user-attachments/assets/66834c4c-b68e-4000-953d-8683cc284afe)
