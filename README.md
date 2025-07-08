# mckelveyschofield

![run_1](https://github.com/user-attachments/assets/02606108-7236-42b9-b1e4-841a8c6247be)

## Description

The [McKelvey-Schofield Chaos Theorem](https://en.wikipedia.org/wiki/McKelvey%E2%80%93Schofield_chaos_theorem) guarantees that, provided certain conditions are met, a set of voters with multidimensional preferences can be made to choose any policy over any other through a series of binary choices. This repo implements an algorithm which demostrates this visually, as well as providing the infrastructure for future expansions of this project towards exploring ranked-choice voting, finding the shortest possible path between any two policy positions, etc.

This project is still very much underway. I hope to elaborate more on much of the above, and much of the following, in a later version of this README.

## Features

- Modeling elections with Euclidean voters (i.e. voters with a simple Euclidean distance utility function).
- Modeling elections with taxicab/L1 norm voters (i.e. voters with a taxicab distance utility function).
- Finding a path, through a series of forced binary choices presented to voters, between any two policy positions in a two-dimensional plane.

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

## Usage

sample_script.py provides a good outline for usage - simply edit as desired, then run `python sample_script.py` in your terminal. 

To run many animations with saved results, use multirun.py.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Planned Additions

- McKelvey-Schofield pathfinding capabilities for voters with any concave (and even some non-concave) utility functions.
- A justification as to the validity of the winset boundary algorithm.
- Exploration and possible implementation of non-greedy McKelvey-Schofield pathfinding.
- Work with ranked-choice voting capabilities/animation.

![pets_election](https://github.com/user-attachments/assets/66834c4c-b68e-4000-953d-8683cc284afe)
