=======
# TwoPlayerGame

## Overview
A Python package that simulates a two-player repeated game. 

## Installation
To install the package, clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/TwoPlayerGame.git
cd TwoPlayerGame
pip install -e .
```

Alternatively, you can install dependencies manually:

```bash
pip install torch numpy matplotlib
```

## Usage

### Running Experiments
To run a full set of experiments and generate visualizations:

```bash
python -m TwoPlayerGame.main
```

This will execute predefined experiments and save animation and plots.


### Manual Execution
To run a manual experiment with a specific configuration, experiments are defined in `main.py/manual_animation`. Tuples like `(0.3, -0.1)` are parameters for the first player, and `(c1, b1, c2, b2)` can be used to configure player 2.


## Learning Algorithm Customization
The learning dynamics of the players can be tuned by modifying the `Universe` class and player-specific parameters. The `self.i` parameter within each `Player` instance determines its role and influences its learning behavior. Adjusting this and other parameters like `N_act_nets`, `N_cost_nets`, learning rates, and memory settings allows control over design of the learning algorithms.

## Package Structure
```
TwoPlayerGame/
│── game.py             # Core game logic. Player and Universe classes.
│── main.py             # Entry point for running experiments
│── memory.py           # Memory management for action & cost history
│── networks.py         # Neural network models for action & cost estimation
│── utils.py            # Utility functions (seeding, sampling, regularization)
│── cost_functions.py   # Cost function definitions for players
│── visualization.py    # Plotting and visualization utilities
│── setup.py            # Package setup configuration
```

## Dependencies
The project relies on:
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

## License
This project is licensed under the Creative Commons License. See the LICENSE file for details.
## Author
Melih Iseri
