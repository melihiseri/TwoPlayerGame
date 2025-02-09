import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime



def plot_all_actions(a1_past: list[float], a2_past: list[float], filename: str = "Player_actions.pdf") -> None:
    """
    Plots the action history of both players over time.

    Args:
        a1_past (list of float): Actions taken by Player 1.
        a2_past (list of float): Actions taken by Player 2.

    Returns:
        None
    """
    if not a1_past or not a2_past:
        print("Error: Action lists are empty.")
        return

    min_length = min(len(a1_past), len(a2_past))
    a1_past, a2_past = a1_past[:min_length], a2_past[:min_length]

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=600)
    plt.plot(range(min_length), a1_past, label="Player 1 Actions", color="firebrick", marker="o", markersize=5.0, linestyle="--", linewidth=0.05)
    plt.plot(range(min_length), a2_past, label="Player 2 Actions", color="navy", marker="o", markersize=3.0, linestyle="--", linewidth=0.05)

    # plt.xlabel("Number of Games", fontsize=9, fontweight="bold")
    # plt.ylabel("Actions Taken", fontsize=9, fontweight="bold")
    # plt.title("Player Actions of Each Game", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    # plt.legend(fontsize=9, loc="best", frameon=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(filename)
    plt.savefig(f"Player_actions_{filename}.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def plot_cumulative_rewards(
        cumulative_cost_p1_history: list[float], 
        cumulative_cost_p2_history: list[float],
        filename: str = "Cumulative_costs.pdf"
) -> None:
    """
    Plots the cumulative costs (negative rewards) of both players over time.

    Args:
        cumulative_cost_p1_history (list of float): Cumulative costs for Player 1.
        cumulative_cost_p2_history (list of float): Cumulative costs for Player 2.

    Returns:
        None
    """
    if not cumulative_cost_p1_history or not cumulative_cost_p2_history:
        print("Error: Cumulative cost lists are empty.")
        return

    min_length = min(len(cumulative_cost_p1_history), len(cumulative_cost_p2_history))
    cumulative_cost_p1_history = cumulative_cost_p1_history[:min_length]
    cumulative_cost_p2_history = cumulative_cost_p2_history[:min_length]

    plt.figure(figsize=(10, 5))
    plt.plot(range(min_length), cumulative_cost_p1_history, label="Player 1 Cumulative Cost", color="firebrick", marker="o", markersize=2)
    plt.plot(range(min_length), cumulative_cost_p2_history, label="Player 2 Cumulative Cost", color="navy", marker="o", markersize=2)

    plt.xlabel("Number of the Game")
    plt.ylabel("Cumulative Cost")
    plt.title("Cumulative Costs Over Time")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.6)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(f"Cumulative_costs_{filename}", format="pdf", bbox_inches="tight")
    plt.close()



def plot_moving_average_rewards(
        cumulative_cost_p1_history: list[float], 
        cumulative_cost_p2_history: list[float], 
        n_last: int = 100,
        filename: str = "Moving_average_rewards.pdf"
) -> None:
    """
    Plots the moving average of per-game costs (not cumulative costs) over time.

    Args:
        cumulative_cost_p1_history (list of float): Cumulative cost history for Player 1.
        cumulative_cost_p2_history (list of float): Cumulative cost history for Player 2.
        n_last (int, optional): Window size for computing the moving average. Default is 100.

    Returns:
        None
    """
    if not cumulative_cost_p1_history or not cumulative_cost_p2_history:
        print("Error: Cumulative cost lists are empty.")
        return

    # --- Convert cumulative costs to per-game costs
    def compute_per_game_cost(cumulative_cost: list[float]) -> list[float]:
        """ Converts cumulative costs to per-game costs. """
        return [cumulative_cost[0]] + [
            cumulative_cost[i] - cumulative_cost[i - 1] for i in range(1, len(cumulative_cost))
        ]

    per_game_p1 = compute_per_game_cost(cumulative_cost_p1_history)
    per_game_p2 = compute_per_game_cost(cumulative_cost_p2_history)

    # --- Compute moving average
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    moving_avg_p1 = moving_average(per_game_p1, n_last)
    moving_avg_p2 = moving_average(per_game_p2, n_last)

    min_length = min(len(moving_avg_p1), len(moving_avg_p2))
    moving_avg_p1 = moving_avg_p1[:min_length]
    moving_avg_p2 = moving_avg_p2[:min_length]

    plt.figure(figsize=(10, 5))
    plt.plot(range(min_length), moving_avg_p1, label="Player 1 Moving Avg Cost", color="firebrick", marker="o", markersize=2)
    plt.plot(range(min_length), moving_avg_p2, label="Player 2 Moving Avg Cost", color="navy", marker="o", markersize=2)

    plt.xlabel("Time Step")
    plt.ylabel("Moving Average Per-Game Cost")
    plt.title(f"Moving Average of Per-Game Costs (Window = {n_last})")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.6)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(f"Moving_average_rewards_{filename}", format="pdf", bbox_inches="tight")
    plt.close()


def setup_plots(player1, player2):
    """
    Initializes and returns the figure and axes for visualizing the two-player game.

    Args:
        player1 (Player): The first player object.
        player2 (Player): The second player object.

    Returns:
        tuple: (fig, ax_dict) where fig is the matplotlib figure and ax_dict is a dictionary of axes.
    """
    fig, axes = plt.subplots(3, 4, figsize=(32, 24))  # 3 rows, 4 columns

    ax_dict = {
        "action_p1": axes[0, 0],
        "action_p2": axes[1, 0],
        "cost_p1": axes[0, 1],
        "cost_p2": axes[1, 1],
        "expected_cost_p1": axes[0, 2],
        "expected_cost_p2": axes[1, 2],
        "player_actions": axes[0, 3],
        "state_transitions": axes[1, 3],
        "cumulative_costs": plt.subplot2grid((3, 4), (2, 0), colspan=4),
    }

    titles = {
        "action_p1": "Player 1 Action Networks",
        "action_p2": "Player 2 Action Networks",
        "cost_p1": "Player 1 Cost Networks",
        "cost_p2": "Player 2 Cost Networks",
        "expected_cost_p1": "Player 1 Expected Costs",
        "expected_cost_p2": "Player 2 Expected Costs",
        "player_actions": "Player Actions Over Time",
        "state_transitions": "State Transitions Over Time",
        "cumulative_costs": "Cumulative Realized Costs",
    }
    for key, title in titles.items():
        ax_dict[key].set_title(title)
    

    # --- Colors and markers
    colors = {"p1": "firebrick", "p2": "navy"}
    markers = {"p1": "o", "p2": "s"}


    # --- Set axis limits
    
    for ax in [ax_dict["action_p1"], ax_dict["action_p2"]]:
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.1)
    
    for ax in [ax_dict["cost_p1"], ax_dict["cost_p2"]]:
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.25, 1.25)

    for ax in [ax_dict["expected_cost_p1"], ax_dict["expected_cost_p2"]]:
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 1.0)

    for ax in [ax_dict["player_actions"], ax_dict["state_transitions"]]:
        ax.set_xlim(0, 100)  # Will be dynamically updated in update_plots
        ax.set_ylim(-0.1, 1.1)

    ax_dict["cumulative_costs"].set_xlim(0, 100)  # Dynamically adjusted later
    ax_dict["cumulative_costs"].set_ylim(0, 100)  # Dynamically adjusted later
    ax_dict["cumulative_costs"].set_title("Cumulative Costs")



    # --- Initialize Empty Lines for Each Plot
    
    # -- Player Action Networks
    lines_action_p1 = [
        ax_dict["action_p1"].plot([], [], label=f"Player 1 - Action {i}", color=colors["p1"], marker=markers["p1"], ms=2.0)[0]
        for i in range(len(player1.action_networks))
    ]
    lines_action_p2 = [
        ax_dict["action_p2"].plot([], [], label=f"Player 2 - Action {i}", color=colors["p2"], marker=markers["p2"], ms=2.0)[0]
        for i in range(len(player2.action_networks))
    ]

    # -- Player Cost Networks
    lines_cost_p1 = [
        ax_dict["cost_p1"].plot([], [], label=f"Player 1 - Cost {i}", color=colors["p1"], marker=markers["p1"], ms=2.0)[0]
        for i in range(len(player1.cost_networks))
    ]
    lines_cost_p2 = [
        ax_dict["cost_p2"].plot([], [], label=f"Player 2 - Cost {i}", color=colors["p2"], marker=markers["p2"], ms=2.0)[0]
        for i in range(len(player2.cost_networks))
    ]

    # -- Expected Costs
    line_exp_cost_p1, = ax_dict["expected_cost_p1"].plot([], [], label="Player 1 Expected Costs", color=colors["p1"], marker=markers["p1"], ms=2.0)
    line_exp_cost_p2, = ax_dict["expected_cost_p2"].plot([], [], label="Player 2 Expected Costs", color=colors["p2"], marker=markers["p2"], ms=2.0)

    # -- Player Actions
    line_action_p1, = ax_dict["player_actions"].plot([], [], label="Player 1 Action", color=colors["p1"], marker=markers["p1"], ms=8.0, linestyle="--", linewidth=0.05)
    line_action_p2, = ax_dict["player_actions"].plot([], [], label="Player 2 Action", color=colors["p2"], marker=markers["p2"], ms=5.0, linestyle="--", linewidth=0.05)

    # -- State Transitions
    line_transition_p1, = ax_dict["state_transitions"].plot([], [], label="Player 1 Transition", color=colors["p1"], marker=markers["p1"], ms=8.0, linestyle="--", linewidth=0.1)
    line_transition_p2, = ax_dict["state_transitions"].plot([], [], label="Player 2 Transition", color=colors["p2"], marker=markers["p2"], ms=5.0, linestyle="--", linewidth=0.1)

    # -- Cumulative Costs
    line_cum_cost_p1, = ax_dict["cumulative_costs"].plot([], [], label="Player 1 Cumulative Cost", color=colors["p1"], marker=markers["p1"], ms=5.0)
    line_cum_cost_p2, = ax_dict["cumulative_costs"].plot([], [], label="Player 2 Cumulative Cost", color=colors["p2"], marker=markers["p2"], ms=5.0)

    plt.tight_layout()
    plt.draw()

    return fig, ax_dict


def update_plots(fig, ax_dict, players, action_space, action_space_tensor, histories, n_last=16):
    """
    Updates all plots dynamically during gameplay.

    Args:
        fig (matplotlib.figure.Figure): The figure object containing all plots.
        ax_dict (dict): Dictionary containing subplot axes.
        players (dict): Dictionary containing Player 1 and Player 2 objects.
        action_space (list): The list of possible actions.
        action_space_tensor (torch.Tensor): The tensor representation of action space.
        histories (dict): Dictionary containing historical data (actions, states, cumulative costs).
        n_last (int, optional): Number of recent time steps to display. Default is 16.

    Returns:
        None
    """

    # --- Player 1 Action Networks
    for i, net in enumerate(players["p1"].action_networks):
        action_output_p1 = net(players["p1"].action_space_tensor).detach().cpu().squeeze(1).numpy()
        ax_dict["action_p1"].lines[i].set_data(action_space, action_output_p1)

    # --- Player 2 Action Networks
    for i, net in enumerate(players["p2"].action_networks):
        action_output_p2 = net(players["p2"].action_space_tensor).detach().cpu().squeeze(1).numpy()
        ax_dict["action_p2"].lines[i].set_data(action_space, action_output_p2)

    # --- Player 1 Cost Networks
    for i, net in enumerate(players["p1"].cost_networks):
        cost_output_p1 = net(players["p1"].action_space_tensor).detach().cpu().squeeze(1).numpy()
        ax_dict["cost_p1"].lines[i].set_data(action_space, cost_output_p1)

    # --- Player 2 Cost Networks
    for i, net in enumerate(players["p2"].cost_networks):
        cost_output_p2 = net(players["p2"].action_space_tensor).detach().cpu().squeeze(1).numpy()
        ax_dict["cost_p2"].lines[i].set_data(action_space, cost_output_p2)

    # --- Expected Costs
    ax_dict["expected_cost_p1"].lines[0].set_data(action_space, players["p1"].my_expected_cost)
    ax_dict["expected_cost_p2"].lines[0].set_data(action_space, players["p2"].my_expected_cost)

    # --- Player Actions (Recent n_last)
    x_range = range(max(0, len(histories["a1"]) - n_last), len(histories["a1"]))
    ax_dict["player_actions"].lines[0].set_data(x_range, histories["a1"][-n_last:])
    ax_dict["player_actions"].lines[1].set_data(x_range, histories["a2"][-n_last:])
    ax_dict["player_actions"].set_xlim(x_range.start, x_range.stop)

    # --- Player State Transitions (Recent n_last)
    x_range_states = range(max(0, len(histories["X1"]) - n_last), len(histories["X1"]))
    ax_dict["state_transitions"].lines[0].set_data(x_range_states, histories["X1"][-n_last:])
    ax_dict["state_transitions"].lines[1].set_data(x_range_states, histories["X2"][-n_last:])
    ax_dict["state_transitions"].set_xlim(x_range_states.start, x_range_states.stop)


    # --- Costs (Averaged over n_last)
    stepwise_costs_p1 = np.diff(histories["cost_p1"], prepend=0)
    stepwise_costs_p2 = np.diff(histories["cost_p2"], prepend=0)

    avg_cum_cost_p1 = np.convolve(stepwise_costs_p1, np.ones(n_last) / n_last, mode="valid")
    avg_cum_cost_p2 = np.convolve(stepwise_costs_p2, np.ones(n_last) / n_last, mode="valid")

    cumulative_x_range = range(len(histories["cost_p1"]) - len(avg_cum_cost_p1), len(histories["cost_p1"]))

    ax_dict["cumulative_costs"].lines[0].set_data(list(cumulative_x_range), avg_cum_cost_p1)
    ax_dict["cumulative_costs"].lines[1].set_data(list(cumulative_x_range), avg_cum_cost_p2)
    ax_dict["cumulative_costs"].set_xlim(cumulative_x_range.start, cumulative_x_range.stop)

    min_avg_cum_cost = min(min(avg_cum_cost_p1, default=0), min(avg_cum_cost_p2, default=0))
    max_avg_cum_cost = max(max(avg_cum_cost_p1, default=1), max(avg_cum_cost_p2, default=1))
    ax_dict["cumulative_costs"].set_ylim(min_avg_cum_cost - 0.2, max_avg_cum_cost + 0.2)
