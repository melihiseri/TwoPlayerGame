import random
import numpy as np
import torch
from torch import optim
from .memory import Memory
from .networks import ActionEstimator, CostEstimator
from .cost_functions import cost1, cost2
from .visualization import plot_all_actions, plot_cumulative_rewards, plot_moving_average_rewards, setup_plots, update_plots
from .utils import set_seed, l4_regularization

from copy import deepcopy

class Player:    
    def __init__(self,
                 action_space: np.ndarray,
                 cost_function: callable,
                 index: int,
                 *,
                 action_memory_size: int = 64,
                 cost_memory_size: int = 64,
                 N_cost_nets: int = 6,
                 N_act_nets: int = 6,
                 c: float = 0.0,
                 action_params: dict = None,
                 cost_params: dict = None,
                 best_possible: float = -0.15):

        action_params = action_params or {"size": 32, "depth": 2, "p_drop": 0.0}
        cost_params = cost_params or {"size": 32, "depth": 2, "p_drop": 0.1}

        # Player metadata
        self.i = index
        self.c = c
        self.cost_function = cost_function
        self.best_possible = best_possible
        self.best_possible_tensor = torch.tensor(best_possible, dtype=torch.float32)
        self.action_memory_size = action_memory_size
        self.cost_memory_size = cost_memory_size
        
        # Action Space
        self.action_space = action_space
        self.action_space_tensor = torch.tensor(action_space, dtype=torch.float32).view(-1, 1)

        # Initialize memory
        self.action_memory = Memory(memory_size=action_memory_size)
        self.cost_memory = Memory(memory_size=cost_memory_size)

        # Initialize Networks
        self.cost_networks = self._initialize_networks(CostEstimator, N_cost_nets, cost_params)
        self.action_networks = self._initialize_networks(ActionEstimator, N_act_nets, action_params)

        # Optimizers
        self.cost_optimizers = [optim.AdamW(c_net.parameters(), lr=0.001, weight_decay=1e-4) for c_net in self.cost_networks]
        self.action_optimizers = [optim.AdamW(a_net.parameters(), lr=0.001, weight_decay=1e-4) for a_net in self.action_networks]

        # Ensure training mode
        for net in self.cost_networks + self.action_networks:
            net.train()

        # Tracking variables
        self.my_current_action_distribution = np.array([])
        self.my_past_action = 0.0
        self.my_past_action_list = []
        self.my_expected_cost = np.zeros_like(action_space)

    def _initialize_networks(self, NetworkClass, N, params):
        """Helper to create multiple networks of the same type."""
        return [NetworkClass(input_dim=1,
                             hidden_size=params["size"],
                             output_dim=1,
                             depth=params["depth"],
                             p_drop=params.get("p_drop", 0.1)) for _ in range(N)]
    

    def observe_and_record(self, x1: int, x2: int, a1: float = None, a2: float = None):
        """
        Records the observations into cost and action memories for the player.

        Args:
            x1 (int): State variable for Player 1.
            x2 (int): State variable for Player 2.
            a1 (float, optional): Action taken by Player 1. Defaults to None.
            a2 (float, optional): Action taken by Player 2. Defaults to None.
        """

        if self.i not in [1, 2]:
            raise ValueError(f"Invalid player index {self.i}. Expected 1 or 2.")

        if self.i == 1:
            self.cost_memory.remember(self.my_past_action,
                                      self.cost_function(0, x1, x2, self.my_past_action, a2, self.c))
            self.action_memory.remember(self.my_past_action, x2)
            
        if self.i == 2:
            self.cost_memory.remember(self.my_past_action,
                                      self.cost_function(0, x1, x2, a1, self.my_past_action, self.c))
            self.action_memory.remember(self.my_past_action, x1)

        return
        
        
    def train(self, action_rate: float = 0.001, action_epoch: int = 16, 
              cost_rate: float = 0.001, cost_epoch: int = 16):
        """
        Trains both action and cost networks.

        Args:
            action_rate (float): Learning rate for action networks. Default is 0.001.
            action_epoch (int): Number of epochs for action training. Default is 16.
            cost_rate (float): Learning rate for cost networks. Default is 0.001.
            cost_epoch (int): Number of epochs for cost training. Default is 16.
        """
        self.train_action_networks(learning_rate=action_rate, epoch=action_epoch)
        self.train_cost_networks(learning_rate=cost_rate, epoch=cost_epoch)


        
    def train_action_networks(self, learning_rate: float = 0.001, epoch: int = 1):
        """
        Trains the player's action networks using memory data.

        Args:
            learning_rate (float): Learning rate for action networks. Default is 0.001.
            epoch (int): Number of training epochs. Default is 1.
        """

        if learning_rate != 0.001:
            for optimizer in self.action_optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

        # Batch size for sampling
        b_size = 1 + min(31, len(self.action_memory.memory))
        
        # Train each action network
        for action_net, optimizer in zip(self.action_networks, self.action_optimizers):
            for ep in range(epoch):
                """
                Note that player 1 samples more uniformly, whereas player 2 is nearsighted.
                """
                if self.i == 1:
                    all_inputs, all_targets = self.action_memory.get_memory(batch_size=b_size, sampling_type='recent',
                                                                            alpha=0.9, add_randomness = True, noise_level = 0.05,
                                                                            exp_start=0.01)
                if self.i == 2:
                    all_inputs, all_targets = self.action_memory.get_memory(batch_size=b_size, sampling_type='recent',
                                                                            alpha=0.99, add_randomness = True, noise_level = 0.05,
                                                                            exp_start=0.005)
                optimizer.zero_grad()
                all_outputs = action_net(all_inputs)

                # Loss function -- l4 regularization
                loss = 10 * ((all_outputs - all_targets)**2).mean() + l4_regularization(action_net, 1e-3)

                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(action_net.parameters(), max_norm=5.0)
                optimizer.step()

        return            


    def train_cost_networks(self, learning_rate: float = 0.001, epoch: int = 1):
        """
        Trains the player's cost networks using memory data and regret-based loss.

        Args:
            learning_rate (float): Learning rate for cost networks. Default is 0.001.
            epoch (int): Number of training epochs. Default is 1.
        """

        # Compute expected costs
        self.my_expected_cost = self._get_current_expected_cost()
        expected_cost = torch.from_numpy(self.my_expected_cost).unsqueeze(1)
        min_expected = min(self.my_expected_cost)

        # Compute average past action if history exists
        if self.my_past_action_list:
            with torch.no_grad():
                my_ave_action = torch.tensor(self.my_past_action_list[-64:], dtype=torch.float32).mean()

                # Create a scale for exploration
                ave_scale = (self.action_space_tensor - my_ave_action) ** 4

        # Batch size for sampling
        b_size = 1 + min(31, len(self.cost_memory.memory))

        if learning_rate != 0.001:
            for optimizer in self.cost_optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
        
        # Train each cost network
        for cost_net, optimizer in zip(self.cost_networks, self.cost_optimizers):
            for ep in range(epoch):
                optimizer.zero_grad()
                
                # Retrieve cost memory samples
                memory_inputs, memory_targets = self.cost_memory.get_memory(batch_size=b_size, sampling_type='recent', alpha=0.9)
                indices = torch.argmin(torch.abs(self.action_space_tensor.T - memory_inputs), dim=1)
                selected_costs = expected_cost[indices]

                # Memory-based training
                memory_outputs = cost_net(memory_inputs)
                memory_loss = 0.3*((memory_outputs + selected_costs - memory_targets)**2).mean()

                # Regret training
                if self.my_past_action_list:
                    regret_outputs = cost_net(self.action_space_tensor)
                    
                    # Train networks to estimate best possible away from average action
                    regret_loss = (ave_scale * (regret_outputs + expected_cost - self.best_possible)**2).mean()

                    # weight assigned to control the level of exploration depending on satisfaction
                    weight = min(100*(max(min(expected_cost) - self.best_possible, 0.0))**4 , 1.0)

                    loss = weight * (memory_loss + regret_loss)
                    loss = loss + (1.0-weight)*(regret_outputs**4).mean()
                    loss  = loss + l4_regularization(cost_net, 1e-3)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(cost_net.parameters(), max_norm=5.0)
                    optimizer.step()

        return

                    
    def _get_current_expected_cost(self) -> np.ndarray:
        """
        Computes the expected cost given the current opponent action distributions.

        Returns:
            np.ndarray: Expected costs computed using opponent action distributions.
        """

        with torch.no_grad():
            opponent_action_distributions = torch.stack([
                a_n(self.action_space_tensor).squeeze(1)
                for a_n in self.action_networks
            ])

        return self.J_a_oa(opponent_action_distributions.numpy())

    
    def J_a_oa(self, o_a_matrix: np.ndarray) -> np.ndarray:
        """
        Computes expected cost for each action given opponent's action distribution.

        Args:
            o_a_matrix (np.ndarray): Opponent's action distribution matrix.

        Returns:
            np.ndarray: Expected costs for each action in self.action_space.
        """
        prob_me = np.stack([1 - self.action_space, self.action_space], axis=-1)
        prob_oppo = np.stack([1 - o_a_matrix, o_a_matrix], axis=-1)

        x_me, x_oppo = np.array([0, 1]), np.array([0, 1])
        x_me_grid, x_oppo_grid = np.meshgrid(x_me, x_oppo, indexing='ij')
        x_me_flat, x_oppo_flat = x_me_grid.ravel(), x_oppo_grid.ravel()

        cost_matrix = np.zeros((len(self.action_space), len(o_a_matrix), len(x_me_flat)))

        # Precompute costs for all (x_me, x_oppo) pairs
        for idx, (x_m, x_o) in enumerate(zip(x_me_flat, x_oppo_flat)):
            if self.i == 1:
                cost_matrix[:, :, idx] = self.cost_function(0, x_m, x_o, self.action_space[:, None], o_a_matrix, self.c)
            else:
                cost_matrix[:, :, idx] = self.cost_function(0, x_o, x_m, o_a_matrix, self.action_space[:, None], self.c)

        # Compute expected cost 
        expected_cost = np.zeros(len(self.action_space))

        for i in range(len(self.action_space)):
            for j in range(len(o_a_matrix)):  
                for idx, (x_m, x_o) in enumerate(zip(x_me_flat, x_oppo_flat)):  
                    expected_cost[i] += (
                        cost_matrix[i, j, idx]
                        * prob_me[i, x_m]  
                        * prob_oppo[j, i, x_o]  
                    )

        # Uniform distribution over cost networks
        expected_cost /= o_a_matrix.shape[0]

        return expected_cost


    
    def take_action(self) -> float:
        """
        Selects the player's next action based on expected costs.

        Returns:
            float: The selected action.
        """

        # Compute opponent action distributions
        with torch.no_grad():
            opponent_action_distributions = np.array([
                a_n(self.action_space_tensor).squeeze(1).numpy()
                for a_n in self.action_networks
            ])

        # Compute expected costs
        expected_costs = self.J_a_oa(opponent_action_distributions)

        # Compute random costs
        with torch.no_grad():
            final_costs = np.array([
                expected_costs + c_n(self.action_space_tensor).squeeze(1).numpy()
                for c_n in self.cost_networks
            ])

        # Select best actions
        best_action_indices = np.argmin(final_costs, axis=1)
        best_actions = self.action_space[best_action_indices]

        # Store action history
        self.my_current_action_distribution = np.array(best_actions)
        self.my_past_action = np.random.choice(self.my_current_action_distribution)
        self.my_past_action_list.append(self.my_past_action)

        return self.my_past_action


    

class Universe:
    def __init__(self, c1: float, b1: float, c2: float = 0.0, b2: float = 0.2):
        """
        Initializes the Universe, including players, tracking variables, and visualization setup.
        """

        # Simulation Parameters
        self.action_space = np.linspace(0.0, 1.0, 32)
        self.action_space_tensor = torch.tensor(self.action_space, dtype=torch.float32).view(-1, 1)
        self.universe_time = 0

        # Cost Tracking
        self.cumulative_cost_p1_history = []
        self.cumulative_cost_p2_history = []
        self.cumulative_cost_p1 = 0.0
        self.cumulative_cost_p2 = 0.0

        # Player Parameters
        player_params = {
            "action_space": self.action_space,
            "action_memory_size": 64,
            "cost_memory_size": 64,
            "N_cost_nets": 6,
            "N_act_nets": 6,
            "action_params": {"size": 128, "depth": 2, "p_drop": 0.0},
            "cost_params": {"size": 128, "depth": 2, "p_drop": 0.1},
        }

        # Initialize Players
        self.Player1 = Player(**player_params, cost_function=cost1, index=1, c=c1, best_possible=b1)
        self.Player2 = Player(**player_params, cost_function=cost2, index=2, c=c2, best_possible=b2)

        # Past Actions Tracking
        self.a1_past, self.a2_past = [], []
        self.X1_past, self.X2_past = [], []

        # Visualization
        self.fig, self.axes = setup_plots(self.Player1, self.Player2)

        
    def pre_train(self, player: str = "both"):
        """
        Pre-trains players by initializing their action memories with sample experiences.

        Not used in simulations, one needs to adjust and generalize this function to use.
        Args:
            player (str): Which player to pre-train. Options: "both", "player1", "player2".
                          Default is "both".
        """

        players = []
        if player == "player1":
            players = [self.Player1]
        elif player == "player2":
            players = [self.Player2]
        elif player == "both":
            players = [self.Player1, self.Player2]
        else:
            raise ValueError(f"Invalid player argument '{player}'. Choose from 'both', 'player1', or 'player2'.")

        for a in self.action_space:
            for p in players:
                p.my_past_action = a

            if a < 0.2 or a > 0.8:
                x_state = 0 if a < 0.2 else 1
                for _ in range(10):
                    for p in players:
                        p.observe_and_record(x1=x_state, x2=x_state, a1=a if p.i == 1 else 0.0, a2=a if p.i == 2 else 1.0)

                        
    def update(self):
        """
        Updates all plots dynamically during gameplay.
        """
        histories = {
            "a1": self.a1_past,
            "a2": self.a2_past,
            "X1": self.X1_past,
            "X2": self.X2_past,
            "cost_p1": self.cumulative_cost_p1_history,
            "cost_p2": self.cumulative_cost_p2_history
        }

        players = {"p1": self.Player1, "p2": self.Player2}

        update_plots(
            fig=self.fig,
            ax_dict=self.axes,
            players=players,
            action_space=self.action_space,
            action_space_tensor=self.action_space_tensor,
            histories=histories
        )

    def play_the_game(self, action_rate: float = 0.001, action_epoch: int = 8,
                      cost_rate: float = 0.001, cost_epoch: int = 8):
        """
        Plays one step of the game. This function should be called iteratively from manual_animation().

        Args:
            action_rate (float): Learning rate for action networks. Default is 0.1.
            action_epoch (int): Number of training epochs for action networks. Default is 16.
            cost_rate (float): Learning rate for cost networks. Default is 0.01.
            cost_epoch (int): Number of training epochs for cost networks. Default is 16.
        """

        self.universe_time += 1

        # Players take actions
        a1, a2 = self.Player1.take_action(), self.Player2.take_action()

        # State transitions
        X1, X2 = np.random.choice([0, 1], p=[1 - a1, a1]), np.random.choice([0, 1], p=[1 - a2, a2])
        
        # Compute costs for players
        cost_p1 = self.Player1.cost_function(0, X1, X2, a1, a2, self.Player1.c)
        cost_p2 = self.Player2.cost_function(0, X1, X2, a1, a2, self.Player2.c)

        # Track cumulative costs
        self.cumulative_cost_p1 += cost_p1
        self.cumulative_cost_p2 += cost_p2

        self.cumulative_cost_p1_history.append(self.cumulative_cost_p1)
        self.cumulative_cost_p2_history.append(self.cumulative_cost_p2)

        # Record observations
        self.Player1.observe_and_record(x1=X1, x2=X2, a1=a1, a2=None)
        self.Player2.observe_and_record(x1=X1, x2=X2, a1=None, a2=a2)

        # Train both players
        self.Player1.train(action_rate=action_rate, action_epoch=action_epoch, 
                           cost_rate=cost_rate, cost_epoch=cost_epoch)
        self.Player2.train(action_rate=action_rate, action_epoch=action_epoch, 
                           cost_rate=cost_rate, cost_epoch=cost_epoch)

        # Track past actions and states
        self.a1_past.append(a1)
        self.a2_past.append(a2)
        self.X1_past.append(X1)
        self.X2_past.append(X2)

