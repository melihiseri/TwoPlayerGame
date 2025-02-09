import torch
import torch.nn as nn


class ActionEstimator(nn.Module):
    """
    Neural network for estimating actions.
    Includes dropout in hidden layers.
    """
    def __init__(self, input_dim: int, hidden_size: int = 10, output_dim: int = 10, depth: int = 4, p_drop: float = 0.3, negative_slope: float = 0.01):
        super(ActionEstimator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)])
        self.fc_prelast = nn.Linear(hidden_size, hidden_size)
        self.fc_last = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(p=p_drop)
        self.negative_slope = negative_slope  # Leaky ReLU slope


    def forward(self, x, scale: float = 1.5): 
        x = 4.0 * (x - 0.5)  # [0,1] -> [-2,2]
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
        for layer in self.hidden_layers:
            x = self.dropout(layer(x))
            x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        x = self.dropout(self.fc_prelast(x))
        x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        x = self.fc_last(x)

        return 0.5 + 0.7 * torch.tanh(x * scale)


class CostEstimator(nn.Module):
    """
    Neural network for estimating costs.
    Includes dropout in hidden layers.
    """
    def __init__(self, input_dim: int, hidden_size: int = 10, output_dim: int = 10, depth: int = 4, p_drop: float = 0.3, negative_slope: float = 0.01):
        super(CostEstimator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)])
        self.fc_prelast = nn.Linear(hidden_size, hidden_size)
        self.fc_last = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(p=p_drop)
        self.negative_slope = negative_slope  # Leaky ReLU slope

        
    def forward(self, x, scale: float = 1.5):
        x = 4.0 * (x - 0.5)  # [0,1] -> [-2,2]
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
        for layer in self.hidden_layers:
            x = self.dropout(layer(x)) 
            x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        x = self.dropout(self.fc_prelast(x)) 
        x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        x = self.fc_last(x)

        return 2 * torch.tanh(x * scale)
