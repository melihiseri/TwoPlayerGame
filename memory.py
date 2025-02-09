import collections
import torch
import numpy as np
from .utils import sample_indices

class Memory:
    """
    A memory buffer that stores (space, observation) pairs and provides sampling methods.
    """

    def __init__(self, memory_size: int):
        """
        Initializes the memory buffer with a fixed maximum size.

        Args:
            memory_size (int): Maximum number of entries the memory can store.
        """
        self.memory_size = memory_size
        self.memory = collections.deque(maxlen=memory_size)

    def remember(self, space: float, observation: float):
        """
        Stores a (space, observation) pair in memory.

        Args:
            space (float): The space value.
            observation (float or np.ndarray): The observation value.
        """
        if isinstance(observation, np.ndarray):
            observation = observation.item()  # Convert single-value NumPy arrays to scalars
        
        self.memory.append((space, observation))

    def get_memory(
            self,
            batch_size: int,
            sampling_type: str,
            alpha: float, 
            add_randomness: bool = False,
            noise_level: float = 1e-6,
            exp_start: float = 0.005
    ):
        """
        Retrieves a batch of samples from memory.

        Args:
            batch_size (int): Number of samples to retrieve.
            sampling_type (str): Sampling strategy.
            alpha (float): Sampling parameter.
            add_randomness (bool, optional): Whether to add Gaussian noise to samples. Default is False.
            noise_level (float, optional): Standard deviation of added noise. Default is 1e-6.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled (space, observation) pairs.
        """

        if len(self.memory) == 0:
            raise ValueError(f"Not enough memory to sample.")

        all_spaces, all_observations = zip(*self.memory)
        all_spaces = torch.tensor(all_spaces, dtype=torch.float32)
        all_observations = torch.tensor(all_observations, dtype=torch.float32)

        memory_indices = sample_indices(memory_length=len(self.memory), batch_size=batch_size,
                                        sampling_type=sampling_type, alpha=alpha, exp_start=exp_start)

        if add_randomness:
            space_samples = torch.normal(mean=all_spaces[memory_indices], std=noise_level).unsqueeze(1)
        else:
            space_samples = all_spaces[memory_indices].unsqueeze(1)

        observation_samples = all_observations[memory_indices].unsqueeze(1)
        return space_samples, observation_samples

    def get_raw_memory(self, batch_size: int, sampling_type: str, alpha: float):
        """
        Retrieves raw memory indices and corresponding stored (space, observation) pairs.

        [Not used in this project]
        Args:
            batch_size (int): Number of samples to retrieve.
            sampling_type (str): Sampling strategy.
            alpha (float): Sampling parameter.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Sampled indices, spaces, and observations.
        """

        if len(self.memory) < batch_size:
            raise ValueError(f"Not enough memory to sample {batch_size} elements (only {len(self.memory)} stored).")

        all_spaces, all_observations = zip(*self.memory)
        all_spaces = torch.tensor(all_spaces, dtype=torch.float32)
        all_observations = torch.tensor(all_observations, dtype=torch.float32)

        memory_indices = sample_indices(len(self.memory), batch_size, sampling_type, alpha)
        return memory_indices, all_spaces, all_observations
    
    def view_memory(self):
        """
        Prints the contents of the memory buffer.
        """
        print("Memory contents:")
        for i, (space, observation) in enumerate(self.memory):
            print(f"Entry {i}:")
            print(f"  Space: {space}")
            print(f"  Observation: {observation}")

    def clear(self):
        """
        Clears the memory buffer.
        """
        self.memory = collections.deque(maxlen=self.memory_size)
