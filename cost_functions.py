import numpy as np

def cost1(t: float, x1: np.ndarray, x2: np.ndarray, a1: np.ndarray, a2: np.ndarray, c: float) -> np.ndarray:
    """
    Computes the cost for Player 1.

    Args:
        t (float): Time variable (not used in the function but included for consistency).
        x1 (np.ndarray or float): Player 1's state.
        x2 (np.ndarray or float): Player 2's state.
        a1 (np.ndarray or float): Player 1's action.
        a2 (np.ndarray or float): Player 2's action.
        c (float): Game parameter.

    Returns:
        np.ndarray: Computed cost for Player 1.
    """
    x1, x2, a1, a2 = map(np.asarray, (x1, x2, a1, a2))
    return np.where(x2 == 1, 1.0 - c * a1, -c * a1)


def cost2(t: float, x1: np.ndarray, x2: np.ndarray, a1: np.ndarray, a2: np.ndarray, c: float) -> np.ndarray:
    """
    Computes the cost for Player 2.

    Args:
        t (float): Time variable (not used in the function but included for consistency).
        x1 (np.ndarray or float): Player 1's state.
        x2 (np.ndarray or float): Player 2's state.
        a1 (np.ndarray or float): Player 1's action.
        a2 (np.ndarray or float): Player 2's action.
        c (float): Game parameter.

    Returns:
        np.ndarray: Computed cost for Player 2.
    """
    x1, x2, a1, a2 = map(np.asarray, (x1, x2, a1, a2))
    return np.where(x1 == x2, -c * a2, 1.0)
