from typing import List
import numpy as np

class OptimizeResult:
    """
    A class to hold the result of an optimization process.

    Attributes:
        x (array-like): The solution of the optimization.
        fun (float): The value of the objective function at the solution.
        success (bool): Whether the optimization was successful.
        message (str): A message describing the outcome of the optimization.
    """

    def __init__(self, x: np.ndarray, fun: float, success: bool, message: str, nit: int, values: List[float], path: List[np.ndarray]):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        self.nit = nit
        self.values = values
        self.path = path