import numpy as np
from numba import njit
from ..basic_function import loss_function_component


@njit(cache=True)
def loss_function_component_gradient_numerical(
    params: np.ndarray,
    t: np.ndarray,
    coupon: float,
    bid: float,
    ask: float,
    element: int,
    tolerance: float = 1e-15,
) -> float:
    """Calculate gradient of bond valuation"""
    param_forward = params.copy()
    param_forward[element] += tolerance
    param_backward = params.copy()
    param_backward[element] -= tolerance
    lj_forward = loss_function_component(param_forward, t, coupon, bid, ask)
    lj_backward = loss_function_component(param_backward, t, coupon, bid, ask)
    return (lj_forward - lj_backward) / (2 * tolerance)
