import numpy as np
from numba import njit
from typing import List
from ..basic_function import loss_function_component

@njit(cache=True)
def loss_function_component_hessian_numerical(
    params: np.ndarray,
    t: np.ndarray,
    coupon: float,
    bid: float,
    ask: float,
    elements: List[int],
    tolerance: float = 1e-15,
) -> float:
    """Calculate hessian of loss component"""
    new_elements = elements.copy()
    elements = sorted(new_elements)
    param_f_f = params.copy()
    param_f_b = params.copy()
    param_b_f = params.copy()
    param_b_b = params.copy()
    param_f_f[elements[0]] += tolerance
    param_f_f[elements[1]] += tolerance
    param_f_b[elements[0]] += tolerance
    param_f_b[elements[1]] -= tolerance
    param_b_f[elements[0]] -= tolerance
    param_b_f[elements[1]] += tolerance
    param_b_b[elements[0]] -= tolerance
    param_b_b[elements[1]] -= tolerance
    lj_f_f = loss_function_component(param_f_f, t, coupon, bid, ask)
    lj_f_b = loss_function_component(param_f_b, t, coupon, bid, ask)
    lj_b_f = loss_function_component(param_b_f, t, coupon, bid, ask)
    lj_b_b = loss_function_component(param_b_b, t, coupon, bid, ask)
    return (lj_f_f - lj_f_b - lj_b_f + lj_b_b) / (4 * tolerance**2)
