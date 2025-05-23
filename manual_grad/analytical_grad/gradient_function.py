import numpy as np
from numba import njit
from ..basic_function import discount_factor, bond_valuation

@njit(cache=True)
def discount_factor_gradient_analytical(
    params: np.ndarray, t: np.ndarray, element: int
) -> np.ndarray:
    """Calculate gradient of discount factor"""
    f0, f1, f2, gamma = params
    if element == 0:
        return -t * discount_factor(params, t)
    elif element == 1:
        return (-gamma + np.exp(-t / gamma) * gamma) * discount_factor(params, t)
    elif element == 2:
        return (-gamma + np.exp(-t / gamma) * (t + gamma)) * discount_factor(params, t)
    else:
        return (
            -f1 * (1 - np.exp(-t / gamma) - t * np.exp(-t / gamma) / gamma)
            - f2
            * (
                1
                - np.exp(-t / gamma)
                - t * np.exp(-t / gamma) * (t + gamma) / (gamma**2)
            )
        ) * discount_factor(params, t)


@njit(cache=True)
def bond_valuation_gradient_analytical(
    params: np.ndarray, t: np.ndarray, coupon: float, element: int
) -> float:
    """Calculate gradient of bond valuation"""
    cf = np.ones_like(t) * coupon / 2
    cf[-1] += 1
    return 100 * np.sum(cf * discount_factor_gradient_analytical(params, t, element))


@njit(cache=True)
def loss_function_component_gradient_analytical(
    params: np.ndarray,
    t: np.ndarray,
    coupon: float,
    bid: float,
    ask: float,
    element: int,
) -> float:
    """Calculate gradient of bond valuation"""
    vj = bond_valuation(params, t, coupon)
    if vj > ask:
        return (2 * (vj - ask) / (ask**2)) * bond_valuation_gradient_analytical(
            params, t, coupon, element
        )
    elif vj < bid:
        return (2 * (vj - bid) / (bid**2)) * bond_valuation_gradient_analytical(
            params, t, coupon, element
        )
    else:
        return 0.0
