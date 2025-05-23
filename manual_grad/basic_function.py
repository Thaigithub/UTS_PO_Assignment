import numpy as np
from numba import njit

@njit(cache=True)
def discount_factor(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calculate discount factor"""
    f0 = params[0]
    f1 = params[1]
    f2 = params[2]
    gamma = params[3]
    if gamma == 0:
        return np.exp(-f0 * t)
    if gamma < 0:
        raise ValueError("Gamma must be positive")
    res = -(
        f0 * t
        + f1 * (gamma - np.exp(-t / gamma) * gamma)
        + f2 * (gamma - np.exp(-t / gamma) * (t + gamma))
    )
    return np.exp(res)


@njit(cache=True)
def bond_valuation(params: np.ndarray, t: np.ndarray, coupon: float) -> float:
    """Calculate bond valuation"""
    cf = np.ones_like(t) * coupon / 2
    cf[-1] += 1
    return 100 * np.sum(cf * discount_factor(params, t))


@njit(cache=True)
def loss_function_component(
    params: np.ndarray, t: np.ndarray, coupon: float, bid: float, ask: float
) -> float:
    """Objective function for optimization"""
    bond_price = bond_valuation(params, t, coupon)
    if np.abs(bond_price) >= 1e154:
        return np.inf
    return (max(0, bond_price - ask) / ask) ** 2 + (max(0, bid - bond_price) / bid) ** 2