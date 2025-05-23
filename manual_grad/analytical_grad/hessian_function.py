import numpy as np
from numba import njit
from typing import List
from ..basic_function import discount_factor, bond_valuation
from .gradient_function import bond_valuation_gradient_analytical

@njit(cache=True)
def discount_factor_hessian_analytical(
    params: np.ndarray, t: np.ndarray, elements: List[int]
) -> np.ndarray:
    """Calculate hessian of discount factor"""
    f0, f1, f2, gamma = params
    sorted_elements = sorted(elements)
    if sorted_elements == [0, 0]:
        return t**2 * discount_factor(params, t)
    elif sorted_elements == [0, 1]:
        return -t * (-gamma + np.exp(-t / gamma) * gamma) * discount_factor(params, t)
    elif sorted_elements == [0, 2]:
        return (
            -t
            * (-gamma + np.exp(-t / gamma) * (t + gamma))
            * discount_factor(params, t)
        )
    elif sorted_elements == [0, 3]:
        return (
            -t
            * (
                -f1 * (1 - np.exp(-t / gamma) - t * np.exp(-t / gamma) / gamma)
                - f2
                * (
                    1
                    - np.exp(-t / gamma)
                    - t * np.exp(-t / gamma) * (t + gamma) / (gamma**2)
                )
            )
            * discount_factor(params, t)
        )
    elif sorted_elements == [1, 1]:
        return ((-gamma + np.exp(-t / gamma) * gamma) ** 2) * discount_factor(params, t)
    elif sorted_elements == [1, 2]:
        return (
            (-gamma + np.exp(-t / gamma) * gamma)
            * (-gamma + np.exp(-t / gamma) * (t + gamma))
            * discount_factor(params, t)
        )
    elif sorted_elements == [1, 3]:
        return (
            (-1 + np.exp(-t / gamma) + np.exp(-t / gamma) * t / gamma) * (f1 + 1)
            - f2
            * (
                1
                - np.exp(-t / gamma)
                - t * (t + gamma) * np.exp(-t / gamma) / (gamma**2)
            )
        ) * discount_factor(params, t)
    elif sorted_elements == [2, 2]:
        return ((-gamma + np.exp(-t / gamma) * (t + gamma)) ** 2) * discount_factor(
            params, t
        )
    elif sorted_elements == [2, 3]:
        return (
            (-1 + np.exp(-t / gamma) + np.exp(-t / gamma) * t / gamma)
            * (1 + f1 * (-gamma + np.exp(-t / gamma) * (t + gamma)))
            - (-gamma + np.exp(-t / gamma) * (t + gamma))
            * (
                1
                - np.exp(-t / gamma)
                - np.exp(-t / gamma) * t * (t + gamma) / (gamma**2)
            )
            * f2
        ) * discount_factor(params, t)
    else:
        return (
            np.exp(-t / gamma) * (t**2) * f1 / (gamma**3)
            - (
                -2 * np.exp(-t / gamma) * t / (gamma**2)
                - np.exp(-t / gamma) * (t**2) * (t + gamma) / (gamma**4)
                + 2 * np.exp(-t / gamma) * t
                + (t + gamma) / (gamma**3)
            )
            * f2
            + (
                f1 * (1 - np.exp(-t / gamma) - np.exp(-t / gamma) * t / gamma)
                + f2
                * (
                    1
                    - np.exp(-t / gamma)
                    - np.exp(-t / gamma) * t * (t + gamma) / (gamma**2)
                )
            )
            ** 2
        ) * discount_factor(params, t)


@njit(cache=True)
def bond_valuation_hessian_analytical(
    params: np.ndarray, t: np.ndarray, coupon: float, elements: List[int]
) -> float:
    """Calculate hessian of bond valuation"""
    cf = np.ones_like(t) * coupon / 2
    cf[-1] += 1
    return 100 * np.sum(cf * discount_factor_hessian_analytical(params, t, elements))


@njit(cache=True)
def loss_function_component_hessian_analytical(
    params: np.ndarray,
    t: np.ndarray,
    coupon: float,
    bid: float,
    ask: float,
    elements: List[int],
) -> float:
    """Calculate hessian of loss component"""
    vj = bond_valuation(params, t, coupon)
    if vj > ask:
        return (2 * (vj - ask) / (ask**2)) * bond_valuation_hessian_analytical(
            params, t, coupon, elements
        ) + (2 / (ask**2)) * bond_valuation_gradient_analytical(
            params, t, coupon, elements[0]
        ) * bond_valuation_gradient_analytical(
            params, t, coupon, elements[1]
        )
    elif vj < bid:
        return (2 * (vj - bid) / (bid**2)) * bond_valuation_hessian_analytical(
            params, t, coupon, elements
        ) + (2 / (bid**2)) * bond_valuation_gradient_analytical(
            params, t, coupon, elements[0]
        ) * bond_valuation_gradient_analytical(
            params, t, coupon, elements[1]
        )
    else:
        return 0.0
