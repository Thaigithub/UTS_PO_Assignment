import numpy as np
import numpy.linalg as la
from scipy.optimize import line_search
from .result import OptimizeResult

def symetry(A: np.ndarray):
    """
    Check if the matrix A is positive definite.
    """
    if len(A.shape) != 2:
        raise False
    if A.shape[0] != A.shape[1]:
        raise False
    return np.allclose(A, A.T)


def positive_definite(A):
    """
    Check if the matrix A is positive definite.
    """
    return np.all(la.eigvals(A) > 0)


def davidon_fletcher_powell(
    fun,
    x0: np.ndarray,
    jac,
    H1: np.ndarray,
    ls=line_search,
    maxiter=100,
    amax=1000.0,
    tol=1.0e-8,
):
    if not symetry(H1) or not positive_definite(H1):
        raise ValueError("H1 must be a symetric positive definite matrix")
    x_eps = tol  # tolerence for convergence on delta x
    f_eps = tol  # tolerence for convergence on delta f
    g_eps = tol  # tolerence for convergence on norm of gradient
    x_k = x0.copy()
    nit = 1
    f_k = fun(x_k)
    H_k = H1.copy()
    d_k = -H_k @ jac(x_k)
    if la.norm(jac(x_k)) < g_eps:
        return OptimizeResult(x_k, f_k, True, "norm of gradient is within tolerence", 0)
    while True:
        alpha_k, _, _, _, _, success = ls(fun, jac, x_k, d_k, amax=amax)
        if success is None:
            raise Exception("Line search failed, change line search method")
        if abs(alpha_k * la.norm(d_k)) < x_eps:
            return OptimizeResult(
                x_k, f_k, True, "change of x is within tolerence", nit
            )
        x_k1 = x_k + alpha_k * d_k
        p_k = alpha_k * d_k
        q_k = jac(x_k1) - jac(x_k)
        H_k1 = (
            H_k
            + np.outer(p_k, p_k) / (p_k.T @ q_k)
            - np.outer(H_k @ q_k, H_k @ q_k) / (q_k.T @ H_k @ q_k)
        )
        d_k1 = -H_k1 @ jac(x_k1)
        if abs(f_k - fun(x_k1)) < f_eps:
            return OptimizeResult(
                x_k, f_k, True, "change of fun is within tolerence", nit
            )
        if la.norm(jac(x_k1)) < g_eps:
            return OptimizeResult(
                x_k, f_k, True, "norm of gradient is within tolerence", nit
            )
        if nit > maxiter:
            return OptimizeResult(x_k, f_k, False, "Max iter reached", nit)
        nit += 1
        x_k = x_k1
        f_k = fun(x_k1)
        d_k = d_k1
        H_k = H_k1
