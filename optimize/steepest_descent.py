import numpy as np
import numpy.linalg as la
from scipy.optimize import line_search
from .result import OptimizeResult

def steepest_descent(
    fun, x0: np.ndarray, jac, ls=line_search, maxiter=100, amax=1000.0, tol=1.0e-8, optional_line_search=None
):
    x_eps = tol  # tolerence for convergence on delta x
    f_eps = tol  # tolerence for convergence on delta f
    g_eps = tol  # tolerence for convergence on norm of gradient
    x_k = x0.copy()
    nit = 1
    f_k = fun(x_k)
    d_k = -jac(x_k)
    values = []
    path = []
    if la.norm(jac(x_k)) < g_eps:
        return OptimizeResult(x_k, f_k, True, "norm of gradient is within tolerence", 0, [f_k], [x_k])
    while True:
        path.append(x_k.copy())
        values.append(f_k)
        alpha_k, _, _, _, _, success = ls(fun, jac, x_k, d_k, amax=amax)
        if success is None:
            if optional_line_search is not None:
                alpha_k = optional_line_search(fun, jac, x_k, d_k, amax=amax)
            else:
                raise Exception("Line search failed, change line search method")
        if abs(alpha_k * la.norm(jac(x_k))) < x_eps:
            return OptimizeResult(x_k, f_k, True, "change of x is within tolerence", nit, values, path)
        x_k1 = x_k + alpha_k * d_k
        d_k1 = -jac(x_k1)
        if abs(f_k - fun(x_k1)) < f_eps:
            return OptimizeResult(x_k, f_k, True, "change of fun is within tolerence", nit, values, path)
        if la.norm(jac(x_k1)) < g_eps:
            return OptimizeResult(x_k, f_k, True, "norm of gradient is within tolerence", nit, values, path)
        if nit > maxiter:
            return OptimizeResult(x_k, f_k, False, "Max iter reached", nit, values, path)
        nit += 1
        x_k = x_k1
        f_k = fun(x_k1)
        d_k = d_k1
