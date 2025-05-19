import numpy as np
import numpy.linalg as la
from .result import OptimizeResult


def newton(
    fun,
    x0: np.ndarray,
    jac,
    hessian,
    maxiter=100,
    tol=1.0e-8,
):
    f_eps = tol  # tolerence for convergence on delta f
    g_eps = tol  # tolerence for convergence on norm of gradient
    x_k = x0.copy()
    nit = 1
    f_k = fun(x_k)
    d_k = -la.inv(hessian(x_k)) @ jac(x_k)
    if la.norm(jac(x_k)) < g_eps:
        return OptimizeResult(x_k, f_k, True, "norm of gradient is within tolerence", 0)
    while True:
        x_k1 = x_k + d_k
        d_k1 = -la.inv(hessian(x_k)) @ jac(x_k1)
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
