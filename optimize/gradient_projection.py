import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize
from .result import OptimizeResult


def gradient_projection(
    fun,
    x0: np.ndarray,
    jac,
    A: np.ndarray,
    b: np.ndarray,
    maxiter=100,
    amax=1000.0,
    tol=1.0e-8,
):
    x_eps = tol  # tolerence for convergence on delta x
    f_eps = tol  # tolerence for convergence on delta f
    g_eps = tol  # tolerence for convergence on norm of gradient
    nit = 1
    x_k = x0.copy()
    f_k = fun(x_k)
    g_k = jac(x_k)
    values = []
    path = []
    if la.norm(g_k) < g_eps:
        return OptimizeResult(
            x_k, f_k, True, "norm of gradient is within tolerence", 0, [f_k], [x_k]
        )
    while True:
        path.append(x_k)
        values.append(f_k)
        active_index = np.where(np.abs(A @ x_k - b) <= tol)[0]
        A_active = A[active_index, :]
        inactive_index = np.setdiff1d(np.arange(A.shape[0]), active_index)
        A_non_active = A[inactive_index, :]
        b_non_active = b[inactive_index]
        M = A_active
        d_k = None
        while True:
            if len(M) == 0:
                if la.norm(g_k) < g_eps:
                    return OptimizeResult(
                        x_k,
                        f_k,
                        True,
                        "norm of projected gradient is within tolerence",
                        nit,
                        values,
                        path,
                    )
                else:
                    d_k = -g_k
                    break
            else:
                d_k = -(np.eye(M.shape[1]) - M.T @ la.inv(M @ M.T) @ M) @ g_k
                if la.norm(d_k) < g_eps:
                    d_k = np.zeros(len(d_k))
                    laragian = -la.inv(M @ M.T) @ M @ g_k
                    muy = laragian
                    if np.all(muy >= 0):
                        return OptimizeResult(
                            x_k,
                            f_k,
                            True,
                            "Optimization terminated due to muy >= 0",
                            nit,
                            values,
                            path,
                        )
                    else:
                        remove_index = np.where(muy < 0)[0][0]
                        M = np.delete(M, remove_index, axis=0)
                else:
                    break

        RHS = b_non_active - A_non_active @ x_k
        LHS = A_non_active @ d_k
        bounds = [0, np.inf]
        for i in range(len(LHS)):
            if LHS[i] > 0:
                bc = RHS[i] / LHS[i]
                if bc < bounds[1]:
                    bounds[1] = bc
            elif LHS[i] < 0:
                bc = RHS[i] / LHS[i]
                if bc > bounds[0]:
                    bounds[0] = bc
        a_max = max(bounds[0], bounds[1], 0)
        bounds = [0, a_max]

        # def extra_condition(alpha, x, f, g):
        #     if alpha >= bounds[0] and alpha <= bounds[1]:
        #         return True
        #     else:
        #         return False

        # alpha, _, _, _, _, success = ls(
        #     fun, jac, x_k, d_k, amax=amax, extra_condition=extra_condition
        # )
        ls = lambda alpha: fun(x_k + alpha[0] * d_k)
        alpha = minimize(ls, 0, bounds=[(bounds[0], bounds[1])], method="L-BFGS-B").x[0]
        if alpha < bounds[0]:
            alpha = bounds[0]
        elif alpha > bounds[1]:
            alpha = bounds[1]
        if abs(alpha * la.norm(d_k)) < x_eps:
            return OptimizeResult(
                x_k, f_k, True, "change of x is within tolerence", nit, values, path
            )
        x_k1 = x_k + alpha * d_k
        if abs(fun(x_k1) - fun(x_k)) < f_eps:
            return OptimizeResult(
                x_k, f_k, True, "change of fun is within tolerence", nit, values, path
            )
        x_k = x_k1
        f_k = fun(x_k)
        g_k = jac(x_k)
        path.append(x_k)
        values.append(f_k)
        nit += 1
        if nit > maxiter:
            return OptimizeResult(
                x_k, f_k, False, "Maximum iterations reached", nit, values, path
            )
