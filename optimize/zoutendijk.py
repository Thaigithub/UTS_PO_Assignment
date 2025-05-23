import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize, OptimizeResult, linprog


def zoutendijk(
    fun,
    x0,
    jac,
    callback,
    options: dict,
):
    A = np.array([[0, 0, 0, -1]])
    b = np.array([0])
    maxiter = options.get("maxiter", 100)
    tol = options.get("gtol", 1.0e-8)
    nit = 1
    nfev = 1
    njev = 1
    x_k = x0.copy()
    f_k = fun(x_k)
    g_k = jac(x_k)
    callback(x_k)
    res = OptimizeResult()
    if la.norm(g_k) < tol:
        res.x = x_k
        res.success = True
        res.status = 0
        res.message = "Norm of gradient is within tolerence"
        res.fun = f_k
        res.nit = 0
        res.nfev = nfev
        res.njev = njev
        res.jac = g_k
        return res
    while True:
        active_index = np.where(np.abs(A @ x - b) <= tol)[0]
        A_active = A[active_index, :]
        inactive_index = np.setdiff1d(np.arange(A.shape[0]), active_index)
        A_non_active = A[inactive_index, :]
        b_non_active = b[inactive_index]
        d_k = np.array(
            linprog(
                g_k,
                A_ub=A_active,
                b_ub=np.zeros(len(A_active)),
                bounds=[(-1, 1), (-1, 1)],
                method="highs",
            ).x
        )
        if np.abs(g_k.T @ d_k) <= tol:
            res.x = x_k
            res.success = True
            res.status = 0
            res.message = "Dot product of gradient and improving direction is within tolerence"
            res.fun = f_k
            res.nfev = nfev
            res.njev = njev
            res.nit = nit
            return res
        RHS = b_non_active - A_non_active @ x
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
        ls = lambda alpha: fun(x_k + alpha[0] * d_k)
        lsr = minimize(ls, [1], method="L-BFGS-B", bounds=[(0, a_max)])
        alpha_k = lsr.x[0]
        nfev += lsr.nfev
        njev += lsr.njev
        x_k1 = x_k + alpha_k * d_k
        g_k1 = jac(x_k1)
        f_k1 = fun(x_k1)
        njev += 1
        nfev += 1
        callback(x_k1)
        if abs(alpha_k * la.norm(d_k)) < tol:
            res.x = x_k1
            res.success = True
            res.status = 0
            res.message = "Change of x is within tolerence"
            res.fun = f_k1
            res.nit = nit
            res.nfev = nfev
            res.njev = njev
            res.jac = g_k1
            return res
        if abs(f_k - f_k1) < tol:
            res.x = x_k1
            res.success = True
            res.status = 0
            res.message = "Change of fun is within tolerence"
            res.fun = f_k1
            res.nfev = nfev
            res.njev = njev
            res.nit = nit
            return res
        if la.norm(g_k1) < tol:
            res.x = x_k1
            res.success = True
            res.status = 0
            res.message = "Norm of gradient is within tolerence"
            res.fun = f_k1
            res.nfev = nfev
            res.njev = njev
            res.nit = nit
            return res
        if nit > maxiter:
            res.x = x_k1
            res.success = False
            res.status = 0
            res.message = "Max iter reached"
            res.fun = f_k1
            res.nfev = nfev
            res.njev = njev
            res.nit = nit
            return res

        x_k = x_k1
        f_k = f_k1
        g_k = g_k1
        nit += 1
