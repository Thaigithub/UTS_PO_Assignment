import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize, OptimizeResult


def davidon_fletcher_powell(
    fun,
    x0,
    jac,
    callback,
    options: dict,
):

    maxiter = options.get("maxiter", 100)
    tol = options.get("gtol", 1.0e-8)
    nit = 1
    nfev = 1
    njev = 1
    x_k = x0.copy()
    f_k = fun(x_k)
    g_k = jac(x_k)
    H_k = np.eye(len(x0))
    d_k = -H_k @ g_k
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
        ls = lambda alpha: fun(x_k + alpha[0] * d_k)
        lsr = minimize(ls, [1], method="L-BFGS-B", bounds=[(0, None)])
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
        p_k = alpha_k * d_k
        q_k = g_k1 - g_k
        H_k1 = (
            H_k
            + np.outer(p_k, p_k) / (p_k.T @ q_k)
            - np.outer(H_k @ q_k, H_k @ q_k) / (q_k.T @ H_k @ q_k)
        )
        d_k1 = -H_k1 @ g_k1
        nit += 1
        x_k = x_k1
        f_k = f_k1
        g_k = g_k1
        d_k = d_k1
        H_k = H_k1
