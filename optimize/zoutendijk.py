def zoutendijk(
    fun,
    x0: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    W: np.ndarray = None,
    w: np.ndarray = None,
    maxiter=100,
    tol=1.0e-8,
):
    jac = grad(fun)
    nit = 1
    x = x0.copy()
    n = len(x0)
    while True:
        if nit > maxiter:
            print("Max iter reached")
            break
        print("______________________________")
        print("ITERATION:", nit)
        print("-----DIRECTION SEARCH-----")
        gra = jac(x)
        print("grad =")
        printer(gra)
        active_index = np.where(np.abs(A @ x - b) <= tol)[0]
        print("active_index =")
        printer(active_index + 1)
        A_active = A[active_index, :]
        print("A_active =")
        printer(A_active)
        b_active = b[active_index]
        print("b_active =")
        printer(b_active)
        inactive_index = np.setdiff1d(np.arange(A.shape[0]), active_index)
        A_non_active = A[inactive_index, :]
        print("A_non_active =")
        printer(A_non_active)
        b_non_active = b[inactive_index]
        print("b_non_active =")
        printer(b_non_active)
        dk = np.array(
            linprog(
                gra,
                A_ub=A_active,
                b_ub=np.zeros(len(A_active)),
                bounds=[(-1, 1), (-1, 1)],
                method="highs",
            ).x
        )
        print("dk =")
        printer(dk)
        if np.abs(gra.T @ dk) <= tol:
            print("Optimization terminated due to grad * dk = 0")
            break
        print("-----LINE SEARCH-----")
        RHS = b_non_active - A_non_active @ x
        print("RHS =")
        printer(RHS)
        LHS = A_non_active @ dk
        print("LHS =")
        printer(LHS)
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
        print("a_max =", str(Fraction(a_max).limit_denominator()))
        bounds = [0, a_max]
        print("bounds =")
        printer(np.array(bounds))
        ls = lambda alpha: fun(x + alpha * dk)
        gs = grad(ls)
        a = fsolve(gs, 1)[0]
        if a < bounds[0]:
            a = bounds[0]
        elif a > bounds[1]:
            a = bounds[1]

        if nit == 2:
            a = 5
        print("alpha =", str(Fraction(a).limit_denominator()))
        x = x + a * dk
        print("x =")
        printer(x)
        nit += 1
