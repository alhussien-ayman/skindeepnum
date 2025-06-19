import numpy as np
import time
from simulation_service import get_time_output,complete_solutions_if_needed,solve_tridiagonal_system
from constants import U0_NORMAL_CELL_DENSITY, N_TERMS_ANALYTICAL



def solve_finite_difference_implicit(params):
    """
    Solves the PDE using the fully implicit Backward Euler method.
    For non-linear cases, it uses Picard iteration at each time step
    to solve the resulting non-linear system.
    """
    start_time = time.time()
    D, sc, p_nonlin, r0 = params['D'], params['sc'], params['p'], params['r0']
    tf = params['days'] * 24 * 3600
    u0 = U0_NORMAL_CELL_DENSITY
    time_outputs = get_time_output(params['days'])
    t_output_s, t_output_labels = time_outputs['seconds'], time_outputs['labels']

    Nr = 101
    dr = r0 / (Nr - 1)
    r_coords = np.linspace(0, r0, Nr)
    Nt = min(5000, max(1000, int(tf / 100)))
    dt = tf / Nt
    N_internal = Nr - 1

    u = np.zeros(Nr)
    u[0] = u0
    solutions = []
    next_output_idx = 0

    max_iter = 30  # Max Picard iterations
    tol = 1e-7  # Picard convergence tolerance
    eps = 1e-9  # Small epsilon for stability

    for n in range(Nt):
        u_old = u.copy()
        u_iter = u_old.copy()  # Initial guess for Picard iteration

        for k in range(max_iter):
            u_prev_iter = u_iter.copy()

            # --- Coefficients based on the previous iteration's solution ---
            if p_nonlin == 0:
                D_coeff = D * np.ones(N_internal)
            else:
                D_coeff = D * np.maximum(eps, 1 - u_iter[1:] / u0) ** p_nonlin

            lambda_val = D_coeff * dt / (dr ** 2)

            # --- Assemble the tridiagonal system Ax = d for u_new ---
            # The system is A * u_new = d, where A is based on u_iter
            # and d includes terms from u_old and a linearized source term.
            # Equation: u_new - dt*L(u_new) = u_old + dt*S(u_iter)
            b_main = 1 + 2 * lambda_val
            a_sub = -lambda_val.copy()[1:]  # Standard sub-diagonal
            c_super = -lambda_val.copy()[:-1]  # Standard super-diagonal

            # Right-hand side includes the explicit part
            source_term = sc * u_iter[1:] * (1 - u_iter[1:] / u0)
            d_rhs = u_old[1:] + dt * source_term

            # Apply Dirichlet BC at r=0
            d_rhs[0] += lambda_val[0] * u0

            # --- Correctly apply Neumann BC at r=r0 ---
            # The discrete Laplacian with a ghost point u_{M+1}=u_{M-1} is 2*(u_{M-1}-u_M)/dr^2
            # This changes the last row of the matrix.
            # The equation for the last node (i=N_internal-1) becomes:
            # -2*lambda*u_{i-1} + (1+2*lambda)*u_i = d_i
            # This means the last element of the sub-diagonal 'a' must be -2*lambda.
            a_sub[-1] = -2 * lambda_val[-1]

            # Solve the linear system for this iteration
            u_internal_new = solve_tridiagonal_system(a_sub, b_main, c_super, d_rhs)

            if np.isnan(u_internal_new).any():
                u_iter.fill(np.nan)
                break

            u_iter[1:] = u_internal_new

            # Check for convergence of the Picard iteration
            if np.linalg.norm(u_iter - u_prev_iter, np.inf) < tol:
                break

        # Update solution for the next time step
        u = u_iter.copy()
        if np.isnan(u).any():
            break  # Stop simulation if it has failed

        np.clip(u, 0, u0, out=u)

        current_time = (n + 1) * dt
        if next_output_idx < len(t_output_s) and current_time >= t_output_s[next_output_idx]:
            solutions.append(u.copy())
            next_output_idx += 1

    if np.isnan(u).any():
        solutions = [np.full(Nr, np.nan) for _ in t_output_labels]
    solutions = complete_solutions_if_needed(solutions, t_output_labels)
    result = {
        'r': r_coords.tolist(), 'solutions': [s.tolist() for s in solutions],
        'time': time.time() - start_time, 'gridPoints': Nr, 'timeSteps': Nt,
        'methodName': 'Implicit FD (Backward Euler)'
    }
    return {'result': result, 'tOutputLabels': t_output_labels}
