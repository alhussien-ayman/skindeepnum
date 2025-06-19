import numpy as np
import time
from simulation_service import get_time_output,complete_solutions_if_needed
from constants import U0_NORMAL_CELL_DENSITY, N_TERMS_ANALYTICAL
def solve_crank_nicolson_newton(params):
    """
    Solves the PDE using the Crank-Nicolson method.
    For non-linear cases (p != 0), it uses an iterative Newton-Raphson
    solver at each time step to solve the resulting non-linear system.
    """
    start_time = time.time()
    D, sc, p_nonlin, r0 = params['D'], params['sc'], params['p'], params['r0']
    tf = params['days'] * 24 * 3600
    u0 = U0_NORMAL_CELL_DENSITY
    time_outputs = get_time_output(params['days'])
    t_output_s, t_output_labels = time_outputs['seconds'], time_outputs['labels']

    # Grid and Time Step setup
    Nr = 101  # Grid points (M+1 in your code)
    M = Nr - 1
    dr = r0 / M
    r_coords = np.linspace(0, r0, Nr)

    # Use a reasonable number of time steps, CN is stable
    Nt = 3000
    dt = tf / Nt

    # --- Initialization ---
    u_old = np.zeros(Nr)
    u_old[0] = u0  # Dirichlet boundary condition at r=0

    solutions = []
    next_output_idx = 0
    max_iter = 20  # Max Newton iterations
    tol = 1e-6  # Newton convergence tolerance
    eps = 1e-8  # Small epsilon for stability

    # --- Main Time Loop ---
    for n in range(Nt):
        u_new = u_old.copy()

        # Pre-calculate the explicit part of the equation (terms at time 'n')
        reaction_old = sc * u_old * (1 - u_old / u0)

        diffusion_old = np.zeros(Nr)
        u_mid_plus_old = 0.5 * (u_old[2:] + u_old[1:-1])
        u_mid_minus_old = 0.5 * (u_old[1:-1] + u_old[:-2])
        if p_nonlin == 0:
            D_eff_plus_old, D_eff_minus_old = D, D
        else:
            D_eff_plus_old = D * np.maximum(eps, 1 - u_mid_plus_old / u0) ** p_nonlin
            D_eff_minus_old = D * np.maximum(eps, 1 - u_mid_minus_old / u0) ** p_nonlin

        flux_plus_old = D_eff_plus_old * (u_old[2:] - u_old[1:-1]) / dr
        flux_minus_old = D_eff_minus_old * (u_old[1:-1] - u_old[:-2]) / dr
        diffusion_old[1:-1] = (flux_plus_old - flux_minus_old) / dr

        # --- Newton-Raphson Iteration for non-linear solve ---
        for k in range(max_iter):
            R = np.zeros(Nr)
            J = np.zeros((Nr, Nr))

            for j in range(1, M):
                uj, uj_left, uj_right = u_new[j], u_new[j - 1], u_new[j + 1]
                reaction_new = sc * uj * (1 - uj / u0)
                u_mid_plus = 0.5 * (uj_right + uj)
                u_mid_minus = 0.5 * (uj + uj_left)
                if p_nonlin == 0:
                    D_eff_plus, D_eff_minus = D, D
                else:
                    D_eff_plus = D * np.maximum(eps, 1 - u_mid_plus / u0) ** p_nonlin
                    D_eff_minus = D * np.maximum(eps, 1 - u_mid_minus / u0) ** p_nonlin
                flux_plus = D_eff_plus * (uj_right - uj) / dr
                flux_minus = D_eff_minus * (uj - uj_left) / dr
                diffusion_new = (flux_plus - flux_minus) / dr
                R[j] = (uj - u_old[j]) / dt - 0.5 * (diffusion_new + diffusion_old[j]) - 0.5 * (
                        reaction_new + reaction_old[j])

                delta = 1e-7
                for m_offset in [-1, 0, 1]:
                    m = j + m_offset
                    if 0 < m < M:
                        u_delta = u_new.copy()
                        u_delta[m] += delta
                        uj_d, uj_left_d, uj_right_d = u_delta[j], u_delta[j - 1], u_delta[j + 1]
                        reaction_new_d = sc * uj_d * (1 - uj_d / u0)
                        u_mid_plus_d = 0.5 * (uj_right_d + uj_d)
                        u_mid_minus_d = 0.5 * (uj_d + uj_left_d)
                        if p_nonlin == 0:
                            D_eff_plus_d, D_eff_minus_d = D, D
                        else:
                            D_eff_plus_d = D * np.maximum(eps, 1 - u_mid_plus_d / u0) ** p_nonlin
                            D_eff_minus_d = D * np.maximum(eps, 1 - u_mid_minus_d / u0) ** p_nonlin
                        flux_plus_d = D_eff_plus_d * (uj_right_d - uj_d) / dr
                        flux_minus_d = D_eff_minus_d * (uj_d - uj_left_d) / dr
                        diffusion_new_d = (flux_plus_d - flux_minus_d) / dr
                        R_delta = (uj_d - u_old[j]) / dt - 0.5 * (diffusion_new_d + diffusion_old[j]) - 0.5 * (
                                reaction_new_d + reaction_old[j])
                        J[j, m] = (R_delta - R[j]) / delta

            J[0, 0] = 1.0
            J[M, M] = 1.0
            J[M, M - 1] = -1.0
            R[0] = u_new[0] - u0
            R[M] = u_new[M] - u_new[M - 1]

            try:
                du = np.linalg.solve(J, -R)
                u_new += du
                if np.linalg.norm(du, np.inf) < tol: break
            except np.linalg.LinAlgError:
                print(f"Warning: Jacobian matrix is singular at time step {n}. Stopping iteration.")
                k = max_iter
                break

        u_old = u_new.copy()
        np.clip(u_old, 0, u0, out=u_old)
        current_time = (n + 1) * dt
        if not np.all(np.isfinite(u_old)):
            u_old.fill(np.nan)
            break
        if next_output_idx < len(t_output_s) and current_time >= t_output_s[next_output_idx]:
            solutions.append(u_old.copy())
            next_output_idx += 1

    if np.isnan(u_old).any():
        solutions = [np.full(Nr, np.nan) for _ in t_output_labels]
    solutions = complete_solutions_if_needed(solutions, t_output_labels)
    result = {
        'r': r_coords.tolist(), 'solutions': [s.tolist() for s in solutions],
        'time': time.time() - start_time, 'gridPoints': Nr, 'timeSteps': Nt,
        'methodName': 'Crank-Nicolson (Newton)'
    }
    return {'result': result, 'tOutputLabels': t_output_labels}
