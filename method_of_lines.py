import numpy as np
import time
from simulation_service import get_time_output,complete_solutions_if_needed
from constants import U0_NORMAL_CELL_DENSITY, N_TERMS_ANALYTICAL
def solve_method_of_lines(params):
    start_time = time.time()
    D_param, sc, p_nonlin, r0 = params['D'], params['sc'], params['p'], params['r0']
    tf = params['days'] * 24 * 3600
    u0 = U0_NORMAL_CELL_DENSITY
    eps = 1e-9
    time_outputs = get_time_output(params['days'])
    t_output_s, t_output_labels = time_outputs['seconds'], time_outputs['labels']

    Nr = 101
    dr = r0 / (Nr - 1)
    r_coords = np.linspace(0, r0, Nr)
    Nt = min(20000, max(2000, int(tf / 50)))
    dt = tf / Nt

    u = np.zeros(Nr)
    u[0] = u0
    solutions = []
    next_output_idx = 0
    dr_inv = 1.0 / dr

    for n in range(Nt):
        u_current = u.copy()
        dudt = np.zeros(Nr)
        u_mid_plus = 0.5 * (u_current[2:] + u_current[1:-1])
        u_mid_minus = 0.5 * (u_current[1:-1] + u_current[:-2])
        if p_nonlin == 0:
            D_eff_plus, D_eff_minus = D_param, D_param
        else:
            D_eff_plus = D_param * np.maximum(eps, 1 - u_mid_plus / u0) ** p_nonlin
            D_eff_minus = D_param * np.maximum(eps, 1 - u_mid_minus / u0) ** p_nonlin
        flux_plus = D_eff_plus * (u_current[2:] - u_current[1:-1]) * dr_inv
        flux_minus = D_eff_minus * (u_current[1:-1] - u_current[:-2]) * dr_inv
        diffusion_term = (flux_plus - flux_minus) * dr_inv
        source_term = sc * u_current[1:-1] * (1 - u_current[1:-1] / u0)
        dudt[1:-1] = diffusion_term + source_term
        u_mid_boundary = 0.5 * (u_current[-1] + u_current[-2])
        if p_nonlin == 0:
            D_eff_boundary = D_param
        else:
            D_eff_boundary = D_param * max(eps, 1 - u_mid_boundary / u0) ** p_nonlin
        flux_boundary = D_eff_boundary * (u_current[-1] - u_current[-2]) * dr_inv
        diffusion_boundary = -flux_boundary * dr_inv
        source_boundary = sc * u_current[-1] * (1 - u_current[-1] / u0)
        dudt[-1] = diffusion_boundary + source_boundary
        if not np.all(np.isfinite(dudt)):
            u.fill(np.nan)
            break
        u[1:] += dt * dudt[1:]
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
        'methodName': 'Method of Lines'
    }
    return {'result': result, 'tOutputLabels': t_output_labels}