import numpy as np
import time
from simulation_service import get_time_output,complete_solutions_if_needed
from constants import U0_NORMAL_CELL_DENSITY, N_TERMS_ANALYTICAL

def solve_finite_difference_explicit(params):
    start_time = time.time()
    D, sc, p_nonlin, r0 = params['D'], params['sc'], params['p'], params['r0']
    tf = params['days'] * 24 * 3600
    u0 = U0_NORMAL_CELL_DENSITY
    time_outputs = get_time_output(params['days'])
    t_output_s, t_output_labels = time_outputs['seconds'], time_outputs['labels']

    Nr = 101
    dr = r0 / (Nr - 1)
    r_coords = np.linspace(0, r0, Nr)

    dt_diffusion = 0.25 * (dr ** 2) / (2 * D)
    dt_reaction = 0.1 / sc if sc > 0 else np.inf
    dt = min(dt_diffusion, dt_reaction)

    Nt = max(1000, int(tf / dt))
    Nt = min(Nt, 50000)
    dt = tf / Nt

    u = np.zeros(Nr)
    u[0] = u0
    solutions = []
    next_output_idx = 0
    dr2_inv = 1.0 / (dr ** 2)

    for n in range(Nt):
        u_old = u.copy()
        if p_nonlin == 0:
            D_coeff = D
        else:
            D_coeff = D * np.maximum(0, 1 - u_old[1:-1] / u0) ** p_nonlin
        laplacian = (u_old[2:] - 2 * u_old[1:-1] + u_old[:-2]) * dr2_inv
        source = sc * u_old[1:-1] * (1 - u_old[1:-1] / u0)
        u[1:-1] = u_old[1:-1] + dt * (D_coeff * laplacian + source)
        if p_nonlin == 0:
            D_coeff_boundary = D
        else:
            D_coeff_boundary = D * max(0, 1 - u_old[-1] / u0) ** p_nonlin
        diffusion_boundary = D_coeff_boundary * 2 * (u_old[-2] - u_old[-1]) * dr2_inv
        source_boundary = sc * u_old[-1] * (1 - u_old[-1] / u0)
        u[-1] = u_old[-1] + dt * (diffusion_boundary + source_boundary)
        np.clip(u, 0, u0, out=u)
        current_time = (n + 1) * dt
        if next_output_idx < len(t_output_s) and current_time >= t_output_s[next_output_idx]:
            solutions.append(u.copy())
            next_output_idx += 1
    solutions = complete_solutions_if_needed(solutions, t_output_labels)
    result = {
        'r': r_coords.tolist(), 'solutions': [s.tolist() for s in solutions],
        'time': time.time() - start_time, 'gridPoints': Nr, 'timeSteps': Nt,
        'methodName': 'Explicit FD (Forward Euler)'
    }
    return {'result': result, 'tOutputLabels': t_output_labels}
