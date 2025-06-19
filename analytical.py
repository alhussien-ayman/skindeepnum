import numpy as np
import time
from constants import U0_NORMAL_CELL_DENSITY, N_TERMS_ANALYTICAL
from simulation_service import get_time_output,complete_solutions_if_needed
def solve_analytical(params):
    start_time = time.time()
    if params['p'] != 0 or params['sc'] != 0:
        return {
            'result': {
                'methodName': f"Analytical (N/A: p≠0 or sc≠0)",
                'time': 0, 'gridPoints': 'N/A', 'timeSteps': 'N/A',
                'r': [], 'solutions': []
            },
            'tOutputLabels': []
        }
    D, r0 = params['D'], params['r0']
    u0 = U0_NORMAL_CELL_DENSITY
    time_outputs = get_time_output(params['days'])
    t_output_s, t_output_labels = time_outputs['seconds'], time_outputs['labels']

    Nr = 101
    r_coords = np.linspace(0, r0, Nr)
    solutions = []

    for t in t_output_s:
        v_rt = np.zeros(Nr)
        for n_term in range(1, N_TERMS_ANALYTICAL + 1):
            lambda_n = (2 * n_term - 1) * np.pi / (2 * r0)
            Bn = -4 * u0 / ((2 * n_term - 1) * np.pi)
            term = Bn * np.sin(lambda_n * r_coords) * np.exp(-D * lambda_n ** 2 * t)
            v_rt += term
        u_profile = u0 + v_rt
        u_profile[0] = u0
        np.clip(u_profile, 0, u0, out=u_profile)
        solutions.append(u_profile)
    solutions = complete_solutions_if_needed(solutions, t_output_labels)
    result = {
        'r': r_coords.tolist(), 'solutions': [s.tolist() for s in solutions],
        'time': time.time() - start_time, 'gridPoints': Nr, 'timeSteps': 'N/A',
        'methodName': 'Analytical Solution'
    }
    return {'result': result, 'tOutputLabels': t_output_labels}

import numpy as np
import matplotlib.pyplot as plt