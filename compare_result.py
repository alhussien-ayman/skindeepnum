import explicit_finite_difference as explicit
import implicit
import crank_nicolson
import finite_element_method as FEM
import method_of_lines as MOL
import analytical
import numpy as np

params = {
    'r0': 0.5,
    'D': 2.0e-9,
    'sc': 8e-6,
    'p': 1,
    'days': 30,
}

def compute_errors(method_solutions, reference_solutions):
    method_solutions = np.array(method_solutions)
    reference_solutions = np.array(reference_solutions)

    assert method_solutions.shape == reference_solutions.shape, "Mismatch in solution shapes"

    epsilon = 1e-10 
    relative_error = np.abs(method_solutions - reference_solutions) / (np.abs(reference_solutions) + epsilon)

    max_error = np.max(relative_error)                  
    mean_error = np.mean(relative_error)                

    return max_error, mean_error

# Solve all methods
mol = MOL.solve_method_of_lines(params)
fdm = explicit.solve_finite_difference_explicit(params)
fem = FEM.solve_finite_element_method(params)
implicit_sol = implicit.solve_finite_difference_implicit(params)
cranck = crank_nicolson.solve_crank_nicolson_newton(params)
# analytical_sol = analytical.solve_analytical(params)


reference_solutions = mol['result']['solutions']

print(f"{'Method':30} | {'Time (s)':>9} | {'#Eqns':>7} | {'Time/Eqn':>10} | {'Max Rel Err':>13} | {'Mean Rel Err':>13}")
print("-" * 95)

for result in [mol, fdm, fem, implicit_sol, cranck]:
    name = result['result']['methodName']
    sol = result['result']['solutions']
    time_taken = result['result']['time']
    num_eqs = result['result']['gridPoints']
    time_per_eq = time_taken / num_eqs
    max_err, mean_err = compute_errors(sol, reference_solutions)

    print(f"{name:30} | {time_taken:9.4f} | {num_eqs:7} | {time_per_eq:10.3e} | {max_err:13.3e} | {mean_err:13.3e}")
