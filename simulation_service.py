# simulation_service.py
import numpy as np

# ==============================================================================
# UTILITY AND HELPER FUNCTIONS
# ==============================================================================

def get_time_output(total_days):
    if total_days <= 0: return {'seconds': [], 'labels': []}
    num_outputs = 10
    t_total_seconds = total_days * 24 * 3600
    t_output_seconds = np.linspace(t_total_seconds / num_outputs, t_total_seconds, num_outputs)
    t_output_labels = [f"{(t / (24 * 3600)):.1f}d" for t in t_output_seconds]
    return {'seconds': t_output_seconds, 'labels': t_output_labels}


def solve_tridiagonal_system(a, b, c, d):
    """Optimized Thomas algorithm for tridiagonal systems"""
    n = len(b)
    if n == 0: return np.array([])
    if n == 1: return np.array([d[0] / b[0]])

    # Pre-allocate arrays
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)
    x_sol = np.zeros(n)

    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = b[i] - a[i - 1] * c_prime[i - 1]
        if np.abs(denominator) < 1e-12:
            return np.full(n, np.nan)
        if i < n - 1:
            c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denominator

    # Back substitution
    x_sol[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x_sol[i] = d_prime[i] - c_prime[i] * x_sol[i + 1]

    return x_sol

def solve_tridiagonal_system(a, b, c, d):
    """Optimized Thomas algorithm for tridiagonal systems"""
    n = len(b)
    if n == 0: return np.array([])
    if n == 1: return np.array([d[0] / b[0]])

    # Pre-allocate arrays
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)
    x_sol = np.zeros(n)

    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = b[i] - a[i - 1] * c_prime[i - 1]
        if np.abs(denominator) < 1e-12:
            return np.full(n, np.nan)
        if i < n - 1:
            c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denominator

    # Back substitution
    x_sol[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x_sol[i] = d_prime[i] - c_prime[i] * x_sol[i + 1]

    return x_sol


def complete_solutions_if_needed(solutions, labels):
    if len(solutions) > 0 and len(solutions) < len(labels):
        last_valid_solution = solutions[-1]
        while len(solutions) < len(labels):
            solutions.append(last_valid_solution.copy())
    return solutions


# def compare_methods(params):
#     from copy import deepcopy

#     # Run both methods
#     result_mol = solve_method_of_lines(deepcopy(params))
#     result_fem = solve_finite_element_method(deepcopy(params))

#     sol_mol = np.array(result_mol['result']['solutions'])
#     sol_fem = np.array(result_fem['result']['solutions'])
#     time_labels = result_mol['tOutputLabels']
#     r_coords = np.array(result_mol['result']['r'])

#     # Ensure shape match
#     assert sol_mol.shape == sol_fem.shape, "Shape mismatch between methods' solutions."

#     # Compute relative error
#     relative_errors = []
#     for u_mol, u_fem in zip(sol_mol, sol_fem):
#         norm_ref = np.linalg.norm(u_mol)
#         if norm_ref == 0:
#             err = 0.0
#         else:
#             err = np.linalg.norm(u_fem - u_mol) / norm_ref
#         relative_errors.append(err)

#     # Print time and error
#     print(f"\nExecution Time (s):")
#     print(f" - Method of Lines: {result_mol['result']['time']:.4f} s")
#     print(f" - Finite Element : {result_fem['result']['time']:.4f} s")

#     print("\nTime Step Label\tRelative Error")
#     for t_label, err in zip(time_labels, relative_errors):
#         print(f"{t_label:>12}\t{err:.3e}")

#     # Optional: plot relative error over time
#     plt.figure(figsize=(8, 4))
#     plt.plot(time_labels, relative_errors, marker='o', label='Relative Error')
#     plt.xlabel("Time Label")
#     plt.ylabel("Relative L2 Error")
#     plt.title("Relative Error between FEM and MoL")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.legend()
#     plt.show()

#     return {
#         "time_mol": result_mol['result']['time'],
#         "time_fem": result_fem['result']['time'],
#         "relative_errors": relative_errors,
#         "time_labels": time_labels
#     }
# params = {
#     'D': 2e-9,
#     'sc': 8e-6,
#     'p': 1,
#     'r0': 0.5,
#     'days': 5
# }
# comparison_results = compare_methods(params)
