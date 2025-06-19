# constants.q
U0_NORMAL_CELL_DENSITY = 1.0

DEFAULT_SIMULATION_PARAMS = {
  'r0': 0.5,
  'D': 2.0e-9,
  'sc': 8.0e-6,
  'p': 0,
  'days': 30,
}

PARAMETER_CONFIGS = [
  { 'id': 'r0', 'label': 'Wound Half-Width (cm)', 'defaultValue': DEFAULT_SIMULATION_PARAMS['r0'], 'step': 0.1, 'min': 0.1, 'max': 2.0, 'type': 'number'},
  { 'id': 'D', 'label': 'Diffusivity (cm²/s)', 'defaultValue': DEFAULT_SIMULATION_PARAMS['D'], 'step': 1e-10, 'min': 1e-10, 'type': 'text'},
  { 'id': 'sc', 'label': 'Source Coeff. (1/s)', 'defaultValue': DEFAULT_SIMULATION_PARAMS['sc'], 'step': 1e-7, 'min': 0, 'type': 'text'},
  { 'id': 'p', 'label': 'Nonlinearity Param.', 'defaultValue': 0, 'step': 0.1, 'min': 0, 'max': 10, 'type': 'number' },
  { 'id': 'days', 'label': 'Simulation Days', 'defaultValue': DEFAULT_SIMULATION_PARAMS['days'], 'step': 1, 'min': 1, 'max': 90, 'type': 'number'},
]

N_TERMS_ANALYTICAL = 100
LINE_COLORS = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#2ecc71', '#1abc9c', '#3498db', '#9b59b6', '#8e44ad', '#38598b', '#d35400']