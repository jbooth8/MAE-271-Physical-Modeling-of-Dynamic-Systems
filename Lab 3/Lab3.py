import numpy as np
import matplotlib.pyplot as plt
from solve_problem import solve_problem
from plot import plot_time, plot_ani
plt.rcParams["animation.html"] = "jshtml"

# -----------------------------
# Initial Global Parameters
# -----------------------------
params: dict[str, float] = {}

# Constants
params["g"] = 9.8  # gravity (m/s^2)

# Mass
params["m_tot"] = 3000.0 / 2.2  # Total vehicle mass (lbs -> kg)
params["msmus"] = 5.0           # Sprung to unsprung mass ratio
params["mams"] = 0.04          # Actuator to sprung mass ratio

# Damping ratios
params["zeta_s"] = 0.7  # Passive damping ratio
params["zeta_c"] = 1.7  # Active damping ratio

# Suspension
params["w_a"] = 2 * np.pi * 5   # Actuator natural frequency (rad/s)
params["w_s"] = 2 * np.pi * 1.2 # Suspension natural frequency (Hz -> rad/s)
params["w_wh"] = 2 * np.pi * 8  # Tire / wheel hop frequency (Hz -> rad/s)

# Voice coil parameters
params["R_c"] = 0.005           # Winding resistance (Ohm)
params["T_c"] = 1.0             # Coupling constant (N/A)

# Vehicle velocity
params["U"] = 40 * 0.46 # Vehicle speed (mph -> m/s)

# Road
params["road_delta_x"] = 0.5        # Distance between randomly generated road heights (m)
params["road_length"] = 500         # Road length (m)
params["road_max_slope"] = 0.005    # Maximum instantaneous slope (dy/dx) of road

# Time
t_start: float = 0
t_end: float = 1
t_increment: float = 0.001

t_span = (t_start, t_end)
t_eval = np.arange(min(t_span), max(t_span)+t_increment, t_increment)


# Run simulations
solutions: list[str, dict[str, float]] = []
params["zeta_c"] = 0
solutions.append(solve_problem(params, t_eval, "baseline"))
params["zeta_c"] = 0.7
solutions.append(solve_problem(params, t_eval, "zeta_c=0.7"))

# ani = plot_ani(solutions)
fig = plot_time(solutions)
# plt.axis('equal')

# fig = plot_time(solutions)

# from IPython.display import HTML
# HTML(ani.to_jshtml())

plt.show()