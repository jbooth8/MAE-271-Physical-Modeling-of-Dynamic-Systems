import numpy as np
import matplotlib.pyplot as plt
from solve_problem import solve_problem
from plot import plot_2d, plot_ani
plt.rcParams["animation.html"] = "jshtml"

# -----------------------------
# Initial Global Parameters
# -----------------------------
params: dict[str, float] = {}
 
# Flywheel and piston rod
params["a"] = 3       # a (ft)
params["weight"] = 1500        # weight (lbs)
params["J_r"] = params["weight"]*(2*params["a"])**2   # J_r
params["D"] = 2.5    # Piston diameters (in)
params["g_p"] = 1                      # Ball screw pitch (in)
params["g_a"] = params["g_p"] / (2 * np.pi)                      # Ball screw gain (in / rad)
params["tau_i"] = 5000    # Input torque (Nm)

params["R_w"] = 1    # Winding resistance (Ohm)
params["R_m"] = 2    # Radius (in)
params["m_m"] = 5                        # Rotor mass (lbs)
params["V_0"] = params["m_m"] * params["R_m"]**2 / 2    # Rotary inertia
params["T_m"] = 0.54        # Motor constant (Nm/A)
params["damping_ratio"] = 0.3                               # Passive damping ratio
params["f_n"] = 1.5 # Natural frequency (Hz)
params["b_d"] = 2 * params["damping_ratio"] * params["f_n"] * 2*np.pi * params["weight"]    # Damping coefficient

# Time
t_start: float = 0
t_end: float = 0.2
t_increment: float = 0.0001

t_span = (t_start, t_end)
t_eval = np.arange(min(t_span), max(t_span)+t_increment, t_increment)


# Run simulations
solutions: list[str, dict[str, float]] = []
params["K_p"] = 0.25 / 10
solutions.append(solve_problem(params, t_eval, f"K_p = {params['K_p']}"))

params["K_p"] = 1 / 10
solutions.append(solve_problem(params, t_eval, f"K_p = {params['K_p']}"))

params["K_p"] = 4 / 10
solutions.append(solve_problem(params, t_eval, f"K_p = {params['K_p']}"))

# ani = plot_ani(solutions)
fig1 = plot_2d(solutions, [("x", "theta"), ("d_x", "w_fw")], "X vs. Theta")
fig2 = plot_2d(solutions, [("t", "P / P_0")], "P / P_0 vs. Time")
fig2 = plot_2d(solutions, [("t", "w_fw")], "w_fw vs. Time")
fig2 = plot_2d(solutions, [("P / P_0", "V / V_0")], "V / V_0 vs. P / P_0")
# plt.axis('equal')

# fig = plot_time(solutions)

# from IPython.display import HTML
# HTML(ani.to_jshtml())

plt.show()