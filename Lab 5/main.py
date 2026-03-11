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
params["a"] = 3 * 12 / 0.0254   # a (ft -> m)
params["m"] = 1500 / 2.2 # weight (lbs -> kg)
params["g"] = 9.81        # Gravitational acceleration (m/s^2)
params["J_r"] = params["m"]*(2*params["a"])**2   # J_r
params["D"] = 2.5 * 0.0254    # Piston diameters (in -> m)
params["A"] = np.pi * params["D"] / 4   # Piston area (m^2)
params["g_p"] = 1 * 0.0254  # Ball screw pitch (in -> m)
params["g_a"] = params["g_p"] / (2 * np.pi)   # Ball screw gain (m / rad)
params["tau_i"] = 5000   # Input torque (Nm)
params["rho"] = 1.225 # Air density (kg/m^3)
params["c"] = 340 # Speed of sound (m/s^2)

params["K_a"] = 100000    # Controller proportional gain
params["R_w"] = 1    # Winding resistance (Ohm)
params["R_m"] = 2 * 0.0254    # Radius (in -> m)
params["m_m"] = 5 / 2.2  # Rotor mass (lbs -> kg)
params["J_m"] = params["m_m"] * params["R_m"]**2 / 2    # Rotary inertia
params["T_m"] = 0.54  # Motor constant (Nm/A)
params["damping_ratio"] = 0.3   # Passive damping ratio
params["f_n"] = 1.5 # Natural frequency (Hz)
params["b_d"] = 2 * params["damping_ratio"] * params["f_n"] * 2*np.pi * params["m"]    # Damping coefficient

params["h"] = 20 * 0.0254 # Cylinder height (in -> m)
params["V_0"] = params["A"] * params["h"] # Cylinder volume (m^3)

# Time
t_start: float = 0
t_end: float = 0.2
t_increment: float = 0.0001

t_span = (t_start, t_end)
t_eval = np.arange(min(t_span), max(t_span)+t_increment, t_increment)


# Run simulations
solutions: list[str, dict[str, float]] = []
solutions.append(solve_problem(params, t_eval, f"base"))

# ani = plot_ani(solutions)
fig1 = plot_2d(solutions, [("t", "p_g")], "X vs. Momentum")
# plt.axis('equal')

# fig = plot_time(solutions)

# from IPython.display import HTML
# HTML(ani.to_jshtml())

plt.show()