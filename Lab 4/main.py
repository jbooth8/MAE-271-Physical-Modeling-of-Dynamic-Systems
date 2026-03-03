import numpy as np
import matplotlib.pyplot as plt
from solve_problem import solve_problem
from plot import plot_time, plot_ani
plt.rcParams["animation.html"] = "jshtml"

# -----------------------------
# Initial Global Parameters
# -----------------------------
params: dict[str, float] = {}

# Flywheel and piston rod
params["m_fw"] = 10 / 2.2       # Flywheel mass (lbs -> kg)
params["R"] = 4 * 0.0254        # Flywheel rod joint radius (in -> m)
params["L"] = 4 * params["R"]   # Rod length (m)
params["J_fw"] = params["m_fw"] * params["R"]**2 / 2    # Flywheel moment of inertia (kg * m^2)

# Piston
params["D_p"] = 4 * 0.0254                      # Piston diameter (in -> m)
params["A_p"] = np.pi * params["D_p"]**2 / 4    # Piston area (m^2)

# Air
params["V_st"] = params["A_p"] * 2 * params["R"]    # Volume displaced due to stroke (m^3)
params["V_TDC"] = 80 / 10**6                        # Volume remaining at top of stroke (cc -> m^3)
params["V_0"] = params["V_st"] + params["V_TDC"]    # Volume at BDC (m^3)
params["P_0"] = 1 * 10**5                           # Atmospheric pressure (atm -> Pa)
params["gamma"] = 1.4                               # Specific heat ratio

# Controller
params["w_fw_des"] = 1500 / 60 * 2 * np.pi  # Desired flywheel rotational speed (RPM -> rad/s)
params["K_p"] = 1   # Controller proportional gain

# Valves
params["d_o"] = 0.5 * 0.0254      # Outlet check-valve flow diameter (in -> m)
params["d_i"] = 0.5 * 0.0254      # Inlet check-valve flow diameter (in -> m)
params["A_o_nom"] = np.pi * params["d_o"]**2 / 4  # Outlet check-valve flow area (m)
params["A_i_nom"] = np.pi * params["d_i"]**2 / 4  # Inlet check-valve flow area (m)

# Accumulator
params["rho"] = 1.28                    # Air density at atmospheric pressure (kg/m^3)
params["c"] = 340                       # Air speed of sound (m/s)
params["V_acc"] = 5 * params["V_0"]     # Accumulator volume (m^3)
params["C_acc"] = params["V_acc"] / (params["rho"] * params["c"]**2)    # Accumulator compliance (m^4*s/kg)

# Time
t_start: float = 0
t_end: float = 1
t_increment: float = 0.001

t_span = (t_start, t_end)
t_eval = np.arange(min(t_span), max(t_span)+t_increment, t_increment)


# Run simulations
solutions: list[str, dict[str, float]] = []
solutions.append(solve_problem(params, t_eval, "baseline"))

# ani = plot_ani(solutions)
fig = plot_time(solutions)
# plt.axis('equal')

# fig = plot_time(solutions)

# from IPython.display import HTML
# HTML(ani.to_jshtml())

plt.show()