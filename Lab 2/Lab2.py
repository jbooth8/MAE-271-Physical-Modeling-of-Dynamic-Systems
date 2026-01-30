import numpy as np
import matplotlib.pyplot as plt
from solve_problem import solve_problem
from plot import plot_time, plot_ani
plt.rcParams["animation.html"] = "jshtml"

# -----------------------------
# Global Parameters
# -----------------------------
params: dict[str, float] = {}
params["l_arm"] = 1.0   # length of pendulum arm [m]
params["m_mass"] = 1.0  # mass of pendulum mass [kg]
params["g"] = 9.8       # gravitational force [m/s^2]

params["theta"] = 0     # initial pendulum angle [deg] measured clockwise from vertical
params["amplitude"] = 0 # oscillator amplitude [m]
params["frequency"] = 0 # oscillator frequency [Hz]

# Time
t_start: float = 0
t_end: float = 2
t_increment: float = 0.010

t_span = (t_start, t_end)
t_eval = np.arange(min(t_span), max(t_span)+t_increment, t_increment)


# Run simulations
solutions = []
params["amplitude"] = 0.1
params["frequency"] = 20
params["theta"] = np.deg2rad(10)
solutions.append(solve_problem(params, t_eval, "10°"))
params["amplitude"] = 0.1
params["frequency"] = 20
params["theta"] = np.deg2rad(20)
solutions.append(solve_problem(params, t_eval, "20°"))
params["amplitude"] = 0.1
params["frequency"] = 20
params["theta"] = np.deg2rad(30)
solutions.append(solve_problem(params, t_eval, "30°"))
params["amplitude"] = 0.15
params["frequency"] = 10
params["theta"] = np.deg2rad(40)
solutions.append(solve_problem(params, t_eval, "40°"))

# ani = plot_ani(solutions, interval = 10)

# plt.axis('equal')

fig = plot_time(solutions)

# from IPython.display import HTML
# HTML(ani.to_jshtml())

plt.show()