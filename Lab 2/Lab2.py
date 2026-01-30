import numpy as np
import matplotlib.pyplot as plt
from solve_problem import solve_problem
from plot import plot_ani
plt.rcParams["animation.html"] = "jshtml"

# -----------------------------
# Global Parameters
# -----------------------------
params: dict[str, float] = {}
params["l_arm"] = 1.0   # length of pendulum arm [m]
params["m_mass"] = 1.0  # mass of pendulum mass [kg]
params["g"] = 9.8       # gravitational force [m/s^2]

params["theta"] = 0     # initial pendulum angle [rad]
params["amplitude"] = 0 # oscillator amplitude [m]
params["frequency"] = 0 # oscillator frequency [Hz]

# Time
t_start: float = 0
t_end: float = 10
t_increment: float = 0.001

t_span = (t_start, t_end)
t_eval = np.arange(min(t_span), max(t_span)+t_increment, t_increment)


# ---------------------------------------------------
# ---------------------------------------------------
solutions = []
params["amplitude"] = 0.1
params["frequency"] = 20*2*np.pi
params["theta"] = np.deg2rad(10)
solutions.append(solve_problem(params, t_eval))

ani = plot_ani(solutions)

# plt.axis('equal')

# ani.save('increasingStraightLine.gif', fps=30, dpi=300)

# from IPython.display import HTML
# HTML(ani.to_jshtml())

plt.show()