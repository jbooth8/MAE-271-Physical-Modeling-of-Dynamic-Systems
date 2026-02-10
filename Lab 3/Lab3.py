import numpy as np
import matplotlib.pyplot as plt
from .solve_problem import solve_problem
from .plot import plot_time, plot_ani
plt.rcParams["animation.html"] = "jshtml"

# -----------------------------
# Global Parameters
# -----------------------------
params: dict[str, float] = {}

# Constants
g = 9.8  # gravity (m/s^2)
params["g"] = g

# Masses
m_tot = 3000.0 / 2.2       # Total vehicle mass (kg)
msmus = 5.0               # Sprung to unsprung mass ratio
m_us = m_tot / (1 + msmus)  # Unsprung mass (kg)
m_s = m_tot - m_us          # Sprung mass (kg)
params["m_tot"] = m_tot
params["m_us"] = m_us
params["m_s"] = m_s

# Suspension
w_s = 2 * np.pi * 1.2      # Suspension natural frequency (rad/s)
k_s = m_s * w_s**2         # Suspension stiffness (N/m)
params["w_s"] = w_s
params["k_s"] = k_s

# Damping ratios
zeta_s = 0.7               # Passive damping ratio
zeta_c = 0.7               # Active damping ratio (can be varied)
params["zeta_s"] = zeta_s
params["zeta_c"] = zeta_c

# Damping constants
b_s = 2 * zeta_s * w_s * m_s  # Suspension damping (N·s/m)
b_c = 2 * zeta_c * w_s * m_s  # Control damping (N·s/m)
params["b_s"] = b_s
params["b_c"] = b_c

# Tire / wheel hop
w_wh = 2 * np.pi * 8       # Wheel hop frequency (rad/s)
k_t = m_us * w_wh**2       # Tire stiffness (N/m)
params["w_wh"] = w_wh
params["k_t"] = k_t

# Electrical / actuator parameters
R_w = 0.005               # Winding resistance (Ohm)
T = 5.0                   # Coupling constant (N/A)
params["R_w"] = R_w
params["T"] = T

m_a = 0.02 * m_s          # Actuator mass (kg)
w_a = 2 * np.pi * 5       # Actuator frequency (rad/s)
params["m_a"] = m_a
params["w_a"] = w_a

k_a = m_a * w_a**2        # Actuator stiffness (N/m)
b_a = 2 * 0.1 * w_a * m_a # Actuator damping (N·s/m)
params["k_a"] = k_a
params["b_a"] = b_a

# Vehicle velocity
U = 40 * 0.46             # Vehicle speed (m/s)
params["U"] = U

# Road input parameters
#A = 2 * 0.0254          # bump height [m]
d = 3 * 0.3048          # bump length [m]
#u = 30 * 0.46           # vehicle speed [m/s]
params["d"] = d

# Time
t_start: float = 0
t_end: float = 1
t_increment: float = 0.001

t_span = (t_start, t_end)
t_eval = np.arange(min(t_span), max(t_span)+t_increment, t_increment)


# Run simulations
solutions = []
params["amplitude"] = 0.1
params["frequency"] = 20
params["theta"] = np.deg2rad(10)
solutions.append(solve_problem(params, t_eval, "base"))

# ani = plot_ani(solutions, interval = 10)
# plt.axis('equal')

# fig = plot_time(solutions)

# from IPython.display import HTML
# HTML(ani.to_jshtml())

# plt.show()