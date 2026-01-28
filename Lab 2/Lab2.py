import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from get_func import get_func
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
rtol, atol = (1e-12, 1e-12)
params["amplitude"] = 0.1
params["frequency"] = 20*2*np.pi
params["theta"] = np.deg2rad(10)
func, func_wrap, initial = get_func(params)
output = solve_ivp(func_wrap, t_span, initial, t_eval=t_eval, method='RK45')#, rtol=rtol, atol=atol)

ts = output.t
ys = output.y

states: list[dict[str, float]] = []
for t, y in zip(ts, ys.T):
    d_state, state = func(t, y)
    states.append(state)

df = pd.DataFrame(states)

t_vals = df.get("t").to_numpy()
X_vals = df.get("X").to_numpy()
Y_vals = df.get("Y").to_numpy()
x_vals = df.get("x").to_numpy()
y_vals = df.get("y").to_numpy()
px_vals = df.get("px").to_numpy()
py_vals = df.get("py").to_numpy()
d_px_vals = df.get("d_px").to_numpy()
d_py_vals = df.get("d_py").to_numpy()
theta_vals = df.get("theta").to_numpy()
tan_theta_vals = df.get("tan_theta").to_numpy()

num_frames = len(t_vals)
theta_fig = plt.figure()
theta_ax = theta_fig.add_subplot()
theta_ax.plot(ts, x_vals, label="x")
theta_ax.plot(ts, y_vals, label="y")
theta_ax.plot(ts, d_px_vals, label="d_px")
theta_ax.plot(ts, theta_vals, label="theta")
# theta_ax.plot(ts, tan_theta_vals, label="tan_theta")
plt.legend()

fig = plt.figure()
ax = fig.add_subplot()

cart, = ax.plot([], [], 'o', c='k', label="cart")
line, = ax.plot([], [], '-', c='r')
pend, = ax.plot([], [], 'o', c='b', label="pendulum")

def update_points(n):
    cart.set_data(([X_vals[n]], [Y_vals[n]]))
    line.set_data(([X_vals[n], x_vals[n]], [Y_vals[n], y_vals[n]]))
    pend.set_data(([x_vals[n]], [y_vals[n]]))
    return cart, line, pend

ani = animation.FuncAnimation(fig, update_points, num_frames, interval=10, blit=True, repeat=True)

all_x = np.concatenate((X_vals.copy(), px_vals.copy()))
all_y = np.concatenate((Y_vals.copy(), py_vals.copy()))
min_x, max_x = min(all_x), max(all_x)
min_y, max_y = min(all_y), max(all_y)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.legend()
# plt.axis('equal')

# ani.save('increasingStraightLine.gif', fps=30, dpi=300)

# from IPython.display import HTML
# HTML(ani.to_jshtml())

plt.show()