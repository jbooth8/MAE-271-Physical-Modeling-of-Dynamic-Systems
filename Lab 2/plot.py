from typing import Any
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd

def plot_time(solutions):
    for solution in solutions:
        params = solution["params"]
        df = solution["data"]

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

        theta_fig = plt.figure()
        theta_ax = theta_fig.add_subplot()
        theta_ax.plot(t_vals, x_vals, label=f"{params}: x")
        theta_ax.plot(t_vals, y_vals, label=f"{params}: y")
        theta_ax.plot(t_vals, d_px_vals, label=f"{params}: d_px")
        theta_ax.plot(t_vals, theta_vals, label=f"{params}: theta")
        # theta_ax.plot(ts, tan_theta_vals, label="tan_theta")
    plt.legend()

def plot_ani(solutions: list[dict[str, Any]]):
    update_funcs = []
    fig = plt.figure()
    ax = fig.add_subplot()
    all_x = np.array([])
    all_y = np.array([])

    for solution in solutions:
        params: dict[str, float] = solution["params"]
        df: pd.DataFrame = solution["data"]

        t_vals = df.get("t").to_numpy()
        X_vals = df.get("X").to_numpy()
        Y_vals = df.get("Y").to_numpy()
        x_vals = df.get("x").to_numpy()
        y_vals = df.get("y").to_numpy()
        num_frames = len(t_vals)

        cart, = ax.plot([], [], 'o', c='b', label="cart")
        line, = ax.plot([], [], '-', c='k')
        pend, = ax.plot([], [], 'o', c='r', label="pendulum")

        def update(n):
            cart.set_data(([X_vals[n]], [Y_vals[n]]))
            line.set_data(([X_vals[n], x_vals[n]], [Y_vals[n], y_vals[n]]))
            pend.set_data(([x_vals[n]], [y_vals[n]]))
            return [cart, line, pend]

        update_funcs.append(update)
        all_x = np.concatenate((all_x, X_vals.copy(), x_vals.copy()))
        all_y = np.concatenate((all_y, Y_vals.copy(), y_vals.copy()))

    def update_points(n):
        returns = []
        for func in update_funcs:
            plots = func(n)
            returns.extend(plots)

        returns = tuple(returns)
        return *returns,

    ani = animation.FuncAnimation(fig, update_points, num_frames, interval=10, blit=True, repeat=True)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    plt.xlim(min_x-0.5, max_x+0.5)
    plt.ylim(min_y-0.5, max_y+0.5)
    plt.legend()

    return ani
