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

def plot_ani(solutions):
    plots = []
    for solution in solutions:
        params = solution["params"]
        df: pd.DataFrame = solution["data"]

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

        fig = plt.figure()
        ax = fig.add_subplot()

        cart, = ax.plot([], [], 'o', c='k', label="cart")
        line, = ax.plot([], [], '-', c='r')
        pend, = ax.plot([], [], 'o', c='b', label="pendulum")

        plots.append([cart, np.array([X_vals, Y_vals])])
        plots.append([line, (np.array([X_vals, x_vals]), np.array([Y_vals, y_vals]))])
        plots.append([pend, (x_vals, y_vals)])

    def update_points(n):
        # cart.set_data(([X_vals[n]], [Y_vals[n]]))
        # line.set_data(([X_vals[n], x_vals[n]], [Y_vals[n], y_vals[n]]))
        # pend.set_data(([x_vals[n]], [y_vals[n]]))
        returns = []
        for plot in plots:
            print(plot)
            print(tuple(plot[1].T[n]))
            plot[0].set_data(tuple(plot[1].T[n]))
            returns.append(plot[0])

        returns = tuple(returns)
        print(returns)
        return *returns,

    ani = animation.FuncAnimation(fig, update_points, num_frames, interval=10, blit=True, repeat=True)

    all_x = np.concatenate((X_vals.copy(), px_vals.copy()))
    all_y = np.concatenate((Y_vals.copy(), py_vals.copy()))
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.legend()
