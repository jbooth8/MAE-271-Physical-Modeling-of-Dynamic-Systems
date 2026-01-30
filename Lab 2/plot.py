from typing import Any
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd


def plot_time(solutions):
    fig = plt.figure()
    ax = fig.add_subplot()
    for solution in solutions:
        name = solution["name"]
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
        theta_vals = df.get("theta").to_numpy() * 180 / np.pi
        tan_theta_vals = df.get("tan_theta").to_numpy()

        # ax.plot(t_vals, x_vals, label=f"{name}: x")
        # ax.plot(t_vals, y_vals, label=f"{name}: y")
        # ax.plot(t_vals, d_px_vals, label=f"{name}: d_px")
        ax.plot(t_vals, theta_vals, label=f"theta: {name}")
        # theta_ax.plot(ts, tan_theta_vals, label="tan_theta")
    plt.legend()
    plt.title("Theta vs. Time")
    plt.xlabel("time (s)")
    plt.ylabel("theta (deg)")
    return fig


def plot_ani(solutions: list[dict[str, Any]], interval: int = 10):
    fig = plt.figure()
    ax = fig.add_subplot()
    all_x = np.array([])
    all_y = np.array([])

    # Initialize arrays to store the animation plots and data to use to update them
    num_solutions = len(solutions)
    plots: list[dict[str, Line2D]] = [{} for n in range(num_solutions)]
    data: list[dict[str, np.ndarray]] = [{} for n in range(num_solutions)]

    for i, solution in enumerate(solutions):
        name: str = solution["name"]
        df: pd.DataFrame = solution["data"]

        data[i]["t_vals"] = df.get("t").to_numpy()
        data[i]["X_vals"] = df.get("X").to_numpy()
        data[i]["Y_vals"] = df.get("Y").to_numpy()
        data[i]["x_vals"] = df.get("x").to_numpy()
        data[i]["y_vals"] = df.get("y").to_numpy()

        (plots[i]["cart"],) = ax.plot([], [], "o", c="b", label=f"cart: {name}")
        (plots[i]["line"],) = ax.plot([], [], "-", c="k")
        (plots[i]["pend"],) = ax.plot([], [], "o", label=f"pendulum: {name}")

        num_frames = len(data[i]["t_vals"])

        all_x = np.concatenate((all_x, data[i]["X_vals"].copy(), data[i]["x_vals"].copy()))
        all_y = np.concatenate((all_y, data[i]["Y_vals"].copy(), data[i]["y_vals"].copy()))

    def update_points(n):
        returns = []
        for i in range(num_solutions):
            plots[i]["cart"].set_data(([data[i]["X_vals"][n]], [data[i]["Y_vals"][n]]))

            plots[i]["line"].set_data(
                (
                    [data[i]["X_vals"][n], data[i]["x_vals"][n]],
                    [data[i]["Y_vals"][n], data[i]["y_vals"][n]],
                )
            )

            plots[i]["pend"].set_data(([data[i]["x_vals"][n]], [data[i]["y_vals"][n]]))
            returns.extend([plots[i]["cart"], plots[i]["line"], plots[i]["pend"]])

        returns = tuple(returns)
        return (*returns,)

    ani = animation.FuncAnimation(
        fig, update_points, num_frames, interval=interval, blit=True, repeat=True
    )

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    plt.xlim(min_x - 0.5, max_x + 0.5)
    plt.ylim(min_y - 0.5, max_y + 0.5)
    plt.legend()
    plt.title("Pendulum Animation")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    return ani
