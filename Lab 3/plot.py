from typing import Any
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd


def plot_time(solutions):
    fig_acc = plt.figure()
    fig_pow = plt.figure()
    acc = fig_acc.add_subplot()
    pow = fig_pow.add_subplot()
    for solution in solutions:
        name = solution["name"]
        df = solution["data"]

        t_vals = df.get("t").to_numpy()
        X_vals = df.get("X").to_numpy()
        Y_vals = df.get("Y").to_numpy()
        a_s_vals = df.get("a_s").to_numpy()
        P_c_vals = df.get("P_c").to_numpy()
        dYdX_vals = df.get("dYdX").to_numpy()

        acc.plot(t_vals, a_s_vals, label=f"sprung mass acceleration: {name}")
        pow.plot(t_vals, P_c_vals, label=f"actuator power: {name}")
        acc.plot(t_vals, dYdX_vals, label=f"dYdX: {name}")
        pow.plot(t_vals, dYdX_vals, label=f"dYdX: {name}")
        # theta_ax.plot(ts, tan_theta_vals, label="tan_theta")
    acc.legend()
    pow.legend()
    acc.set_title("Acceleration vs. Time")
    pow.set_title("Power vs. Time")
    acc.set_xlabel("time (s)")
    pow.set_xlabel("time (s)")
    acc.set_ylabel("accleration (m/s^2)")
    pow.set_ylabel("power (watts)")
    return fig_acc, fig_pow


def plot_ani(solutions: list[dict[str, Any]], interval: int = 10):
    fig = plt.figure()
    ax = fig.add_subplot()
    all_x = np.array([])
    all_y = np.array([])

    # Initialize arrays to store the animation plots and data to use to update them
    num_solutions = len(solutions)
    plots: list[dict[str, Line2D]] = [{} for _ in range(num_solutions)]
    data: list[dict[str, np.ndarray]] = [{} for _ in range(num_solutions)]

    for i, solution in enumerate(solutions):
        name: str = solution["name"]
        df: pd.DataFrame = solution["data"]

        data[i]["t_vals"] = df.get("t").to_numpy()
        data[i]["X_vals"] = df.get("X").to_numpy()
        data[i]["Y_vals"] = df.get("Y").to_numpy()
        data[i]["y_us_vals"] = 12 - df.get("q_t").to_numpy()
        data[i]["y_s_vals"] = 12 - df.get("q_s").to_numpy() + data[i]["y_us_vals"]
        data[i]["y_a_vals"] = 6 - df.get("q_a").to_numpy() + data[i]["y_s_vals"]

        if i == 0:
            road = ax.plot(data[i]["X_vals"], data[i]["Y_vals"], "-", c="k", label="road")

        (plots[i]["unsprung"],) = ax.plot([], [], "o", label=f"unsprung: {name}")
        (plots[i]["sprung"],) = ax.plot([], [], "o", label=f"sprung: {name}")
        (plots[i]["actuator"],) = ax.plot([], [], "o", label=f"actuator: {name}")

        num_frames = len(data[i]["t_vals"])

        all_x = np.concatenate((all_x, data[i]["X_vals"].copy()))
        all_y = np.concatenate((
            all_y, data[i]["Y_vals"].copy(),
            data[i]["y_us_vals"].copy(),
            data[i]["y_s_vals"].copy(),
            data[i]["y_a_vals"].copy(),
        ))

    def update_points(n):
        returns = []
        for i in range(num_solutions):
            plots[i]["unsprung"].set_data(([data[i]["X_vals"][n]], [data[i]["y_us_vals"][n]]))
            plots[i]["sprung"].set_data(([data[i]["X_vals"][n]], [data[i]["y_s_vals"][n]]))
            plots[i]["actuator"].set_data(([data[i]["X_vals"][n]], [data[i]["y_a_vals"][n]]))
            returns.extend([plots[i]["unsprung"], plots[i]["sprung"], plots[i]["actuator"]])

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
