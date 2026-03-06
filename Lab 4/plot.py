from typing import Any
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd


def plot_2d(
    solutions: list[dict[str, float] | dict[str, pd.DataFrame]], 
    plot_keys: list[tuple[str, str]], 
    title: str
    ):

    fig = plt.figure()

    hue = np.linspace(0, 1, len(plot_keys), endpoint=False)
    sat = np.linspace(1, 0.2, len(solutions))
    val = 0.8

    # ---- Generate axes for each key pair ----
    axes: dict[tuple[str, str], Axes] = {}  # Keys and their associated axis
    x_keys: dict[str, list[Axes]] = {}
    y_keys: dict[str, list[Axes]] = {}
    keys_dict: dict[str, list[float]] = {}  # Consolidated data
    for i, keys in enumerate(plot_keys):
        assert keys not in axes.keys(), f"Redundant plotting keys '{keys}'. Each plotting key tuple should be unique."
        if keys[0] in x_keys.keys():
            # If x-key has been seen before, make a copy of the Axes associated with it
            seen_ax = x_keys[keys[0]][0]
            axes[keys] = seen_ax.twinx()
            x_keys[keys[0]].append(axes[keys])
            
            # Add corresponding y-key to dict
            if keys[1] in y_keys.keys():
                y_keys[keys[1]].append(axes[keys])
            else:
                y_keys[keys[1]] = [axes[keys]]
        else:
            if keys[1] in y_keys.keys():
                # If y-key has been seen before, make a copy of the Axes associated with it
                seen_ax = y_keys[keys[1]][0]
                axes[keys] = seen_ax.twiny()
                y_keys[keys[1]].append(axes[keys])
                
                # Add corresponding x-key to dict
                if keys[0] in x_keys.keys():
                    x_keys[keys[0]].append(axes[keys])
                else:
                    x_keys[keys[0]] = [axes[keys]]
            elif i > 0:
                phantom_x_axis = axes[plot_keys[0]].twinx()
                axes[keys] = phantom_x_axis.twiny()
                x_keys[keys[0]] = [axes[keys]]
                y_keys[keys[1]] = [phantom_x_axis]
            else: 
                # If neither key in key-pair have been seen before, 
                # make a new plot and create new entries for the keys
                axes[keys] = fig.add_subplot()
                x_keys[keys[0]] = [axes[keys]]
                y_keys[keys[1]] = [axes[keys]]


    assert len(x_keys.keys()) <= 2, f"Too many x-axes for 2D plot: {x_keys}. Maximum is 2."
    assert len(y_keys.keys()) <= 2, f"Too many y-axes for 2D plot: {y_keys}. Maximum is 2."

    for j, solution in enumerate(solutions):
        name = solution["name"]
        df: pd.DataFrame = solution["data"]

        # Collect data for all referenced keys in solution
        for key in (x_keys.keys() | y_keys.keys()):
            keys_dict[key] = df.get(key).to_numpy()

        for i, keys in enumerate(plot_keys):
            axes[keys].plot(
                keys_dict[keys[0]], keys_dict[keys[1]], 
                label=f"{keys[0]} vs. {keys[1]}: {name}", 
                c=hsv_to_rgb([hue[i], sat[j], val])
                )
    
    # Add axis labels for each key on each axis
    for i, key in enumerate(x_keys):
        x_keys[key][0].set_xlabel(f"{key}")
        if i > 0: 
            x_keys[key][0].xaxis.tick_top()
            x_keys[key][0].xaxis.set_label_position("top")
    for i, key in enumerate(y_keys):
        y_keys[key][0].set_ylabel(f"{key}")
        if i > 0: 
            y_keys[key][0].yaxis.tick_right()
            y_keys[key][0].yaxis.set_label_position("right")

    # Set plot title using a single axis
    axes[plot_keys[0]].set_title(title, y=1.12)
    fig.legend() # Add legend
    fig.tight_layout()
    return fig


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
