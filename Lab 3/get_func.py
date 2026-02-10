from copy import deepcopy
import numpy as np


def get_func(params: dict[str, float]):
    # Make a copy of the input parameters to ensure the returned function
    # does not change when the original parameters dictionary changes
    params = deepcopy(params)

    # Extract parameters from dict
    theta = params["theta"]
    ampl = params["amplitude"]
    freq = params["frequency"] * 2 * np.pi  # Convert from Hz to rad/s
    leng = params["l_arm"]
    mass = params["m_mass"]
    g = params["g"]

    # Get initial condition (add position into the system so solve_ivp can integrate it for us)
    px = 0
    initial = [px, theta]

    # Create derivative calculator function to be used by solve_ivp()
    # state = [px, theta]
    def func(t, state):
        state = deepcopy(state)
        ps = state[0]
        pus = state[1]
        pa = state[2]
        qs = state[3]
        qt = state[4]
        qa = state[5]

        # Sine and cosine of theta, for computational speed
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)

        # Position of cart and its derivatives
        X = 0
        Y = ampl * np.sin(freq * t)
        d_Y = ampl * freq * np.cos(freq * t)
        dd_Y = -(freq**2) * Y

        # State derivatives
        d_theta = px / (leng * cos_theta * mass)
        d_px = (
            mass * g * sin_theta * cos_theta
            + mass * dd_Y * sin_theta * cos_theta
            - tan_theta * d_theta * px
        )

        # Other state variables
        x = X + leng * np.sin(theta)  # x-position of pendulum mass
        y = Y + leng * np.cos(theta)  # y-position of pendulum mass
        py = mass * (d_Y - tan_theta * px / mass)  # y-momentum of pendulum mass
        d_py = mass * (
            dd_Y - tan_theta * d_px / mass - px * d_theta / (mass * cos_theta**2)
        )  # first time-derivative of y-momentum of pendulum mass

        # Concatenate state derivatives
        d_state = [d_px, d_theta]

        # Create current state dict
        state = {}
        state["t"] = t
        state["X"] = X
        state["Y"] = Y
        state["x"] = x
        state["y"] = y
        state["px"] = px
        state["py"] = py
        state["theta"] = theta
        state["d_Y"] = d_Y
        state["d_px"] = d_px
        state["d_py"] = d_py
        state["d_theta"] = d_theta
        state["dd_Y"] = dd_Y
        state["tan_theta"] = tan_theta

        return d_state, state

    def func_wrap(t, state):
        """
        Since solve_ivp() needs a function that only returns the state
        derivatives, we create a wrapper for func() that discards the rest of
        the state and returns only the state derivatives
        """
        d_state, state = func(t, state)
        return d_state

    return func, func_wrap, initial
