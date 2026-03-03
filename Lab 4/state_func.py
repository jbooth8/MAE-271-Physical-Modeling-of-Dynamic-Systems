from copy import deepcopy
import numpy as np


def get_func(params: dict[str, float]):
    '''
    Generate state derivative calculator functions for current simulation.
    '''
    # Make a copy of the input parameters to ensure the returned function
    # does not change when the original parameters dictionary changes
    params = deepcopy(params)

    # Get initial condition
    initial = [
        0,                                      # Flywheel position (theta)
        params["V_0"],                          # Cylinder air volume at atmospheric pressure (q_air)
        params["V_acc"],                        # Accumulator air volume at atmospheric pressure (q_acc)
        params["w_fw_des"] * params["J_fw"],    # Flywheel momentum (p_fw)
        ]

    def func(t: float, state: list[float]):
        '''
        Calculate state derivative from current state.

        Parameters:
            t (float): Current time of simulation.
            state (list[float]): Current state of simulation.

        Returns:
            d_state (list[float]): Current state derivative.
            state_ext (dict[str, float]): Extended state information.
        '''
        # Create current extended state dict
        s: dict[str, float] = params
        s["t"] = t
        
        # Make a copy of state variables in case they get changed
        state = deepcopy(state)
        s["theta"] = state[0]  # Actuator mass momentum
        s["q_air"] = state[1]  # Cylinder air volume at atmospheric pressure
        s["q_acc"] = state[2] # Accumulator air volume at atmospheric pressure
        s["p_fw"] = state[3]  # Flywheel angular momentum
        
        # Calculate m(theta) -> dx/dtheta
        sth = np.sin(s["theta"])
        cth = np.cos(s["theta"])
        R2_L2 = s["R"]**2 / s["L"]**2
        m_theta = s["R"] * sth - R2_L2 * (s["L"] * sth * cth) / np.sqrt(1 - R2_L2 * sth**2)

        # Cylinder pressure
        s["P"] = s["P_0"] * (1 / (1 - s["q_air"] / s["V_0"])**s["gamma"] - 1)

        # Flywheel rotational acceleration
        s["d_p_fw"] = s["tau_in"] - m_theta * s["A_p"] * s["P"]


        s["P_acc"] = s["q_acc"] / s["C_acc"]
        del_P_i = 0 - s["P"]
        del_P_o  = s["P"] - s["P_acc"]
        Q_i = s["A_i"] * np.sqrt(2 / s["rho"] * np.abs(del_P_i)) * np.sign(del_P_i)
        Q_o = s["A_o"] * np.sqrt(2 / s["rho"] * np.abs(del_P_o)) * np.sign(del_P_o)
        s["d_q_air"] = 1 / s["A_p"] * m_theta * (s["p_fw"] / s["J_fw"]) + Q_i - Q_o    # Cylinder air flow rate
        s["d_q_acc"] = Q_o   # Accumulator air flow rate

        # Concatenate state derivatives
        d_state: list[float] = [
            d_p_a,
            d_p_s,
            d_p_us,
            d_q_a,
            d_q_s,
            d_q_t, # Don't worry, this is a float, not an NDArray
            ]

        return d_state, s

    def func_wrap(t: float, state: list[float]):
        """
        Since solve_ivp() needs a function that only returns the state
        derivatives, we create a wrapper for func() that discards the rest of
        the state and returns only the state derivatives
        """
        d_state, _ = func(t, state)
        return d_state

    return func, func_wrap, initial
