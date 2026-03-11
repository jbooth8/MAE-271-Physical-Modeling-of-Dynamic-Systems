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
        0,  # Flywheel position (theta)
        params["V_0"],  # Cylinder air volume at atmospheric pressure (q_air)
        0,  # Accumulator air volume at atmospheric pressure (q_acc)
        0,  # Flywheel momentum (p_fw)
        0,
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
        s: dict[str, float] = params.copy()
        s["t"] = t
        
        # Make a copy of state variables in case they get changed
        state = deepcopy(state)
        s["p_g"] = state[0]
        s["p_J_m"] = state[1]
        s["p_J_r"] = state[2]
        s["q_L"] = state[3]
        s["q_R"] = state[4]

        s["i_c"] = s["K_a"] * s["p_J_r"] / s["J_r"]     # Controller current (A)

        s["C_L"] = (s["V_0"] + s["q_L"]) / (s["rho"] * s["c"]**2)    # Left accumulator compliance (m^4*s/kg)
        s["C_R"] = (s["V_0"] + s["q_L"]) / (s["rho"] * s["c"]**2)    # Right accumulator compliance (m^4*s/kg)
        
        v_L = s["p_g"] / s["m"] - s["p_J_r"] * s["a"] / s["J_r"]
        v_R = s["p_g"] / s["m"] + s["p_J_r"] * s["a"] / s["J_r"]
        e_v_L = s["q_L"] * s["C_L"] - s["b_d"] * v_L
        e_v_R = s["q_R"] * s["C_R"] - s["b_d"] * v_R
        s["f_c"] = s["i_c"] * s["R_w"] + s["T_m"] * s["p_J_m"] / s["J_m"]
        s["P_c"] = s["i_c"] * s["f_c"]  # Controller power

        #----- Flywheel/piston kinematics --------------------
        s["d_p_g"] = - s["m"] * s["g"] + e_v_L + e_v_R
        s["d_p_J_m"] = s["q_L"] * s["C_L"] / (s["A"] * s["g_a"]) - s["q_R"] * s["C_R"] / (s["A"] * s["g_a"]) + s["T_m"] * s["i_c"]
        s["d_p_J_r"] = s["tau_i"] + e_v_L / s["a"] + e_v_R / s["a"]
        s["d_q_L"] = - v_L - s["p_J_m"] * s["A"] * s["g_a"] / s["J_m"]
        s["d_q_R"] = - v_R - s["p_J_m"] * s["A"] * s["g_a"] / s["J_m"]


        #----- Concatenate state derivatives -----------------
        d_state: list[float] = [
            s["d_p_g"],
            s["d_p_J_m"],
            s["d_p_J_r"],
            s["d_q_L"],
            s["d_q_R"], # Don't worry, this is a float, not an NDArray
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
