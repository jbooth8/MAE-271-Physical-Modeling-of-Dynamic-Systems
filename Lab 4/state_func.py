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
        s: dict[str, float] = params.copy()
        s["t"] = t
        
        # Make a copy of state variables in case they get changed
        state = deepcopy(state)
        s["theta"] = state[0]  # Actuator mass momentum
        s["q_air"] = state[1]  # Cylinder air volume at atmospheric pressure
        s["q_acc"] = state[2] # Accumulator air volume at atmospheric pressure
        s["p_fw"] = state[3]  # Flywheel angular momentum
        
        #----- Flywheel/piston kinematics --------------------
        # m(theta) -> dx/dtheta
        sth = np.sin(s["theta"])
        cth = np.cos(s["theta"])
        R2_L2 = s["R"]**2 / s["L"]**2
        m_theta = s["R"] * sth - R2_L2 * (s["L"] * sth * cth) / np.sqrt(1 - R2_L2 * sth**2) # Piston position derivative wrt theta

        #----- Flywheel position and velocity ----------------
        s["w_fw"] = s["p_fw"] / s["J_fw"]                               # Flywheel angular velocity
        s["x"] = np.sqrt(s["L"]**2 - s["R"]**2 * sth**2) - s["R"] * cth # Piston position
        s["d_x"] = m_theta * s["w_fw"]                                  # Piston velocity (dx/dtheta * dtheta/dt)
        s["V"] = s["A_p"] * (s["L"] + s["R"] - s["x"]) + s["V_TDC"]              # Cylinder volume (m^3)

        #----- Air -------------------------------------------
        # Pressures
        s["P"] = s["P_0"] * (s["V"]/s["q_air"])**s["gamma"]   # Cylinder pressure
        s["P_acc"] = s["q_acc"] / s["C_acc"]    # Accumulator pressure
        
        # Delta pressures
        del_P_i = s["P_0"] - s["P"]    # Pressure delta at inlet check-valve
        del_P_o  = s["P"] - s["P_acc"]  # Pressure delta at accumulator check-valve
        
        # Check valve areas
        s["A_i"] = 0 if del_P_i <= 0 else s["A_i_nom"]  # Inlet check-valve area correction
        s["A_o"] = 0 if del_P_i <= 0 else s["A_o_nom"]  # Accumulator check-valve area correction
        
        # Inlet/accumulator flow rates
        s["Q_i"] = s["A_i"] * np.sqrt(2 / s["rho"] * np.abs(del_P_i)) * np.sign(del_P_i) # Inlet air flow rate
        s["Q_o"] = s["A_o"] * np.sqrt(2 / s["rho"] * np.abs(del_P_o)) * np.sign(del_P_o) # Accumulator air flow rate
        
        # State flow rates
        s["d_q_air"] = s["A_p"] * m_theta * (s["p_fw"] / s["J_fw"]) + s["Q_i"] - s["Q_o"]   # Cylinder air flow rate
        s["d_q_acc"] = s["Q_o"]                                                             # Accumulator air flow rate

        #----- Flywheel controller and acceleration ----------
        s["tau_in"] = s["K_p"] * (s["w_fw_des"] - s["w_fw"])    # Controller torque
        s["d_p_fw"] = s["tau_in"] - m_theta * s["A_p"] * s["P"] # Rotational momentum acceleration (sum of forces)

        #----- Concatenate state derivatives -----------------
        d_state: list[float] = [
            s["w_fw"],
            s["d_q_air"],
            s["d_q_acc"],
            s["d_p_fw"], # Don't worry, this is a float, not an NDArray
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
