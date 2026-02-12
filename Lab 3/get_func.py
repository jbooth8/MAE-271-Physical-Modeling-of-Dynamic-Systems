from copy import deepcopy
import numpy as np
import scipy


def get_func(params: dict[str, float]):
    '''
    Generate state derivative calculator functions for current simulation.
    '''
    # Make a copy of the input parameters to ensure the returned function
    # does not change when the original parameters dictionary changes
    params = deepcopy(params)

    # Extract parameters from dict
    g = params["g"]         # Gravitational constant (m/s^2)
    U = params["U"]         # Vehicle speed in x-direction (m/s)
    
    # Masses
    m_tot = params["m_tot"]     # Actuated mass (kg)
    msmus = params["msmus"]     # Sprung to unsprung mass ratio
    m_us = m_tot / (1 + msmus)  # Unsprung mass (kg)
    m_s = m_tot - m_us          # Sprung mass (kg)
    m_a = 0.02 * m_s            # Actuator mass (kg)
    params["m_us"] = m_us
    params["m_s"] = m_s
    params["m_a"] = m_a

    # Passive Stiffness
    w_s = params["w_s"]     # Suspension natural frequency (rad/s)
    k_s = m_s * w_s**2      # Suspension stiffness (N/m)
    params["k_s"] = k_s
    w_wh = params["w_wh"]   # Tire wheel hop frequency (rad/s)
    k_t = m_us * w_wh**2    # Tire stiffness (N/m)
    params["k_t"] = k_t

    # Passive Damping
    zeta_s = params["zeta_s"]       # Passive damping ratio
    b_s = 2 * zeta_s * w_s * m_s    # Suspension damping (N·s/m)
    params["b_s"] = b_s

    # Actuator - Passive
    w_a = 2 * np.pi * 5         # Actuator frequency (rad/s)
    k_a = m_a * w_a**2          # Actuator stiffness (N/m)
    b_a = 2 * 0.1 * w_a * m_a   # Actuator damping (N·s/m)
    params["w_a"] = w_a
    params["k_a"] = k_a
    params["b_a"] = b_a

    # Actuator - Active
    zeta_c = params["zeta_c"]       # Active damping ratio
    T_c = params["T_c"]             # Voice coil coupling constant (N/A)
    R_c = params["R_c"]             # Voice coil winding resistance (Ohms)
    b_c = 2 * zeta_c * w_s * m_s    # Voice coil damping (N·s/m)
    params["b_c"] = b_c

    # Road
    road_delta_x = params["road_delta_x"]       # Distance between randomly generated road heights (m)
    road_length = params["road_length"]         # Road length (m)
    road_max_slope = params["road_max_slope"]   # Maximum instantaneous slope (dy/dx) of road
    X_i = np.arange(0, road_length+road_delta_x, road_delta_x)  # Generate road position vector
    road_slope_raw = np.random.random(np.shape(X_i))    # Generate random slope vector
    slope_i = road_max_slope * (road_slope_raw - np.mean(road_slope_raw)) # Scale and subtract mean from road slope vector
    def slope_interpolator(x: float):
        return np.interp(x, X_i, slope_i)

    # Get initial condition
    initial = [
        0,  # Actuator mass momentum
        0,  # Sprung mass momentum
        0,  # Unsprung mass momentum
        (m_a) * g / k_a,                # Actuator displacement
        (m_a + m_s) * g / k_s,          # Suspension displacement
        (m_a + m_s + m_us) * g / k_t,   # Tire displacement
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
        p_a = state[0]  # Actuator mass momentum
        p_s = state[1]  # Sprung mass momentum
        p_us = state[2] # Unsprung mass momentum
        q_a = state[3]  # Actuator displacement
        q_s = state[4]  # Suspension displacement
        q_t = state[5]  # Tire displacement
        s["p_a"] = p_a
        s["p_s"] = p_s
        s["p_us"] = p_us
        s["q_a"] = q_a
        s["q_s"] = q_s
        s["q_t"] = q_t

        # Position of road and its derivatives
        slope_interp = deepcopy(slope_interpolator)
        X = U * t   # vehicle velocity x time
        Y = np.zeros_like(X)
        dYdX = slope_interp(X)
        s["X"] = X
        s["Y"] = Y
        s["dYdX"] = dYdX

        # Velocities
        v_a = p_a / m_a     # Actuator mass velocity
        v_s = p_s / m_s     # Sprung mass velocity
        v_us = p_us / m_us  # Unsprung mass velocity
        v_in = U * dYdX            # Tire input velocity
        s["v_a"] = v_a
        s["v_s"] = v_s
        s["v_us"] = v_us
        s["v_in"] = v_in

        # Voice Coil Current
        i_c = b_c * v_s / T_c   # Voice coil input current (A)
        e_c = R_c * i_c         # Voice coil input voltage (V)
        P_c = i_c * e_c         # Voice coil power (W)
        s["i_c"] = i_c
        s["e_c"] = e_c
        s["P_c"] = P_c

        # Gravitational Force
        F_g_a = -m_a * g    # Actuator mass gravitational force
        F_g_s = -m_s * g    # Sprung mass gravitational force
        F_g_us = -m_us * g  # Unsprung mass gravitational force
        s["F_g_a"] = F_g_a
        s["F_g_s"] = F_g_s
        s["F_g_us"] = F_g_us

        # Actuator Force
        F_b_a = b_a * (v_s - v_a)   # Actuator damping force
        F_s_a = k_a * q_a           # Actuator spring force
        F_vc = T_c * i_c            # Voice coil force
        F_a = F_b_a + F_s_a + F_vc  # Total actuator force
        s["F_b_a"] = F_b_a
        s["F_s_a"] = F_s_a
        s["F_vc"] = F_vc
        s["F_a"] = F_a
        
        # Suspension Force
        F_b_s = b_s * (v_us - v_s)  # Suspension damping force
        F_s_s = k_s * q_s           # Suspension spring force
        F_s = F_b_s + F_s_s         # Total suspension force
        s["F_b_s"] = F_b_s
        s["F_s_s"] = F_s_s
        s["F_s"] = F_s
        
        # Tire Force
        F_t = k_t * q_t if q_t > 0 else 0   # Prevent tire sticking the ground
        s["F_t"] = F_t

        # State Derivatives
        d_p_a = F_g_a + F_a
        d_p_s = F_g_s + F_s - F_a
        d_p_us = F_g_us + F_t - F_s
        d_q_a = v_s - v_a
        d_q_s = v_us - v_s
        d_q_t = v_in - v_us
        s["d_p_a"] = d_p_a
        s["d_p_s"] = d_p_s
        s["d_p_us"] = d_p_us
        s["d_q_a"] = d_q_a
        s["d_q_s"] = d_q_s
        s["d_q_t"] = d_q_t

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
