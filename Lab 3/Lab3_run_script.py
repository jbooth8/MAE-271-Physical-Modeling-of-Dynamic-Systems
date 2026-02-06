import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from rich import print
from rich.traceback import install as rich_tracback

rich_tracback()

###########################################
# Copy from Lab 1
# Same concept of the quarter car with added inertial actuator for active damping effects


# Global Parameters

# Constants
g = 9.8  # gravity (m/s^2)

# Masses
m_tot = 3000.0 / 2.2       # Total vehicle mass (kg)
msmus = 5.0               # Sprung to unsprung mass ratio
m_us = m_tot / (1 + msmus)  # Unsprung mass (kg)
m_s = m_tot - m_us          # Sprung mass (kg)

# Suspension
w_s = 2 * np.pi * 1.2      # Suspension natural frequency (rad/s)
k_s = m_s * w_s**2         # Suspension stiffness (N/m)

# Damping ratios
zeta_s = 0.7               # Passive damping ratio
zeta_c = 0.7               # Active damping ratio (can be varied)

# Damping constants
b_s = 2 * zeta_s * w_s * m_s  # Suspension damping (N·s/m)
b_c = 2 * zeta_c * w_s * m_s  # Control damping (N·s/m)

# Tire / wheel hop
w_wh = 2 * np.pi * 8       # Wheel hop frequency (rad/s)
k_t = m_us * w_wh**2       # Tire stiffness (N/m)

# Electrical / actuator parameters
R_w = 0.005               # Winding resistance (Ohm)
T = 5.0                   # Coupling constant (N/A)

m_a = 0.02 * m_s          # Actuator mass (kg)
w_a = 2 * np.pi * 5       # Actuator frequency (rad/s)

k_a = m_a * w_a**2        # Actuator stiffness (N/m)
b_a = 2 * 0.1 * w_a * m_a # Actuator damping (N·s/m)

# Vehicle velocity
U = 40 * 0.46             # Vehicle speed (m/s)
                          # gravity [m/s^2]

# Road input parameters
#A = 2 * 0.0254          # bump height [m]
d = 3 * 0.3048          # bump length [m]
#u = 30 * 0.46           # vehicle speed [m/s]

# Time span
t_span = (0, 1)
t_eval = np.arange(0, 1.001, 0.001)

#----------------------------------------------------------------------------------

# A = 2in bump height
for A in [2*0.0254]:   # Set bump height (convert in to m)
    results_A_2 = []  # store all simulation cases
    # [ps, pus, pa, qs, qt, qa]
    initial = [0, 0, 0, 1.0 * m_s * g / k_s, m_tot * g / k_t, m_a * g / k_a]

    for u in [20*0.46, 25*0.46, 30*0.46]:  # Loop through varying car speeds for each bump height (convert mph to m/s)

            # Model Function
            def LabDemoFunc(t, s):
                ps, pus, pa, qs, qt, qa = s

                # Road input
                X = u * t
                Y = 0.5 * A * (1 - np.cos(2 * np.pi * X / d))
                dYdX = 0.5 * A * (2 * np.pi / d) * np.sin(2 * np.pi * X / d)

                if X > d:
                    Y = 0
                    dYdX = 0

                vin = u * dYdX

                # Tire force
                Ft = k_t * qt
                if qt <= 0:
                    Ft = 0

                # Velocities
                vs = ps / m_s
                vus = pus / m_us
                va = pa / m_a

                # Spring forces
                F_a = b_c * vs
                F_s_a = k_a * (vs - va)
                F_s = k_s * qs + b_s * (vus - vs)
                F_t = k_t * (vin - vus)

                # State derivatives
                dps = -m_s * g + F_s
                dpus = -m_us * g + Ft - F_s
                dpa = -m_a * g + F_s_a + F_a
                dqs = vus - vs
                dqt = vin - vus
                dqa = vs - va

                ds = [dps, dpus, dpa, dqs, dqt, dqa]
                ext = [X, Y, F_t, vs, vus, va, vin, F_a]

                return ds, ext

            # Wrapper for solver
            def ode_wrapper(t, s):
                ds, _ = LabDemoFunc(t, s)
                return ds

            # -----------------------------
            # Run Simulation
            # -----------------------------
            sol = solve_ivp(ode_wrapper, t_span, initial, t_eval=t_eval)

            t = sol.t
            s = sol.y.T

            # Extract states
            ps, pus, pa, qs, qt, qa = s.T

            # Compute extra outputs
            ext = np.zeros((len(t), 8))
            ds = np.zeros((len(t), 6))

            for i in range(len(t)):
                ds[i], ext[i] = LabDemoFunc(t[i], s[i])

            dps, dpus, dpa, dqs, dqt, dqa = ds.T
            X, Y, F_t, vs, vus, va, vin, F_a = ext.T

            # Store results

            results_A_2.append({
                "A": A,
                "u": u,
                "t": t,
                "ps": ps,
                "pus": pus,
                "qs": qs,
                "qt": qt,
                "X": X,
                "Y": Y,
                "Ft": F_t,
                "vs": vs,
                "vus": vus,
                "vin": vin,
                "Fss": F_a,
                "dps": dps,
                "ms": m_s,
            })
    # print(results_A_2)
            # qs_array = results_A_2[qs]
            # print(qs_array(0))
    # Plot Results

    # Sprung masss acceleration
    plt.figure()
    for r in results_A_2:
        label = f"A={r['A']:.2f} m, u={r['u']:.2f} "
        plt.plot(r["t"], (r["dps"]/r["ms"]), label=label)
        plt.ylabel("Acceleration [m/s^2]")
        plt.legend()

    plt.twinx()
    plt.plot(t, Y, label="Road", color="k")
    plt.ylabel("Road Input [m]")
    plt.xlabel("Time (s)")
    plt.legend(loc='lower right')
    plt.title('Accelration of Sprung Mass (A = 2in)')
    plt.grid(True)
    plt.show()

    # Relative displacement across the suspension
    plt.figure()
    for r in results_A_2:
        qs_array = r["qs"]
        print(qs_array)
        label = f"A={r['A']:.2f} m, u={r['u']:.2f} m/s "
        plt.plot(r["t"], (r["qs"]-r["qs"][0]), label=label)
        plt.ylabel("Displacement [m]")
        plt.legend()

    plt.twinx()
    plt.plot(t, Y, label="Road", color="k")
    plt.ylabel("Road Input [m]")
    plt.xlabel("Time (s)")
    plt.ylim(0,0.15)
    plt.legend(loc='lower right')
    plt.title('Relative displacement across the suspension (A = 2in)')
    plt.grid(True)
    plt.show()

    # Tire force
    plt.figure()
    for r in results_A_2:
        label = f"A={r['A']:.2f} m, u={r['u']:.2f} m/s"
        plt.plot(r["t"], (r["Ft"]), label=label)
        plt.ylabel("Tire Force [N]")
        plt.legend()

    plt.twinx()
    plt.plot(t, Y, label="Road", color="k")
    plt.ylabel("Road Input [m]")
    plt.xlabel("Time (s)")
    plt.legend(loc='lower right')
    plt.title('Tire Force (A = 2in)')
    plt.grid(True)
    plt.show()

# --------------------------------------------------------------------------------------
