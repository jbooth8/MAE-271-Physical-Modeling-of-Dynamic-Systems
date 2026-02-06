from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from solve_problem import solve_problem
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

params: dict[str, float] = {}
params["l_arm"] = 1.0   # length of pendulum arm [m]
params["m_mass"] = 1.0  # mass of pendulum mass [kg]
params["g"] = 9.8       # gravitational force [m/s^2]

# Time
t_start: float = 0
t_end: float = 1.5
t_increment: float = 0.005 # visualization time step duration (sec)
# NOTE: playback speed is t_increment/(interval/1000)

t_span = (t_start, t_end)
t_eval = np.arange(min(t_span), max(t_span)+t_increment, t_increment)

num_frequencies = 10
num_amplitudes = 10
frequencies = np.linspace(15, 25, num_frequencies)
amplitudes = np.linspace(4, 5, num_amplitudes)

F, A = np.meshgrid(frequencies, amplitudes)
T = np.zeros_like(A)

pbar = tqdm(total=num_frequencies*num_amplitudes)

def process(params: dict, i, j, freq, ampl):
    params.copy()
    params["frequency"] = freq
    params["amplitude"] = ampl
    solution = solve_problem(params, t_eval, "sweep")
    result = np.max(np.abs(solution["data"]["theta"]))
    pbar.update()
    return result

params["theta"] = np.deg2rad(40)
results = {}
with ThreadPoolExecutor(max_workers=16) as executor:
    for j, i in product(range(num_frequencies), range(num_amplitudes)):
        freq = F[i, j]
        ampl = A[i, j]

        results[executor.submit(process, params, i, j, freq, ampl)] = [i, j]

    for future in as_completed(results):
        i, j = results[future]
        T[i, j] = future.result()

T_min, T_max = T.min(), T.max()

fig, ax = plt.subplots()

c = ax.pcolormesh(F, A, T, cmap='RdBu', vmin=T_min, vmax=T_max)
# set the limits of the plot to the limits of the data
ax.axis([F.min(), F.max(), A.min(), A.max()])
ax.set_title('Theta Maximum Magnitude')
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude (m)")
fig.colorbar(c, ax=ax)

plt.show()
