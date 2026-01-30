from get_func import get_func
from scipy.integrate import solve_ivp
import pandas as pd


def solve_problem(
        params: dict[str, float],
        t_eval: list[float],
        name: str,
        rtol: float = 1e-12,
        atol: float = 1e-12,
    ):
    
    func, func_wrap, initial = get_func(params)
    output = solve_ivp(
        func_wrap,
        (min(t_eval), max(t_eval)),
        initial,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )

    ts = output.t
    ys = output.y

    states: list[dict[str, float]] = []
    for t, y in zip(ts, ys.T):
        d_state, state = func(t, y)
        states.append(state)

    df = pd.DataFrame(states)

    solution = {
        "params": params.copy(),
        "data": df,
        "name": name,
    }

    return solution
