#!/usr/bin/env python3

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm

def ten_dim_oscillator_cpu(x, t, gamma=0.1):
    """
    x = [x0, v0, x1, v1, ..., x4, v4]
    dx0/dt = v0
    dv0/dt = -x0 - gamma*v0
    etc.
    """
    dxdt = np.zeros_like(x)
    pos = x[0::2]  # even indices
    vel = x[1::2]  # odd indices
    
    dxdt[0::2] = vel
    dxdt[1::2] = -pos - gamma*vel
    
    return dxdt

def main():
    print("Running CPU-based ODE integration with SciPy...")

    # Initial condition
    x0 = np.random.randn(10)
    
    # Time points
    t0, t1 = 0.0, 100.0
    steps = 10000
    times = np.linspace(t0, t1, steps)
    
    start_time = time.time()
    
    solution = []
    x = x0.copy()
    
    for i in tqdm(range(steps - 1)):
        # Solve from times[i] to times[i+1]
        t_span = [times[i], times[i+1]]
        sol = odeint(ten_dim_oscillator_cpu, x, t_span)
        
        # Last value is the new state
        x = sol[-1]
        solution.append(x)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Finished integration in {total_time:.4f} seconds.")
    
    # Convert to NumPy array for plotting
    solution = np.array(solution)  # shape: (steps-1, 10)
    time_array = times[:-1]

    # Plot a few states (e.g., x0, x2, x4)
    plt.figure(figsize=(8, 5))
    for i in [0, 2, 4]:
        plt.plot(time_array, solution[:, i], label=f"x{i}")
    plt.title("SciPy ODE Integration (CPU) of 10D Oscillator")
    plt.xlabel("Time")
    plt.ylabel("State Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cpu_ode_solution.png")
    plt.show()

if __name__ == "__main__":
    main()
