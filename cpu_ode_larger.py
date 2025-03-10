#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

###############################################################################
# 1) Define the 100-DIM OSCILLATOR ODE (CPU Version)
###############################################################################
def large_oscillator_cpu(x, t, gamma=0.1):
    """
    100D system -> 50 oscillators:
      - Even indices (0, 2, 4, ...) = positions
      - Odd indices (1, 3, 5, ...)  = velocities
    
    dx[2i]/dt   = x[2i+1]
    dx[2i+1]/dt = - x[2i] - gamma * x[2i+1]
    """
    dxdt = np.zeros_like(x)
    pos = x[0::2]  # positions at even indices
    vel = x[1::2]  # velocities at odd indices
    
    # dpos/dt = vel
    dxdt[0::2] = vel
    
    # dvel/dt = -pos - gamma * vel
    dxdt[1::2] = -pos - gamma * vel
    
    return dxdt

###############################################################################
# 2) MAIN SCRIPT - Solve the 100D ODE on CPU with minimal overhead
###############################################################################
def main():
    print("Running CPU-based ODE integration...")

    # Initial condition of size 100
    x0 = np.random.randn(1000000)

    # Time range and resolution
    steps = 2000
    t0, t1 = 0.0, 100.0
    times = np.linspace(t0, t1, steps)

    # Measure solve time
    start_time = time.time()
    
    # Single-call integration with SciPy
    # shape of solution -> (steps, 100)
    solution = odeint(large_oscillator_cpu, x0, times, args=(0.1,))
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"ODE integration completed in {elapsed:.4f} seconds.")

    # Plot a few states (e.g., x0, x2, x4, x6, x8)
    plt.figure(figsize=(10, 6))
    for i in [0, 2, 4, 6, 8]:
        plt.plot(times, solution[:, i], label=f"x{i}")

    plt.title("100D Oscillator (CPU, SciPy odeint, single-call solve)")
    plt.xlabel("Time")
    plt.ylabel("State Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cpu_ode_large.png")
    plt.show()

if __name__ == "__main__":
    main()
