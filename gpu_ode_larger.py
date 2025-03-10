#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# 1) Define the 100-DIM OSCILLATOR ODE
###############################################################################
class LargeOscillator(nn.Module):
    """
    100D system -> 50 oscillators:
      - Even indices (0, 2, 4, ...) = positions
      - Odd indices (1, 3, 5, ...)  = velocities
    dx[2i]/dt   = x[2i+1]
    dx[2i+1]/dt = - x[2i] - gamma * x[2i+1]
    """

    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, t, x):
        # x can be shape (..., 100)
        dxdt = torch.zeros_like(x)

        pos = x[..., 0::2]  # positions at even indices
        vel = x[..., 1::2]  # velocities at odd indices

        # dpos/dt = vel
        dxdt[..., 0::2] = vel

        # dvel/dt = -pos - gamma*vel
        dxdt[..., 1::2] = -pos - self.gamma * vel

        return dxdt

###############################################################################
# 2) MAIN SCRIPT - Solve the ODE on the GPU with minimal overhead
###############################################################################
def main():
    # Check for GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create the ODE function and move to GPU
    ode_func = LargeOscillator(gamma=0.1).to(device)

    # Initial condition of size 100 (50 positions, 50 velocities)
    # Start with random or any custom initialization
    x0 = torch.randn(1000000, device=device)

    # Time range -> single call with many steps
    steps = 2000     # Increase steps for smoother solution
    t0, t1 = 0.0, 100.0
    times = torch.linspace(t0, t1, steps, device=device)

    # Measure solve time
    start_time = time.time()

    # Single call to integrate the system
    # shape of solution -> (steps, 100)
    solution = odeint(ode_func, x0, times)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"ODE integration completed in {elapsed:.4f} seconds.")

    # Move result to CPU for plotting
    solution_cpu = solution.detach().cpu().numpy()
    times_cpu = times.detach().cpu().numpy()

    # Plot a few representative states
    plt.figure(figsize=(10, 6))
    # For example, plot the first 5 positions
    for i in [0, 2, 4, 6, 8]:
        plt.plot(times_cpu, solution_cpu[:, i], label=f"x{i}")

    plt.title("100D Oscillator (PyTorch + torchdiffeq, single-call GPU solve)")
    plt.xlabel("Time")
    plt.ylabel("State Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gpu_ode_large.png")
    plt.show()

if __name__ == "__main__":
    main()
