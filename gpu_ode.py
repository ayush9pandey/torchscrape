#!/usr/bin/env python3

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from tqdm import tqdm

# For optional GPU usage logging
import pynvml

###############################################################################
# GPU usage logging setup
###############################################################################
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use GPU 0 (change if needed)

def log_gpu_usage(file_obj):
    """Log current GPU memory usage (bytes) & GPU utilization (%) with timestamps."""
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
    file_obj.write(f"{time.time()}, {mem_info.used}, {util_info.gpu}\n")
    file_obj.flush()

###############################################################################
# 10-DIM OSCILLATOR ODE (PyTorch)
###############################################################################
class TenDimOscillator(nn.Module):
    """
    x = [x0, v0, x1, v1, ..., x4, v4] -> 5 oscillators (position, velocity) pairs
    dx0/dt = v0
    dv0/dt = -x0 - gamma*v0
    etc.
    """
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, t, x):
        # x shape: (batch_size, 10) or just (10,)
        dxdt = torch.zeros_like(x)
        
        pos = x[..., 0::2]
        vel = x[..., 1::2]
        
        dxdt[..., 0::2] = vel
        dxdt[..., 1::2] = -pos - self.gamma * vel
        return dxdt

###############################################################################
# Main script
###############################################################################
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create model and send to GPU (if available)
    ode_func = TenDimOscillator(gamma=0.1).to(device)

    # Initial condition on device
    x0 = torch.randn(10, device=device)
    
    # Time range
    t0, t1 = 0.0, 100.0
    steps = 10000
    times = torch.linspace(t0, t1, steps, device=device)

    # We will integrate step by step to:
    # 1) Show progress with tqdm
    # 2) Log GPU usage each step
    start_time = time.time()
    
    solution = []
    x = x0.clone()
    
    with open("gpu_usage.log", "w") as f:
        f.write("timestamp,mem_used_bytes,gpu_util_percent\n")
        
        for i in tqdm(range(steps - 1)):
            # Integrate from times[i] to times[i+1]
            t_span = times[i : i+2]  # 2 points: [ti, ti+1]
            
            # Single step
            x_next = odeint(ode_func, x, t_span)[-1]
            x = x_next.clone()
            
            solution.append(x.cpu().detach().numpy())
            
            # Log GPU usage
            log_gpu_usage(f)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Finished integration in {total_time:.4f} seconds.")
    
    # Convert solution to NumPy for plotting
    solution = np.array(solution)  # shape: (steps-1, 10)
    time_array = times[:-1].cpu().numpy()  # The times for each step

    # Plot a few states (e.g., x0, x2, x4)
    plt.figure(figsize=(8, 5))
    for i in [0, 2, 4]:
        plt.plot(time_array, solution[:, i], label=f"x{i}")
    plt.title("PyTorch ODE Integration (GPU) of 10D Oscillator")
    plt.xlabel("Time")
    plt.ylabel("State Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gpu_ode_solution.png")
    plt.show()

if __name__ == "__main__":
    main()
