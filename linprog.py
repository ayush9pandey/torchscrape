#!/usr/bin/env python3

"""
lp_feasibility.py

Sets up and solves a simple linear program with variables x_i, y_i, t_i
for i=0..N, subject to constraints similar to the snippet provided:

  t_i - t_(i-1) = 1
  (x_i - x_(i-1)) ± (y_i - y_(i-1)) <= 1
  0 <= x_i <= grid_n - 1
  0 <= y_i <= grid_m - 1

We create a minimal "dummy" objective (e.g., 0) to solve for feasibility.
"""

import pulp
# Parameters
N = 4
grid_n = 10
grid_m = 10

# Create PuLP problem (minimize dummy objective just to find a feasible solution)
problem = pulp.LpProblem("Feasibility_Example", pulp.LpMinimize)

# Define variables:
#  x_i, y_i, t_i for i in [0..N]
#  We'll let them be continuous (LpVariable default) or set cat='Integer' if needed.
x = {}
y = {}
t = {}
for i in range(N+1):
    x[i] = pulp.LpVariable(f"x_{i}", lowBound=0, upBound=grid_n-1)
    y[i] = pulp.LpVariable(f"y_{i}", lowBound=0, upBound=grid_m-1)
    t[i] = pulp.LpVariable(f"t_{i}")  # no explicit bound provided in snippet

# Dummy objective function: 0
problem.setObjective(pulp.lpSum([]))  # Or pulp.lpSum(0)

# Add constraints for i=1..N referencing i-1
for i in range(1, N+1):
    # t_i - t_(i-1) = 1
    problem += (t[i] - t[i-1] == 1), f"t_constraint_{i}"

    # (x_i - x_(i-1)) + (y_i - y_(i-1)) <= 1
    problem += (x[i] - x[i-1] + y[i] - y[i-1] <= 1), f"xy_plus_{i}"

    # (x_i - x_(i-1)) - (y_i - y_(i-1)) <= 1
    problem += (x[i] - x[i-1] - y[i] + y[i-1] <= 1), f"xy_minus_{i}_1"

    # -(x_i - x_(i-1)) + (y_i - y[i-1]) <= 1
    problem += (-x[i] + x[i-1] + y[i] - y[i-1] <= 1), f"xy_minus_{i}_2"

    # -(x_i - x_(i-1)) - (y_i - y[i-1]) <= 1
    problem += (-x[i] + x[i-1] - y[i] + y[i-1] <= 1), f"xy_minus_{i}_3"

    # The snippet also had 0 <= x_i <= grid_n - 1, 0 <= y_i <= grid_m - 1
    # but we captured that in variable bounds.

# Solve
print("Solving the linear program for feasibility...")
result = problem.solve(pulp.PULP_CBC_CMD(msg=False))

# Report
print(f"Solver status: {pulp.LpStatus[result]}")
if pulp.LpStatus[result] == "Optimal":
    print("Feasible solution found. Variable values:")
    for i in range(N+1):
        print(f" i={i} -> x_{i} = {x[i].varValue:.3f}, y_{i} = {y[i].varValue:.3f}, t_{i} = {t[i].varValue:.3f}")
else:
    print("No feasible solution found or solver issue.")