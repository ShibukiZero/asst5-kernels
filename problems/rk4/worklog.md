# RK4 Work Log

## Problem

Optimize the 3D heat equation solver with 8th-order finite-difference Laplacian and RK4 time integration for the H100 leaderboard.

Key workload:
- Input field: `(Nz, Ny, Nx)`, float32
- Output field: `(Nz, Ny, Nx)` after RK4 updates
- Main benchmark case: `grid_size=600; n_steps=10`

## Optimization Entries

Append one entry per meaningful experiment.

### Entry 0 - Baseline

Date:

Code version:
- File(s):
- Short description:

Command:
```bash
python ../eval.py benchmark test_cases/test.txt
```

Correctness:
- Status:
- Notes:

Performance:
- Runtime:
- Profiler stats:

Observation:
- What seems to limit performance?

Hypothesis:
- What change should improve it, and why?

Next step:
- 
