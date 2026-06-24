# 1D Occupancy Decoder Work Log

## Problem

Optimize the 1D occupancy decoder forward pass for the H100 leaderboard.

Key workload:
- Queries: `[batch_size, num_queries, q_in_dim]`
- Latents: `[batch_size, num_latents, width]`
- Output: `[batch_size, num_queries, 1]`
- Main benchmark case: `batch_size=1; num_queries=250000; num_latents=1024; width=768; num_heads=12; q_in_dim=3`

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
