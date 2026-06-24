# FlashAttention Work Log

## Problem

Optimize exact attention for the H100 leaderboard.

Key workload:
- Inputs: `Q`, `K`, `V`, each `[batch_size, num_heads, seq_len, head_dim]`
- Output: `[batch_size, num_heads, seq_len, head_dim]`
- Benchmark cases:
  - `batch_size=1; num_heads=64; seq_len=1024; head_dim=128`
  - `batch_size=2; num_heads=64; seq_len=4096; head_dim=128`
  - `batch_size=4; num_heads=64; seq_len=8192; head_dim=128`

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
