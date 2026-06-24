# SwiGLU Work Log

## Problem

Optimize the SwiGLU operation for the H100 leaderboard.

Key workload:
- Input `x`: `[batch_size, seq_len, in_features]`
- Weights `W`, `V`: `[in_features, hidden_size]`
- Biases `b`, `c`: `[hidden_size]`
- Output: `[batch_size, seq_len, hidden_size]`
- Main benchmark case: `batch_size=256; seq=64; in_features=2048; hidden_size=4096`

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
