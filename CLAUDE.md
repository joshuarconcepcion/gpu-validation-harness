# GPU Validation Harness

A validation and benchmarking harness for the NVIDIA RTX 3090.

## Purpose

Collect GPU metrics via pynvml, run compute workloads, compare results against baselines, log everything to SQLite, and expose results through a FastAPI HTTP layer.

## Tech Stack

| Component | Library |
|---|---|
| GPU metrics | `pynvml` (wraps NVML) |
| Compute workloads | `cupy` (CUDA array ops — Linux only) |
| API layer | `fastapi` + `uvicorn` |
| Persistence | `sqlite3` (stdlib) |
| Tests | `pytest` |

## Project Structure

```
gpu-validation-harness/
  src/
    instrumentation.py  # pynvml metric collection (temp, power, utilization, memory)
    workload.py         # GPU compute workloads (matrix ops, memory bandwidth, etc.)
    validator.py        # baseline comparison logic
    database.py         # SQLite schema and logging helpers
    api.py              # FastAPI routes
  tests/
    test_instrumentation.py
    test_validator.py
  main.py               # entry point
  CLAUDE.md
  .venv/                # virtual environment (not committed)
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pynvml fastapi uvicorn pytest
# cupy requires Linux + CUDA drivers — install on the target machine:
# pip install cupy-cuda12x
```

## Run Commands

```bash
# Activate venv
source .venv/bin/activate

# Start API server
uvicorn src.api:app --reload --port 8000

# Run all tests
pytest tests/

# Run main harness
python main.py
```

## Notes

- `cupy` is Linux-only (CUDA drivers not available on macOS). Install it on the RTX 3090 machine after setting up CUDA 12.x drivers.
- pynvml requires the NVIDIA Management Library (`libnvidia-ml.so`) to be present at runtime.
