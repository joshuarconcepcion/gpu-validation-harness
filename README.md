# GPU Validation Harness

A testing and validation framework for the NVIDIA RTX 3090 Ti, built to programmatically stress the GPU, collect real-time hardware metrics, and validate performance against defined baselines.

## Overview

This project provides two modes of operation:

**CLI mode** — run a workload, collect metrics, validate against baseline, log results, and print a report to terminal:
```bash
python main.py --workload light
python main.py --workload medium
python main.py --workload stress
```

**API mode** — start a FastAPI server on port 8000 to trigger and retrieve benchmark runs over HTTP:
```bash
python main.py --serve
```

## Tech Stack

| Component | Library |
|---|---|
| GPU metrics | `nvidia-ml-py` (pynvml) |
| GPU compute workloads | `cupy` |
| API layer | `fastapi` + `uvicorn` |
| Persistence | `sqlite3` (stdlib) |
| Tests | `pytest` |

## Project Structure

```
gpu-validation-harness/
  src/
    instrumentation.py  # pynvml metric collection (utilization, memory, temp, power, clocks)
    workload.py         # GPU compute workloads (light / medium / stress)
    validator.py        # baseline comparison and pass/fail logic
    database.py         # SQLite schema and result logging
    api.py              # FastAPI routes
  tests/
    test_instrumentation.py
    test_validator.py
  main.py               # CLI entry point (--workload / --serve)
```

## Metrics Collected

Each benchmark run captures the following per reading:

- GPU core utilization (%)
- VRAM used / total (MB)
- Core temperature (°C)
- Power draw (W)
- Fan speed (%)
- Graphics clock (MHz)
- Memory clock (MHz)

## Status

| Module | Status |
|---|---|
| `instrumentation.py` | Complete |
| `workload.py` | In progress |
| `validator.py` | In progress |
| `database.py` | In progress |
| `api.py` | In progress |
| `main.py` | In progress |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install nvidia-ml-py fastapi uvicorn pytest cupy-cuda12x
```

> Requires Windows or Linux with NVIDIA drivers installed. CUDA 12.x required for cupy.

## Running Tests

```bash
pytest tests/
```
