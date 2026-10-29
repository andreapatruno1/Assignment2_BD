# Assignment 2 — Optimised Dense and Sparse (CSR) Matrix Multiplication in C

Big Data, ULPGC, 2026-2027 · Andrea Patruno · Report: `Assignment02_Patruno_Report.pdf` (source `report/report.md`)

Questions: (A) how do loop interchange (i-k-j + hoisting) and cache blocking change dense multiplication vs the
Assignment 1 baseline (i-j-k)? (B) for which densities does CSR multiplication beat a dense method on the same inputs?

## Contents

| Path | What it is |
|---|---|
| `src/matmul2.c` | all methods (`naive` = Assignment 1 loop, `ikj`, `blocked`, `csr`, `convert`) + correctness tests |
| `scripts/build.sh` | `gcc -O2 -fno-tree-vectorize -std=c11 -Wall -o build/matmul2 src/matmul2.c -lm` |
| `scripts/gen_inputs.py` | dense inputs, Assignment 1 generator (PCG64, seed 42 + n, uniform [-1,1), float64) |
| `scripts/run_benchmarks.py` | protocol: 1 warm-up + 5 reps, 20 s budget per multiplication, one process per configuration |
| `scripts/analyze.py` | `results/summary_*.csv` and `results/fig/*.png` |
| `scripts/env_info.sh` | `results/environment.txt` |
| `results/raw_dense.csv`, `results/raw_sparse.csv` | every measured repetition |
| `results/limits.csv`, `results/correctness.csv` | timeouts; test results |

Sparse inputs are generated inside the program (splitmix64, seeds 2026/2027, Bernoulli(d) positions, uniform [-1,1) values).

## Exact commands (Linux / WSL2, gcc, python3 with numpy and matplotlib)

```bash
bash scripts/build.sh
./build/matmul2 test > results/correctness.csv          # must print ALL TESTS PASSED
python3 scripts/gen_inputs.py                            # data/A_n.bin, data/B_n.bin (~270 MB, MATMUL_DATA to change dir)
python3 scripts/run_benchmarks.py dense  --method naive ikj blocked --n 128 256 512 1024 1536 2048 3072
python3 scripts/run_benchmarks.py dense  --method blocked --n 1024 --bs 16 32 128 256 512
python3 scripts/run_benchmarks.py dense  --method blocked --n 2048 --bs 32 128 256
python3 scripts/run_benchmarks.py sparse --method csr ikj convert --n 2000 \
        --density 0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5 0.7 0.9 1.0
python3 scripts/run_benchmarks.py sparse --method csr convert --n 4000 --density 0.001 0.01 0.05
python3 scripts/run_benchmarks.py sparse --method ikj --n 4000 --density 0.01
python3 scripts/run_benchmarks.py sparse --method csr --n 10000 --density 0.001 0.005
bash scripts/env_info.sh
python3 scripts/analyze.py
python3 report/build_pdf.py Assignment02_Patruno_Report.pdf   # needs python markdown + playwright
```
Configurations after a timeout were not run (naive 3072, ikj 4000 at other densities). Rows are appended: delete
`results/raw_*.csv` and `results/limits.csv` before a full re-run.
