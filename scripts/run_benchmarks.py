"""Runs one or more benchmark configurations of build/matmul2 and appends the raw rows to results/.

Protocol (fixed before running):
  warm-up      1 run (not reported); repetitions 5 measured runs per configuration
  time budget  20 s for ONE multiplication (BUDGET_S, enforced inside the program with alarm());
               a timeout is written to results/limits.csv, never as a time
  memory       budget 3 GB peak resident memory (VM has 3.8 GiB); metric VmHWM of the process
  each configuration runs in its own process, so VmHWM belongs to that method and size only

Usage:
  python3 scripts/run_benchmarks.py dense  --method naive --n 512 1024 [--bs 64]
  python3 scripts/run_benchmarks.py sparse --method csr ikj convert --n 2000 --density 0.001 0.01
Outputs (appended):
  results/raw_dense.csv   method,n,bs,rep,batch,time_s,checksum,peak_rss_kb
  results/raw_sparse.csv  method,n,density,rep,time_s,nnzA,nnzB,nnzC,checksum,peak_rss_kb
  results/limits.csv      part,method,n,bs_or_density,event,limit
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BIN = os.path.join(ROOT, "build", "matmul2")
RES = os.path.join(ROOT, "results")
DATA = os.environ.get("MATMUL_DATA", os.path.join(ROOT, "data"))
WARMUP, REPS, BUDGET_S, SEED = 1, 5, 20, 2026
HEAD = {"raw_dense.csv": "method,n,bs,rep,batch,time_s,checksum,peak_rss_kb",
        "raw_sparse.csv": "method,n,density,rep,time_s,nnzA,nnzB,nnzC,checksum,peak_rss_kb",
        "limits.csv": "part,method,n,bs_or_density,event,limit"}


def append(name, lines):
    path = os.path.join(RES, name)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(HEAD[name] + "\n")
        for line in lines:
            f.write(line + "\n")


def run(cmd, part, method, n, param):
    env = dict(os.environ, BUDGET_S=str(BUDGET_S))
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    rows = [r for r in p.stdout.splitlines() if r.strip()]
    if p.returncode == 3:
        append("limits.csv", [f"{part},{method},{n},{param},timeout (single multiplication not finished),{BUDGET_S} s"])
        print(f"{part} {method} n={n} {param}: TIMEOUT", file=sys.stderr)
    elif p.returncode != 0:
        append("limits.csv", [f"{part},{method},{n},{param},failure exit code {p.returncode},-"])
        print(p.stderr, file=sys.stderr)
    return rows, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["dense", "sparse"])
    ap.add_argument("--method", nargs="+", required=True)
    ap.add_argument("--n", nargs="+", type=int, required=True)
    ap.add_argument("--bs", nargs="+", type=int, default=[64])
    ap.add_argument("--density", nargs="+", type=float, default=[])
    a = ap.parse_args()
    for method in a.method:
        for n in a.n:
            if a.part == "dense":
                for bs in (a.bs if method == "blocked" else [0]):
                    rows, rc = run([BIN, "dense", method, str(n), str(WARMUP), str(REPS), DATA, str(bs or 64)],
                                   "dense", method, n, bs)
                    if rc == 0:
                        append("raw_dense.csv", [r.replace(f",{n},64,", f",{n},{bs},", 1) if not bs else r for r in rows])
                    print(method, n, bs, "rc", rc, rows[-1:] if rows else "")
            else:
                for d in a.density:
                    rows, rc = run([BIN, "sparse", method, str(n), repr(d), str(WARMUP), str(REPS), str(SEED)],
                                   "sparse", method, n, d)
                    if rc == 0:
                        append("raw_sparse.csv", rows)
                    print(method, n, d, "rc", rc, rows[-1:] if rows else "")


if __name__ == "__main__":
    main()
