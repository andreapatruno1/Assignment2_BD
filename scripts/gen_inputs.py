"""Generates the dense input matrices (same generator, seed and format as Assignment 1).

Every matrix is stored as raw float64, little-endian, row-major (n*n*8 bytes, no header).
Generator: NumPy PCG64 (numpy.random.default_rng), seed = 42 + n, values uniform in [-1, 1).
A is drawn first, then B, from the same generator.

Usage: python3 scripts/gen_inputs.py [n1 n2 ...]   (default: the benchmark sizes)
"""
import os
import sys

import numpy as np

SIZES = [128, 256, 512, 1024, 1536, 2048, 3072]
DATA = os.environ.get("MATMUL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def generate(n):
    rng = np.random.default_rng(42 + n)
    a = rng.uniform(-1.0, 1.0, size=(n, n))
    b = rng.uniform(-1.0, 1.0, size=(n, n))
    a.astype("<f8").tofile(os.path.join(DATA, f"A_{n}.bin"))
    b.astype("<f8").tofile(os.path.join(DATA, f"B_{n}.bin"))


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    sizes = [int(x) for x in sys.argv[1:]] or SIZES
    for n in sizes:
        generate(n)
        print(f"written data/A_{n}.bin and data/B_{n}.bin")
