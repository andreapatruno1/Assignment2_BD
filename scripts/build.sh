#!/bin/bash
# Builds build/matmul2. -fno-tree-vectorize keeps all methods scalar (SIMD is the optional topic of Assignment 3).
set -e
cd "$(dirname "$0")/.."
mkdir -p build
gcc -O2 -fno-tree-vectorize -std=c11 -Wall -o build/matmul2 src/matmul2.c -lm
echo "build OK"
