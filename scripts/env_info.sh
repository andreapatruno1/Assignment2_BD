#!/bin/bash
# Records the execution environment in results/environment.txt
cd "$(dirname "$0")/.."
{
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "== CPU =="; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|L1d|L2|L3|Hypervisor"
echo "== memory =="; free -m
echo "== OS =="; grep PRETTY_NAME /etc/os-release; uname -r
echo "== C =="; gcc --version | head -1; echo "flags: -O2 -fno-tree-vectorize -std=c11 -Wall -lm"
echo "== Python (inputs and analysis only) =="; python3 --version; python3 -c "import numpy, matplotlib; print('numpy', numpy.__version__, 'matplotlib', matplotlib.__version__)"
echo "== perf =="; echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid) (perf counters not readable, no root)"
} > results/environment.txt
cat results/environment.txt
