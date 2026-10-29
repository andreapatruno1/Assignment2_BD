"""Summary tables and figures from results/raw_dense.csv and results/raw_sparse.csv.

Central value = median of the 5 measured repetitions, variability = IQR (Q3 - Q1, numpy linear quantiles).
Memory: peak_rss_MB = VmHWM of the process (MiB); storage_MB = analytical size of the arrays (MiB):
  dense  3 * 8 n^2                         (A, B, C)
  CSR    sum over A, B, C of (12 nnz + 4 (n + 1))  (8-byte value + 4-byte column per nonzero, int row pointers)
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "fig")
MB = 1024 ** 2
COL = {"naive": "#4e79a7", "ikj": "#f28e2b", "blocked": "#59a14f", "csr": "#e15759", "convert": "#76b7b2"}


def read(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def stats(v):
    v = np.array(v, float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return med, q1, q3, q3 - q1, v.min(), v.max(), len(v)


def dense():
    g = defaultdict(list)
    rss = {}
    for r in read("raw_dense.csv"):
        key = (r["method"], int(r["n"]), int(r["bs"]))
        g[key].append(float(r["time_s"]))
        rss[key] = int(r["peak_rss_kb"]) / 1024
    out = []
    for (m, n, bs), v in sorted(g.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        med, q1, q3, iqr, mn, mx, k = stats(v)
        out.append(dict(method=m, n=n, bs=bs, reps=k, median_s=med, q1_s=q1, q3_s=q3, iqr_s=iqr, min_s=mn, max_s=mx,
                        ns_per_iteration=med / n ** 3 * 1e9, peak_rss_MB=rss[(m, n, bs)], storage_MB=3 * 8 * n * n / MB))
    base = {r["n"]: r["median_s"] for r in out if r["method"] == "naive"}
    for r in out:
        r["speedup_vs_naive"] = base[r["n"]] / r["median_s"] if r["n"] in base else float("nan")
    write("summary_dense.csv", out)
    return out


def sparse():
    g = defaultdict(list)
    info = {}
    for r in read("raw_sparse.csv"):
        key = (r["method"], int(r["n"]), float(r["density"]))
        g[key].append(float(r["time_s"]))
        info[key] = r
    out = []
    for (m, n, d), v in sorted(g.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
        med, q1, q3, iqr, mn, mx, k = stats(v)
        i = info[(m, n, d)]
        nA, nB, nC = int(i["nnzA"]), int(i["nnzB"]), int(i["nnzC"])
        if m == "csr":
            nCc = nC
        else:
            nCc = int(info[("csr", n, d)]["nnzC"]) if ("csr", n, d) in info else nC
        storage = 3 * 8 * n * n if m == "ikj" else sum(12 * x + 4 * (n + 1) for x in (nA, nB, nCc))
        if m == "convert":
            storage = 2 * 8 * n * n + sum(12 * x + 4 * (n + 1) for x in (nA, nB))
        out.append(dict(method=m, n=n, density_pct=d * 100, reps=k, median_s=med, q1_s=q1, q3_s=q3, iqr_s=iqr,
                        min_s=mn, max_s=mx, nnzA=nA, nnzB=nB, nnzC=nC if m != "convert" else "",
                        output_density_pct=(100 * nC / n / n) if m != "convert" else "",
                        peak_rss_MB=int(i["peak_rss_kb"]) / 1024, storage_MB=storage / MB))
    med = {(r["method"], r["n"], r["density_pct"]): r["median_s"] for r in out}
    for r in out:
        k = ("ikj", r["n"], r["density_pct"])
        r["ikj_over_csr"] = med[k] / r["median_s"] if r["method"] == "csr" and k in med else ""
    write("summary_sparse.csv", out)
    return out


def write(name, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def figures(dn, sp):
    os.makedirs(FIG, exist_ok=True)
    # 1. dense time vs n (bs = 64 for blocked)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in ["naive", "ikj", "blocked"]:
        rs = [r for r in dn if r["method"] == m and r["bs"] in (0, 64)]
        n = [r["n"] for r in rs]
        ax[0].errorbar(n, [r["median_s"] for r in rs], yerr=[[r["median_s"] - r["q1_s"] for r in rs], [r["q3_s"] - r["median_s"] for r in rs]],
                       marker="o", capsize=3, color=COL[m], label=m if m != "blocked" else "blocked (bs=64)")
        ax[1].plot(n, [r["ns_per_iteration"] for r in rs], marker="o", color=COL[m], label=m if m != "blocked" else "blocked (bs=64)")
    ref = np.array([128, 3072])
    ax[0].plot(ref, 0.0008 * (ref / 128) ** 3, "k:", label="∝ n³")
    ax[0].axhline(20, color="grey", ls="--", lw=1); ax[0].text(130, 23, "20 s budget", fontsize=8, color="grey")
    ax[0].set(xscale="log", yscale="log", xlabel="n", ylabel="time per multiplication [s] (median, IQR)", title="Dense: runtime")
    ax[1].set(xscale="log", xlabel="n", ylabel="ns per inner iteration (median / n³)", title="Dense: cost per inner iteration")
    for a in ax:
        a.set_xticks([128, 256, 512, 1024, 2048, 3072]); a.set_xticklabels(["128", "256", "512", "1024", "2048", "3072"]); a.grid(alpha=.3)
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "dense_time.png"), dpi=160); plt.close(fig)

    # 2. block size sweep
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for n, c in [(1024, "#59a14f"), (2048, "#2f6b2a")]:
        rs = sorted([r for r in dn if r["method"] == "blocked" and r["n"] == n], key=lambda r: r["bs"])
        ikj = [r["median_s"] for r in dn if r["method"] == "ikj" and r["n"] == n][0]
        ax.errorbar([r["bs"] for r in rs], [r["median_s"] / ikj for r in rs],
                    yerr=[[(r["median_s"] - r["q1_s"]) / ikj for r in rs], [(r["q3_s"] - r["median_s"]) / ikj for r in rs]],
                    marker="o", capsize=3, color=c, label=f"n = {n}")
    ax.axhline(1, color=COL["ikj"], ls="--", lw=1, label="ikj (no blocking)")
    ax.set(xscale="log", xlabel="block size bs", ylabel="time / time of ikj", title="Blocked: effect of the block size")
    ax.set_xticks([16, 32, 64, 128, 256, 512]); ax.set_xticklabels(["16", "32", "64", "128", "256", "512"])
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "block_size.png"), dpi=160); plt.close(fig)

    # 3. sparse vs dense at n = 2000: time and memory vs density
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for m, lab in [("csr", "CSR multiplication"), ("ikj", "dense ikj"), ("convert", "dense→CSR conversion (A and B)")]:
        rs = [r for r in sp if r["method"] == m and r["n"] == 2000]
        ax[0].errorbar([r["density_pct"] for r in rs], [r["median_s"] for r in rs],
                       yerr=[[r["median_s"] - r["q1_s"] for r in rs], [r["q3_s"] - r["median_s"] for r in rs]],
                       marker="o", capsize=3, color=COL[m], label=lab)
    ax[0].set(xscale="log", yscale="log", xlabel="input density [% nonzero]", ylabel="time [s] (median, IQR)", title="n = 2000: runtime")
    for m, lab in [("csr", "CSR"), ("ikj", "dense ikj")]:
        rs = [r for r in sp if r["method"] == m and r["n"] == 2000]
        x = [r["density_pct"] for r in rs]
        ax[1].plot(x, [r["peak_rss_MB"] for r in rs], marker="o", color=COL[m], label=f"{lab}: peak RSS")
        ax[1].plot(x, [r["storage_MB"] for r in rs], ls="--", color=COL[m], label=f"{lab}: array storage (A+B+C)")
    ax[1].set(xscale="log", xlabel="input density [% nonzero]", ylabel="MiB", title="n = 2000: memory")
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sparse.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    d, s = dense(), sparse()
    figures(d, s)
    print("written results/summary_dense.csv, results/summary_sparse.csv, results/fig/*.png")
