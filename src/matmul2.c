/*
 * Assignment 2 - optimised dense multiplication and sparse (CSR) multiplication, in C.
 * Dense storage: one flat double array per matrix, row-major (element (i, j) at i*n + j), as in Assignment 1.
 *
 * Dense methods (all sequential, all O(n^3)):
 *   naive    i-j-k triple loop, copied unchanged from Assignment 1 (src/c/matmul.c) = baseline
 *   ikj      loop interchange i-k-j + hoisting of A[i][k] and of the row pointers
 *   blocked  i-k-j inside square tiles of side bs (cache blocking / tiling)
 * Sparse method:
 *   csr      CSR x CSR -> CSR, row-by-row (Gustavson) with a dense accumulator; own code, no library
 *
 * Build: gcc -O2 -fno-tree-vectorize -std=c11 -Wall -o build/matmul2 src/matmul2.c
 * Usage (one CSV row per measured repetition on stdout):
 *   matmul2 test                                      correctness tests
 *   matmul2 dense  METHOD n warmup reps datadir [bs]  dense methods on data/A_n.bin, data/B_n.bin
 *   matmul2 sparse METHOD n density warmup reps seed  METHOD = csr | ikj | convert, on generated sparse inputs
 * Environment: BUDGET_S = time budget in seconds for ONE timed multiplication (exit code 3 on timeout).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>

/* ---------------- dense kernels ---------------- */

/* Baseline: identical to Assignment 1. */
static void mm_naive(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; i++) {
        int row = i * n;
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += a[row + k] * b[k * n + j];
            c[row + j] = s;
        }
    }
}

/* Loop interchange: the inner loop walks B and C along a row (stride 1) instead of
 * walking B down a column (stride n). A[i][k] and the row pointers are hoisted. */
static void mm_ikj(const double *a, const double *b, double *c, int n) {
    memset(c, 0, (size_t) n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double *ci = c + (size_t) i * n;
        const double *ai = a + (size_t) i * n;
        for (int k = 0; k < n; k++) {
            const double aik = ai[k];
            const double *bk = b + (size_t) k * n;
            for (int j = 0; j < n; j++) ci[j] += aik * bk[j];
        }
    }
}

/* Cache blocking: same i-k-j body, restricted to bs x bs tiles so that the tile of B
 * reused by all rows i of a block stays in cache. Loop order ii, kk, jj keeps, for every
 * C[i][j], the same order of the k terms as the other methods. */
static void mm_blocked(const double *a, const double *b, double *c, int n, int bs) {
    memset(c, 0, (size_t) n * n * sizeof(double));
    for (int ii = 0; ii < n; ii += bs) {
        int ie = ii + bs < n ? ii + bs : n;
        for (int kk = 0; kk < n; kk += bs) {
            int ke = kk + bs < n ? kk + bs : n;
            for (int jj = 0; jj < n; jj += bs) {
                int je = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < ie; i++) {
                    double *ci = c + (size_t) i * n;
                    const double *ai = a + (size_t) i * n;
                    for (int k = kk; k < ke; k++) {
                        const double aik = ai[k];
                        const double *bk = b + (size_t) k * n;
                        for (int j = jj; j < je; j++) ci[j] += aik * bk[j];
                    }
                }
            }
        }
    }
}

/* ---------------- CSR ---------------- */

typedef struct { int n; long nnz, cap; int *rowptr; int *col; double *val; } csr_t;

static void csr_free(csr_t *m) { free(m->rowptr); free(m->col); free(m->val); }

static void csr_reserve(csr_t *m, long need) {
    if (need <= m->cap) return;
    long cap = m->cap ? m->cap : 1024;
    while (cap < need) cap *= 2;
    m->col = realloc(m->col, cap * sizeof(int));
    m->val = realloc(m->val, cap * sizeof(double));
    if (!m->col || !m->val) { fprintf(stderr, "out of memory\n"); exit(2); }
    m->cap = cap;
}

/* Dense row-major -> CSR (used to measure the conversion cost and in the tests). */
static csr_t dense_to_csr(const double *d, int n) {
    csr_t m = {n, 0, 0, malloc((n + 1) * sizeof(int)), NULL, NULL};
    long cnt = 0;
    for (long x = 0; x < (long) n * n; x++) if (d[x] != 0.0) cnt++;
    csr_reserve(&m, cnt > 0 ? cnt : 1);
    cnt = 0;
    for (int i = 0; i < n; i++) {
        m.rowptr[i] = (int) cnt;
        for (int j = 0; j < n; j++) {
            double v = d[(size_t) i * n + j];
            if (v != 0.0) { m.col[cnt] = j; m.val[cnt] = v; cnt++; }
        }
    }
    m.rowptr[n] = (int) cnt;
    m.nnz = cnt;
    return m;
}

/* C = A * B, row by row: row i of C = sum over nonzeros A[i][k] of A[i][k] * (row k of B).
 * acc[] is a dense accumulator of length n, mark[] records which columns of row i are already
 * present, so each product term costs O(1). Work = sum over nonzeros A[i][k] of nnz(row k of B).
 * Output allocation (growing col/val arrays) is part of the method and is inside the timed region. */
static csr_t csr_mult(const csr_t *A, const csr_t *B) {
    int n = A->n;
    csr_t C = {n, 0, 0, malloc((n + 1) * sizeof(int)), NULL, NULL};
    double *acc = calloc(n, sizeof(double));
    int *mark = malloc(n * sizeof(int)), *cols = malloc(n * sizeof(int));
    for (int j = 0; j < n; j++) mark[j] = -1;
    csr_reserve(&C, A->nnz + B->nnz + 1);
    long nnz = 0;
    for (int i = 0; i < n; i++) {
        int cnt = 0;
        for (int p = A->rowptr[i]; p < A->rowptr[i + 1]; p++) {
            int k = A->col[p];
            double aik = A->val[p];
            for (int q = B->rowptr[k]; q < B->rowptr[k + 1]; q++) {
                int j = B->col[q];
                if (mark[j] != i) { mark[j] = i; acc[j] = aik * B->val[q]; cols[cnt++] = j; }
                else acc[j] += aik * B->val[q];
            }
        }
        C.rowptr[i] = (int) nnz;
        csr_reserve(&C, nnz + cnt);
        for (int t = 0; t < cnt; t++) { C.col[nnz] = cols[t]; C.val[nnz] = acc[cols[t]]; nnz++; }
    }
    C.rowptr[n] = (int) nnz;
    C.nnz = nnz;
    free(acc); free(mark); free(cols);
    return C;   /* column indices inside a row are in first-touch order, not sorted */
}

static void csr_to_dense(const csr_t *m, double *d) {
    memset(d, 0, (size_t) m->n * m->n * sizeof(double));
    for (int i = 0; i < m->n; i++)
        for (int p = m->rowptr[i]; p < m->rowptr[i + 1]; p++) d[(size_t) i * m->n + m->col[p]] = m->val[p];
}

/* ---------------- sparse input generator ---------------- */
/* splitmix64; every entry is nonzero independently with probability = density (uniform positions),
 * nonzero values uniform in [-1, 1). The same seed gives the same logical matrix for dense and CSR. */
static uint64_t sm_state;
static uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double sm_unif(void) { return (sm_next() >> 11) * 0x1.0p-53; }   /* [0, 1) */

static void gen_sparse_dense(double *d, int n, double density, uint64_t seed) {
    sm_state = seed;
    for (long x = 0; x < (long) n * n; x++) d[x] = sm_unif() < density ? 2.0 * sm_unif() - 1.0 : 0.0;
}

/* Same sequence of random numbers, written directly in CSR (the dense matrix is never built). */
static csr_t gen_sparse_csr(int n, double density, uint64_t seed) {
    csr_t m = {n, 0, 0, malloc((n + 1) * sizeof(int)), NULL, NULL};
    sm_state = seed;
    long cnt = 0;
    csr_reserve(&m, (long) (density * n * n * 1.05) + 16);
    for (int i = 0; i < n; i++) {
        m.rowptr[i] = (int) cnt;
        for (int j = 0; j < n; j++)
            if (sm_unif() < density) {
                csr_reserve(&m, cnt + 1);
                m.col[cnt] = j; m.val[cnt] = 2.0 * sm_unif() - 1.0; cnt++;
            }
    }
    m.rowptr[n] = (int) cnt;
    m.nnz = cnt;
    return m;
}

/* ---------------- helpers ---------------- */

static double *load(const char *path, int n) {
    double *m = malloc((size_t) n * n * sizeof(double));
    FILE *f = fopen(path, "rb");
    if (!m || !f || fread(m, sizeof(double), (size_t) n * n, f) != (size_t) n * n) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    fclose(f);
    return m;
}

static long peak_rss_kb(void) {   /* VmHWM = peak resident set size of this process */
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long kb = -1;
    while (f && fgets(line, sizeof line, f))
        if (strncmp(line, "VmHWM:", 6) == 0) { kb = strtol(line + 6, NULL, 10); break; }
    if (f) fclose(f);
    return kb;
}

static double checksum(const double *c, long len) {
    double s = 0.0;
    for (long i = 0; i < len; i++) s += c[i];
    return s;
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Time budget: if BUDGET_S is set, every single timed run is armed with alarm(BUDGET_S);
 * a run that does not finish in time ends the process with exit code 3 (recorded as timeout). */
static unsigned budget_s = 0;
static void on_alarm(int sig) { (void) sig; static const char m[] = "TIMEOUT\n"; if (write(2, m, sizeof m - 1) < 0) _exit(4); _exit(3); }
static void arm(void) { if (budget_s) alarm(budget_s); }
static void disarm(void) { if (budget_s) alarm(0); }

static int batch_size(int n) {    /* small sizes are timed in batches (as in Assignment 1) */
    long b = (1L << 24) / ((long) n * n * n);
    return b < 1 ? 1 : (int) b;
}

static void run_dense(const char *method, const double *a, const double *b, double *c, int n, int bs) {
    if (!strcmp(method, "naive")) mm_naive(a, b, c, n);
    else if (!strcmp(method, "ikj")) mm_ikj(a, b, c, n);
    else if (!strcmp(method, "blocked")) mm_blocked(a, b, c, n, bs);
    else { fprintf(stderr, "unknown method %s\n", method); exit(1); }
}

/* ---------------- correctness tests ---------------- */

static double max_rel_err(const double *x, const double *ref, long len, double atol, double rtol, int *ok) {
    double worst = 0.0;
    *ok = 1;
    for (long i = 0; i < len; i++) {
        double e = fabs(x[i] - ref[i]);
        if (e > atol + rtol * fabs(ref[i])) *ok = 0;
        if (e > worst) worst = e;
    }
    return worst;
}

static int tests(void) {
    const double ATOL = 1e-12, RTOL = 1e-12;
    int all = 1, ok;
    printf("test,n,method,max_abs_diff,pass\n");
    /* 1. known 2x2 answer: [1 2;3 4] * [5 6;7 8] = [19 22;43 50] */
    double a2[] = {1, 2, 3, 4}, b2[] = {5, 6, 7, 8}, e2[] = {19, 22, 43, 50}, c2[4];
    const char *ms[] = {"naive", "ikj", "blocked"};
    for (int m = 0; m < 3; m++) {
        run_dense(ms[m], a2, b2, c2, 2, 1);
        double d = max_rel_err(c2, e2, 4, 0, 0, &ok);
        printf("known_2x2,2,%s,%g,%d\n", ms[m], d, ok); all &= ok;
    }
    /* 2. identity and zero matrices, random A, sizes not multiple of the block size */
    int sizes[] = {1, 7, 67, 130};
    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        long len = (long) n * n;
        double *a = malloc(len * 8), *id = calloc(len, 8), *z = calloc(len, 8), *c = malloc(len * 8),
               *ref = malloc(len * 8), *b = malloc(len * 8);
        gen_sparse_dense(a, n, 1.0, 11 + n);
        gen_sparse_dense(b, n, 1.0, 99 + n);
        for (int i = 0; i < n; i++) id[(size_t) i * n + i] = 1.0;
        mm_naive(a, b, ref, n);
        for (int m = 0; m < 3; m++) {
            run_dense(ms[m], a, id, c, n, 32);
            double d = max_rel_err(c, a, len, 0, 0, &ok);
            printf("A*I=A,%d,%s,%g,%d\n", n, ms[m], d, ok); all &= ok;
            run_dense(ms[m], a, z, c, n, 32);
            d = max_rel_err(c, z, len, 0, 0, &ok);
            printf("A*0=0,%d,%s,%g,%d\n", n, ms[m], d, ok); all &= ok;
            run_dense(ms[m], a, b, c, n, 32);
            d = max_rel_err(c, ref, len, ATOL, RTOL, &ok);
            printf("random_vs_naive,%d,%s,%g,%d\n", n, ms[m], d, ok); all &= ok;
        }
        free(a); free(id); free(z); free(c); free(ref); free(b);
    }
    /* 3. CSR vs dense naive on the same logical inputs, several densities */
    double dens[] = {0.0, 0.01, 0.1, 0.5, 1.0};
    int n = 150;
    long len = (long) n * n;
    for (int t = 0; t < 5; t++) {
        double *a = malloc(len * 8), *b = malloc(len * 8), *ref = malloc(len * 8), *c = malloc(len * 8);
        gen_sparse_dense(a, n, dens[t], 5);
        gen_sparse_dense(b, n, dens[t], 6);
        mm_naive(a, b, ref, n);
        csr_t A = gen_sparse_csr(n, dens[t], 5), B = gen_sparse_csr(n, dens[t], 6);
        csr_t A2 = dense_to_csr(a, n);
        int same = A.nnz == A2.nnz && !memcmp(A.col, A2.col, A.nnz * sizeof(int))
                   && !memcmp(A.val, A2.val, A.nnz * sizeof(double));
        printf("csr_generator_equals_dense_generator,%d,density=%g,0,%d\n", n, dens[t], same); all &= same;
        csr_t C = csr_mult(&A, &B);
        csr_to_dense(&C, c);
        double d = max_rel_err(c, ref, len, ATOL, RTOL, &ok);
        printf("csr_vs_naive,%d,density=%g,%g,%d\n", n, dens[t], d, ok); all &= ok;
        csr_free(&A); csr_free(&B); csr_free(&A2); csr_free(&C);
        free(a); free(b); free(ref); free(c);
    }
    fprintf(stderr, all ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    return all ? 0 : 1;
}

/* ---------------- main ---------------- */

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    if (!strcmp(argv[1], "test")) return tests();
    if (getenv("BUDGET_S")) { budget_s = (unsigned) atoi(getenv("BUDGET_S")); signal(SIGALRM, on_alarm); }

    if (!strcmp(argv[1], "dense")) {
        const char *method = argv[2];
        int n = atoi(argv[3]), warmup = atoi(argv[4]), reps = atoi(argv[5]);
        int bs = argc > 7 ? atoi(argv[7]) : 64;
        char pa[1024], pb[1024];
        snprintf(pa, sizeof pa, "%s/A_%d.bin", argv[6], n);
        snprintf(pb, sizeof pb, "%s/B_%d.bin", argv[6], n);
        double *a = load(pa, n), *b = load(pb, n);              /* outside the timed region */
        double *c = calloc((size_t) n * n, sizeof(double));     /* output allocated outside */
        int batch = batch_size(n);
        for (int r = -warmup; r < reps; r++) {
            arm();
            double t0 = now_s();
            for (int i = 0; i < batch; i++) run_dense(method, a, b, c, n, bs);
            double t1 = now_s();
            disarm();
            double chk = checksum(c, (long) n * n);                /* consumes the result */
            if (r >= 0) printf("%s,%d,%d,%d,%d,%.9f,%.12e,%ld\n", method, n, bs, r, batch, (t1 - t0) / batch, chk, peak_rss_kb());
            fflush(stdout);
        }
        return 0;
    }

    if (!strcmp(argv[1], "sparse")) {
        const char *method = argv[2];
        int n = atoi(argv[3]), warmup = atoi(argv[5]), reps = atoi(argv[6]);
        double density = atof(argv[4]);
        uint64_t seed = strtoull(argv[7], NULL, 10);
        /* output columns: method,n,density,rep,time_s,nnzA,nnzB,nnzC,checksum,peak_rss_kb */
        if (!strcmp(method, "csr")) {
            csr_t A = gen_sparse_csr(n, density, seed), B = gen_sparse_csr(n, density, seed + 1);
            for (int r = -warmup; r < reps; r++) {
                arm();
                double t0 = now_s();
                csr_t C = csr_mult(&A, &B);
                double t1 = now_s();
                disarm();
                if (r >= 0) printf("csr,%d,%g,%d,%.9f,%ld,%ld,%ld,%.12e,%ld\n", n, density, r, t1 - t0,
                                   A.nnz, B.nnz, C.nnz, checksum(C.val, C.nnz), peak_rss_kb());
                fflush(stdout);
                csr_free(&C);
            }
        } else if (!strcmp(method, "ikj") || !strcmp(method, "convert")) {
            size_t len = (size_t) n * n;
            double *a = malloc(len * 8), *b = malloc(len * 8), *c = calloc(len, 8);
            gen_sparse_dense(a, n, density, seed);
            gen_sparse_dense(b, n, density, seed + 1);
            long nzA = 0, nzB = 0;
            for (size_t x = 0; x < len; x++) { nzA += a[x] != 0.0; nzB += b[x] != 0.0; }
            for (int r = -warmup; r < reps; r++) {
                if (!strcmp(method, "ikj")) {
                    arm();
                    double t0 = now_s();
                    mm_ikj(a, b, c, n);
                    double t1 = now_s();
                    disarm();
                    long nzC = 0;
                    for (size_t x = 0; x < len; x++) nzC += c[x] != 0.0;
                    if (r >= 0) printf("ikj,%d,%g,%d,%.9f,%ld,%ld,%ld,%.12e,%ld\n", n, density, r, t1 - t0,
                                       nzA, nzB, nzC, checksum(c, len), peak_rss_kb());
                } else {   /* conversion of both dense inputs to CSR, measured separately */
                    double t0 = now_s();
                    csr_t A = dense_to_csr(a, n), B = dense_to_csr(b, n);
                    double t1 = now_s();
                    if (r >= 0) printf("convert,%d,%g,%d,%.9f,%ld,%ld,0,%.12e,%ld\n", n, density, r, t1 - t0,
                                       A.nnz, B.nnz, checksum(A.val, A.nnz), peak_rss_kb());
                    csr_free(&A); csr_free(&B);
                }
                fflush(stdout);
            }
        }
        return 0;
    }
    return 1;
}
