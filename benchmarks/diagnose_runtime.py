"""Diagnose why the transfer-matrix solver is slow on a given machine.

The transfer-matrix solver spends ~75 % of its time in scipy.linalg.expm on a stack of small (2N x 2N, N <= 9)
complex matrices, i.e. thousands of tiny BLAS/LAPACK calls. On machines with many cores, the multithreaded BLAS
can add a large fixed overhead to every such call; the solver therefore limits BLAS to one thread while it runs
(blas_threads=1, the default). This script times the building blocks with the default BLAS threading and with BLAS
limited to one thread, and the solver with blas_threads=None (BLAS threading left unchanged) and blas_threads=1,
and prints the BLAS configuration.

Usage:   pip install threadpoolctl      (small pure-Python package, used here only for the diagnosis)
         python benchmarks/diagnose_runtime.py
Output:  printed and written to benchmarks/diagnose_runtime.txt
"""
import io, os, platform, sys, time
import numpy as np
import scipy
import scipy.linalg
import wrmodes as wr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_comsol_case as rc  # noqa: E402  (the COMSOL case of Figure 6)

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:
    sys.exit("please `pip install threadpoolctl` first")

out = io.StringIO()


def log(*a):
    print(*a)
    print(*a, file=out)


def best(fn, repeat=5):
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


log(f"machine : {platform.node()} / {platform.system()} {platform.release()} / {platform.processor()}")
log(f"cpus    : {os.cpu_count()} logical")
log(f"python  : {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, "
    f"wrmodes {getattr(wr, '__version__', '?')}")
for k in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    log(f"  {k} = {os.environ.get(k)}")
for p in threadpool_info():
    log(f"  {p.get('internal_api')} {p.get('version')}  threads={p.get('num_threads')}  "
        f"arch={p.get('architecture')}  ({os.path.basename(p.get('filepath', ''))})")

rng = np.random.default_rng(0)
cases = {}
for n in [6, 10, 18]:                                   # 2N for N = 3, 5, 9 coupled modes
    A = (rng.standard_normal((256, n, n)) + 1j * rng.standard_normal((256, n, n))) * 0.3
    cases[n] = A
B = cases[18][0]


def micro():
    r = {}
    for n, A in cases.items():
        r[f'expm 256 x ({n}x{n})'] = best(lambda: scipy.linalg.expm(A))
    r['1000 x matmul 18x18'] = best(lambda: [B @ B for _ in range(1000)])
    r['1000 x solve 18x18'] = best(lambda: [np.linalg.solve(B, B) for _ in range(1000)])
    return r


def full(blas_threads):
    """The 22-point sweep of the COMSOL case; blas_threads=None leaves the BLAS threading as it is
    (behaviour of the first 0.1.0 version), 1 is the default of the solver since then."""
    def sweep():
        for f in rc.FREQS:
            wr.solve_transfer_matrix(f, rc.SIDE_A, rc.SIDE_B, rc.MODE_IN, rc.LENGTH, rc.curvature,
                                     blas_threads=blas_threads)
    sweep()                                            # warm-up
    return best(sweep, repeat=3)


log("\n--- default BLAS threading")
m_def = micro(); f_def = full(None)
log("\n--- BLAS limited to 1 thread")
with threadpool_limits(limits=1):
    m_one = micro()
f_one = full(1)

log(f"\n{'':32s} {'default':>12s} {'1 thread':>12s} {'ratio':>8s}")
for k in m_def:
    log(f"{k:32s} {m_def[k]*1e3:10.2f} ms {m_one[k]*1e3:10.2f} ms {m_def[k]/m_one[k]:8.1f}")
log(f"{'COMSOL case, 22 points (solver)':32s} {f_def:10.3f} s  {f_one:10.3f} s  {f_def/f_one:8.1f}")
log("\n(reference, 2-core Linux VM, BLAS at 1 thread: expm 256 x (18x18) ~ 15 ms; COMSOL case ~ 0.15 s)")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diagnose_runtime.txt'), 'w') as fh:
    fh.write(out.getvalue())
