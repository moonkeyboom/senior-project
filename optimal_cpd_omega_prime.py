"""
Optimal 1-D k-means using Omega-prime (Ω′) as the objective function
====================================================================

Instead of minimising within-cluster sum of squares (withinss, as in
Ckmeans.1d.dp — Wang & Song, 2011), this module searches for the 1-D
partition (norm-referenced CPD) that MAXIMISES the conditional-unbiasedness
metric Ω′ defined in:

    Banditwattanawong & Masdisornchote (2025),
    "Unbiased machine learning-assisted approach for conditional
     discretization of human performances", PeerJ CS, DOI 10.7717/peerj-cs.2804
    (Eq. 2)

Two search strategies are provided and benchmarked:
  (a) EXHAUSTIVE  — evaluate every contiguous partition C(n-1, k-1); the
                    global optimum of Ω′ (ground truth). Feasible for small n,k.
  (b) DP-BASED    — Ckmeans.1d.dp dynamic programme gives the SSE-optimal
                    partition for each k; we then pick the k whose SSE-optimal
                    partition has the highest Ω′. Fast (O(n²k)) but heuristic
                    w.r.t. Ω′ (SSE-optimal ≠ Ω′-optimal in general).

------------------------------------------------------------------------------
IMPORTANT — how Ω′ is actually computed (reverse-engineered from the paper's
worked examples, because the typeset Eq. 2 is ambiguous after PDF extraction):

  Ω′ = Ω1 · Ω2 · Ω3

  Ω1 = 1 − |Θ − θ| / Θ               if Θ ≥ 1   else 1     (clamped to [0,1])
       θ = number of UNASSIGNED PRLs = |L| − N
       Θ = number of gaps in {γu, δ1..δ_{N-1}, γl} that are ≥ the widest PVI
       Θ is how many PRLs OUGHT to stay unassigned, so the penalty is
       two-sided: θ < Θ and θ > Θ are both punished. NOT θ/Θ (see below).

  Ω2 = (Σδi − ΣD_min) / (ΣD_max − ΣD_min)   if N ≥ 3   else 1
       δi   = gap between clusters i and i+1  (min(cluster_i) − max(cluster_{i+1}))
       D_min/D_max = i-th narrowest / widest gap between ADJACENT performance
                     values (boundary gaps γu, γl are EXCLUDED here)

  Ω3 = 1 / (1 + σ)                   if N ≥ 2   else 1
       σ = SAMPLE standard deviation (ddof=1) of the PVIs of all clusters

  where, on descending-sorted values:
       PVI(cluster) = max(cluster) − min(cluster)
       γu = U − max(all values)     (U = upper bound, e.g. 100)
       γl = min(all values) − L     (L = lower bound, e.g. 0)
       N  = number of clusters (= number of assigned PRLs)
       |L| = number of ASSIGNABLE PRLs (grades), a fixed input

This reproduction is verified against Table 3 of the paper:
       A1 (N=4) → Ω′ ≈ 0.08 ,   A2 (N=3) → Ω′ ≈ 0.22
(see test_paper_examples()).
"""

from __future__ import annotations

import time
from itertools import combinations
from dataclasses import dataclass, field

import numpy as np


# ===========================================================================
# Core data structures
# ===========================================================================

class SearchCancelled(Exception):
    """Raised when a `monitor` callback aborts an in-progress search."""


# How often `exhaustive_best_for_k` calls its monitor. Small enough that an
# abort lands within a few milliseconds, large enough that the callback cost
# stays under a rounding error of the search itself.
MONITOR_INTERVAL = 8192


@dataclass
class CPDResult:
    """Outcome of one partition, with its Ω′ decomposition."""
    cuts: tuple            # internal boundary indices (start-of-new-cluster)
    N: int                 # number of clusters / assigned PRLs
    omega_prime: float
    omega1: float
    omega2: float
    omega3: float
    theta: int             # unassigned PRLs
    Theta: int             # gaps >= widest PVI
    sigma: float
    pvis: list = field(default_factory=list)
    deltas: list = field(default_factory=list)
    cluster_bounds: list = field(default_factory=list)  # [(hi, lo), ...]


# ===========================================================================
# 1) Ω′  — the objective function
# ===========================================================================

def _clusters_from_cuts(values_desc, cuts):
    """Split descending-sorted values into contiguous clusters at `cuts`.

    `cuts` = iterable of internal indices in [1, n-1] where a NEW cluster
    starts. Returns list of (hi, lo, start, stop) per cluster.
    """
    n = len(values_desc)
    bounds = [0] + sorted(int(c) for c in cuts) + [n]
    clusters = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = values_desc[a:b]
        clusters.append((float(seg[0]), float(seg[-1]), a, b))  # sorted desc: seg[0]=max
    return clusters


def calculate_omega_prime(values_desc, cuts, num_labels, U, L, ddof=1,
                          return_details=False):
    """Compute Ω′ for a contiguous partition of descending-sorted values.

    Parameters
    ----------
    values_desc : 1-D array, sorted DESCENDING
    cuts        : internal boundary indices (start-of-new-cluster), len = N-1
    num_labels  : |L|, number of ASSIGNABLE PRLs (grades)
    U, L        : upper / lower performance bounds (e.g. 100, 0)
    ddof        : 1 → sample std for σ (matches the paper's 5.07 / 2.89)
    """
    v = np.asarray(values_desc, dtype=float)
    n = len(v)
    clusters = _clusters_from_cuts(v, cuts)
    N = len(clusters)

    pvis = np.array([hi - lo for (hi, lo, a, b) in clusters], dtype=float)
    widest_pvi = pvis.max()

    # boundary + inter-cluster gaps  <γu, δ1..δ_{N-1}, γl>
    gamma_u = U - clusters[0][0]        # U − max value
    gamma_l = clusters[-1][1] - L       # min value − L
    deltas = np.array([clusters[i][1] - clusters[i + 1][0]  # min(hi PRL) − max(lo PRL)
                       for i in range(N - 1)], dtype=float)
    boundary_gaps = np.concatenate([[gamma_u], deltas, [gamma_l]])

    # ---- Ω1 = 1 − |Θ−θ|/Θ -------------------------------------------------
    # Θ is the number of PRLs that OUGHT to remain unassigned; θ is how many
    # actually are. The penalty is two-sided: unassigning too FEW *or* too MANY
    # both reduce Ω1. (Using θ/Θ is wrong — it only penalises one side and
    # saturates at 1.0 for every θ ≥ Θ. Both forms agree when θ ≤ Θ, which is
    # why the paper's Table 3 examples alone cannot tell them apart.)
    theta = num_labels - N                                   # unassigned PRLs
    Theta = int(np.sum(boundary_gaps >= widest_pvi))         # gaps ≥ widest PVI
    if Theta >= 1:
        omega1 = 1.0 - abs(Theta - theta) / Theta
        omega1 = min(max(omega1, 0.0), 1.0)                  # range [0,1]
    else:
        omega1 = 1.0

    # ---- Ω2 --------------------------------------------------------------
    if N >= 3:
        internal_gaps = np.abs(np.diff(v))                   # adjacent-value gaps only
        sorted_g = np.sort(internal_gaps)
        m = N - 1
        D_min = sorted_g[:m].sum()
        D_max = sorted_g[::-1][:m].sum()
        denom = D_max - D_min
        omega2 = 1.0 if denom == 0 else (deltas.sum() - D_min) / denom
        omega2 = min(max(omega2, 0.0), 1.0)
    else:
        omega2 = 1.0

    # ---- Ω3 = 1/(1+σ) ----------------------------------------------------
    if N >= 2:
        sigma = float(np.std(pvis, ddof=ddof)) if len(pvis) > 1 else 0.0
        omega3 = 1.0 / (1.0 + sigma)
    else:
        sigma = 0.0
        omega3 = 1.0

    omega_prime = omega1 * omega2 * omega3

    if not return_details:
        return omega_prime

    return CPDResult(
        cuts=tuple(sorted(int(c) for c in cuts)), N=N,
        omega_prime=omega_prime, omega1=omega1, omega2=omega2, omega3=omega3,
        theta=theta, Theta=Theta, sigma=sigma,
        pvis=[round(p, 4) for p in pvis.tolist()],
        deltas=[round(d, 4) for d in deltas.tolist()],
        cluster_bounds=[(hi, lo) for (hi, lo, a, b) in clusters],
    )


# ===========================================================================
# 2a) EXHAUSTIVE optimal-Ω′ search
# ===========================================================================

def exhaustive_best_for_k(values_desc, k, num_labels, U, L, ddof=1, monitor=None):
    """Best Ω′ partition into exactly k contiguous clusters (brute force).

    Parameters
    ----------
    monitor : callable(examined: int) -> bool, optional
        Called every MONITOR_INTERVAL candidates with the running count. Return
        False to abort; the search then raises SearchCancelled instead of
        returning. Lets a caller running this in a worker thread stop it
        promptly rather than orphaning a CPU-bound thread it can no longer use.
    """
    n = len(values_desc)
    if k < 1 or k > n:
        return None
    if k == 1:
        return calculate_omega_prime(values_desc, (), num_labels, U, L, ddof, True)
    best = None
    examined = 0
    for cuts in combinations(range(1, n), k - 1):   # C(n-1, k-1)
        res = calculate_omega_prime(values_desc, cuts, num_labels, U, L, ddof, True)
        if best is None or res.omega_prime > best.omega_prime:
            best = res
        examined += 1
        if monitor is not None and examined % MONITOR_INTERVAL == 0:
            if not monitor(examined):
                raise SearchCancelled(f"exhaustive search aborted after {examined:,} candidates")
    return best


def exhaustive_optimal(values_desc, num_labels, U, L, k=None,
                       k_min=2, k_max=None, ddof=1):
    """Global optimum of Ω′.

    k is not None  → fixed-k mode (search partitions into exactly k clusters)
    k is None      → search k = k_min .. k_max and return the overall best.
    """
    n = len(values_desc)
    if k is not None:
        return exhaustive_best_for_k(values_desc, k, num_labels, U, L, ddof)

    k_max = num_labels if k_max is None else min(k_max, num_labels, n)
    best = None
    per_k = {}
    for kk in range(k_min, k_max + 1):
        res = exhaustive_best_for_k(values_desc, kk, num_labels, U, L, ddof)
        if res is None:
            continue
        per_k[kk] = res
        if best is None or res.omega_prime > best.omega_prime:
            best = res
    return best, per_k


# ===========================================================================
# 2b) DP-BASED search  (Ckmeans.1d.dp for SSE, then pick k by Ω′)
# ===========================================================================

def ckmeans_1d_dp(values_asc, k):
    """Optimal 1-D k-means (min withinss) via dynamic programming, O(n²k).

    Returns the list of cluster index-boundaries (start of each cluster) in
    ASCENDING order. Follows Wang & Song (2011): D[i,m] recurrence with a
    constant-time within-cluster SSE via prefix sums.
    """
    x = np.asarray(values_asc, dtype=float)
    n = len(x)
    if k >= n:
        return list(range(n))          # every point its own cluster
    if k <= 1:
        return [0]

    # prefix sums for O(1) segment SSE
    pre = np.concatenate([[0.0], np.cumsum(x)])
    pre2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def sse(a, b):                      # SSE of x[a..b] inclusive (0-based)
        cnt = b - a + 1
        s = pre[b + 1] - pre[a]
        s2 = pre2[b + 1] - pre2[a]
        return s2 - s * s / cnt

    INF = float("inf")
    D = np.full((k + 1, n + 1), INF)
    B = np.zeros((k + 1, n + 1), dtype=int)
    D[0, 0] = 0.0
    for m in range(1, k + 1):
        for i in range(1, n + 1):
            if m > i:
                continue
            for j in range(m, i + 1):          # first index of cluster m (1-based)
                prev = D[m - 1, j - 1]
                if prev == INF:
                    continue
                cost = prev + sse(j - 1, i - 1)
                if cost < D[m, i]:
                    D[m, i] = cost
                    B[m, i] = j - 1            # 0-based start of cluster m

    # backtrack cluster starts
    starts = []
    i = n
    for m in range(k, 0, -1):
        s = B[m, i]
        starts.append(s)
        i = s
    return sorted(starts)                       # ascending starts, starts[0]==0


def dp_best(values_desc, num_labels, U, L, k=None, k_min=2, k_max=None, ddof=1):
    """DP-based CPD: SSE-optimal partition per k, choose k maximising Ω′."""
    v = np.asarray(values_desc, dtype=float)
    n = len(v)
    v_asc = v[::-1]                              # DP works on ascending input

    def cuts_for_k(kk):
        starts_asc = ckmeans_1d_dp(v_asc, kk)    # ascending-space starts
        # map ascending cluster starts -> descending internal cut positions
        # cluster sizes are order-independent; rebuild boundaries in desc space
        sizes = []
        s = starts_asc + [n]
        for a, b in zip(s[:-1], s[1:]):
            sizes.append(b - a)
        sizes = sizes[::-1]                       # reverse for descending order
        cuts, acc = [], 0
        for sz in sizes[:-1]:
            acc += sz
            cuts.append(acc)
        return tuple(cuts)

    if k is not None:
        return calculate_omega_prime(v, cuts_for_k(k), num_labels, U, L, ddof, True)

    k_max = num_labels if k_max is None else min(k_max, num_labels, n)
    best, per_k = None, {}
    for kk in range(k_min, k_max + 1):
        res = calculate_omega_prime(v, cuts_for_k(kk), num_labels, U, L, ddof, True)
        per_k[kk] = res
        if best is None or res.omega_prime > best.omega_prime:
            best = res
    return best, per_k


# ===========================================================================
# 2c) WGF-CPD — reduce-from-|L| search (paper's Algorithm 1)
# ===========================================================================

def _cuts_from_widest_gaps(values_desc, n_clusters):
    """Cut at the (n_clusters − 1) widest gaps between adjacent values.

    Ties are broken by preferring the earlier (higher-score) gap, matching the
    paper's "if there are multiple identical widest gaps, the one leading to
    more uniform PVIs" only loosely — exact tie-breaking is left to the caller.
    """
    v = np.asarray(values_desc, dtype=float)
    n = len(v)
    if n_clusters <= 1:
        return ()
    gaps = v[:-1] - v[1:]                       # gap before index i+1
    # order by gap size desc, then by position asc for stable ties
    idx = sorted(range(len(gaps)), key=lambda i: (-gaps[i], i))
    chosen = sorted(i + 1 for i in idx[:n_clusters - 1])
    return tuple(chosen)


def wgf_reduce(values_desc, num_labels, U, L, ddof=1, max_iter=None):
    """Faithful reduce-from-|L| CPD (paper's Algorithm 1, WGF-CPD).

    Starts by assigning ALL |L| labels via the |L|−1 widest gaps, then—while
    Requirement 1 is unmet (Θ ≥ 1)—sacrifices one label per iteration and
    re-partitions into one fewer cluster. This is the loop the manual
    spreadsheet performs; it never searches down to the degenerate N=2
    optimum the way a free sweep over k does.

    The loop RUNS UNTIL Θ = 0, i.e. until no gap still exceeds the widest PVI.
    The returned result is that terminal Θ=0 state — Requirement 1 is a
    constraint to satisfy, not a score to maximise, so we do NOT return the
    iteration with the highest Ω′. (An earlier iteration may well score higher;
    it is reported in `history` but is not the algorithm's answer.)

    Which label is dropped: the label sitting immediately BELOW the qualifying
    gap; if that gap is γl (below the lowest label), the lowest label is
    dropped. Either way the surviving labels are the top-N symbols, so the
    partition is fully described by N — but the dropped SYMBOL differs, which
    matters for reporting.

    Returns (final, history) where `final` is the Θ=0 result and history is the
    list of CPDResult per iteration (index 0 = the initial |L|-label
    assignment). If the loop exhausts max_iter with Θ still ≥ 1, Requirement 1
    is only partially met (the paper allows this) and the last state is
    returned.
    """
    v = np.asarray(values_desc, dtype=float)
    n = len(v)
    N = min(num_labels, n)
    if max_iter is None:
        max_iter = num_labels                    # Θ is bounded by N+1

    history = []
    for _ in range(max_iter + 1):
        if N < 1:
            break
        cuts = _cuts_from_widest_gaps(v, N)
        res = calculate_omega_prime(v, cuts, num_labels, U, L, ddof, True)
        history.append(res)
        # Requirement 1 satisfied once no gap still exceeds the widest PVI.
        if res.Theta < 1:
            break
        N -= 1
    return (history[-1] if history else None), history


def dropped_label(values_desc, cuts, grade_symbols, U, L):
    """Which grade symbol the qualifying gap says to sacrifice.

    Returns the symbol sitting immediately below the widest qualifying gap,
    or the lowest symbol when that gap is γl. Returns None if no gap qualifies.
    """
    v = np.asarray(values_desc, dtype=float)
    clusters = _clusters_from_cuts(v, cuts)
    N = len(clusters)
    pvis = [hi - lo for (hi, lo, a, b) in clusters]
    widest = max(pvis)

    gamma_u = U - clusters[0][0]
    gamma_l = clusters[-1][1] - L
    deltas = [clusters[i][1] - clusters[i + 1][0] for i in range(N - 1)]

    # candidates: (gap value, symbol below that gap)
    syms = list(grade_symbols[:N])
    cands = [(gamma_u, syms[0] if syms else None)]
    cands += [(deltas[i], syms[i + 1]) for i in range(N - 1)]
    cands += [(gamma_l, syms[-1] if syms else None)]   # γl → drop lowest

    qualifying = [(g, s) for g, s in cands if g >= widest]
    if not qualifying:
        return None
    return max(qualifying, key=lambda t: t[0])[1]


# ===========================================================================
# Helpers: pretty printing & grade assignment
# ===========================================================================

def assign_labels(values_desc, cuts, grade_symbols):
    """Attach the top-N grade symbols to the N contiguous clusters."""
    clusters = _clusters_from_cuts(values_desc, cuts)
    N = len(clusters)
    syms = grade_symbols[:N]
    out = []
    for (hi, lo, a, b), g in zip(clusters, syms):
        for val in values_desc[a:b]:
            out.append((float(val), g))
    return out, syms


def describe(res: CPDResult, grade_symbols=None):
    lines = [f"  Ω′ = {res.omega_prime:.4f}  "
             f"(Ω1={res.omega1:.4f}, Ω2={res.omega2:.4f}, Ω3={res.omega3:.4f})",
             f"  N={res.N}  θ={res.theta}  Θ={res.Theta}  σ={res.sigma:.4f}",
             f"  cuts={res.cuts}",
             f"  cluster [hi,lo] PVIs: "
             + ", ".join(f"[{hi:g},{lo:g}](PVI={hi-lo:g})"
                         for hi, lo in res.cluster_bounds)]
    if grade_symbols:
        lines.append(f"  grades: {grade_symbols[:res.N]}")
    return "\n".join(lines)
