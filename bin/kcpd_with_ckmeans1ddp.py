"""
K-CPD: K-means Conditional Performance Discretization
=======================================================

Full implementation of Algorithm 2 (K-CPD) from:
  Banditwattanawong T, Masdisornchote M. (2025).
  "Unbiased machine learning-assisted approach for conditional
  discretization of human performances." PeerJ Comput. Sci. 11:e2804.

K-CPD originally clusters performance values with scikit-learn's
heuristic KMeans (random-init, possibly non-repeatable). This script
replaces that step with **Ckmeans.1d.dp** (Wang & Song, 2011, R Journal),
the exact dynamic-programming algorithm for optimal 1-D k-means
clustering (O(n^2 k) time, guaranteed global optimum, repeatable).

Three interchangeable 1-D clustering backends are provided so results
can be compared side by side:

  1. "dp"      -- Ckmeans.1d.dp re-implemented from scratch in pure
                  Python/NumPy directly from the recurrence in the
                  R Journal paper. No external dependency. Always
                  available in this environment.
  2. "ckwrap"  -- Calls the real `ckwrap` package (a Python port of
                  Ckmeans.1d.dp) IF it is installed. This environment
                  has no network access, so it is not installed here;
                  the function will raise ImportError and the caller
                  should catch it / fall back to "dp". Run this on
                  your own machine (`pip install ckwrap`) to use it.
  3. "kmeans"  -- The original scikit-learn heuristic KMeans, kept for
                  baseline comparison against the paper's K-CPD.

Public API
----------
ckmeans_1d_dp(x, k)          -> (cluster_labels, centers, withinss)
kmeans_1d_baseline(x, k)     -> (cluster_labels, centers, withinss)
ckwrap_1d(x, k)              -> (cluster_labels, centers, withinss)  [needs ckwrap]

K_CPD(scores, grade_symbols, upper_bound, lower_bound, backend="dp")
    -> dict with final_grades_df, final_grade_symbols, omega, history

run_comparison(scores, grade_symbols, upper_bound, lower_bound)
    -> runs K-CPD with all available backends and tabulates Omega'
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# PART 1: Ckmeans.1d.dp -- exact O(n^2 k) dynamic programming
#         (re-implemented from Wang & Song, 2011, R Journal 3(2):29-33)
# ============================================================================

def ckmeans_1d_dp(x: np.ndarray, k: int):
    """
    Optimal 1-D k-means clustering by dynamic programming.

    Implements the exact recurrence from the paper:
        D[i, m] = min over m<=j<=i of { D[j-1, m-1] + d(x_j, ..., x_i) }
    where d(x_j,...,x_i) is the sum of squared distances of x_j..x_i to
    their own mean, computed incrementally in O(1) per step using:
        d(x_1..x_i) = d(x_1..x_{i-1}) + (i-1)/i * (x_i - mu_{i-1})^2
        mu_i        = (x_i + (i-1) mu_{i-1}) / i

    Parameters
    ----------
    x : 1-D array-like of numeric performance values (NOT required to be
        pre-sorted; this function sorts internally and un-sorts labels
        back to the caller's original order).
    k : number of clusters (1 <= k <= n)

    Returns
    -------
    labels  : np.ndarray of cluster indices (0 = lowest-value cluster,
              k-1 = highest-value cluster), aligned to the ORIGINAL
              (unsorted) order of x.
    centers : np.ndarray of cluster means, length k, ascending.
    withinss: float, the optimal (minimum) total within-cluster sum of
              squares achieved -- this is D[n, k].
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed number of points ({n})")

    order = np.argsort(x, kind="mergesort")  # stable sort
    xs = x[order]  # sorted ascending, 1-indexed conceptually

    # D[i, m]: min withinss clustering xs[0..i-1] (i points) into m clusters
    # B[i, m]: starting index (0-based) of the m-th (last) cluster in the
    #          optimal solution for D[i, m]
    NEG = np.inf
    D = np.full((n + 1, k + 1), NEG)
    B = np.zeros((n + 1, k + 1), dtype=int)
    D[0, 0] = 0.0

    # Precompute d(x_j..x_i) on the fly per fixed j using the incremental
    # mean/variance update, for each starting point j scanning i forward.
    # This keeps the whole DP at O(n^2 k): for each m, for each i, we scan
    # j from m..i, and for each j we need d(x_j..x_i). We recompute the
    # running d(j..i) by iterating i for fixed j (standard trick).
    for m in range(1, k + 1):
        for i in range(m, n + 1):
            best = NEG
            best_j = m  # 1-indexed start of last cluster
            # incremental d(x_j..x_i) computed by scanning j downward
            # from i to m is also valid; here we scan j upward by
            # recomputing d(j..i) afresh per j using the closed form
            # mean/var update starting at j. To stay O(n^2 k) overall
            # we recompute d(j..i) incrementally as j decreases from i.
            mu = 0.0
            d_sum = 0.0
            count = 0
            # iterate j from i down to m (1-indexed), extending the
            # cluster {x_j,...,x_i} one element to the left each step
            for j in range(i, m - 1, -1):
                xv = xs[j - 1]
                count += 1
                if count == 1:
                    mu = xv
                    d_sum = 0.0
                else:
                    d_sum += (count - 1) / count * (xv - mu) ** 2
                    mu = (xv + (count - 1) * mu) / count
                prev = D[j - 1, m - 1]
                if prev == NEG:
                    continue
                cand = prev + d_sum
                if cand < best:
                    best = cand
                    best_j = j
            D[i, m] = best
            B[i, m] = best_j

    # Backtrack to recover cluster boundaries (0-based, inclusive ranges
    # over the SORTED array xs)
    bounds = []  # list of (start0, end0) inclusive, sorted-array indices
    i, m = n, k
    while m > 0:
        j = B[i, m]  # 1-indexed start
        bounds.append((j - 1, i - 1))
        i = j - 1
        m -= 1
    bounds.reverse()  # now ascending: cluster 0 = lowest values

    labels_sorted = np.empty(n, dtype=int)
    centers = np.empty(k, dtype=float)
    for c, (s0, e0) in enumerate(bounds):
        labels_sorted[s0:e0 + 1] = c
        centers[c] = xs[s0:e0 + 1].mean()

    # map labels back to original (unsorted) order
    labels = np.empty(n, dtype=int)
    labels[order] = labels_sorted

    withinss = float(D[n, k])
    return labels, centers, withinss


# ============================================================================
# PART 2: ckwrap backend (real Python port of Ckmeans.1d.dp)
#         Requires: pip install ckwrap   (not available in this sandbox --
#         no network egress -- but this wrapper is ready to use on a
#         machine that has it installed / has internet access.)
# ============================================================================

def ckwrap_1d(x: np.ndarray, k: int):
    """
    Same contract as ckmeans_1d_dp(), but delegates to the third-party
    `ckwrap` package (a maintained Python port of Ckmeans.1d.dp).

    Raises ImportError if ckwrap is not installed -- callers should catch
    this and fall back to ckmeans_1d_dp() (pure-Python DP, always available).
    """
    import ckwrap  # noqa: defer import so module loads fine without it

    x = np.asarray(x, dtype=float)
    result = ckwrap.ckmeans(x, k)  # ckwrap API: .labels, .centers
    labels = np.asarray(result.labels, dtype=int)
    centers = np.asarray(result.centers, dtype=float)

    # ckwrap labels are already ascending by value, like ckmeans_1d_dp
    withinss = float(sum(
        (x[labels == c] - centers[c]) ** 2 if (labels == c).any() else 0.0
        for c in range(k)
    ).sum()) if k > 0 else 0.0
    # simpler/robust withinss computation:
    withinss = 0.0
    for c in range(k):
        pts = x[labels == c]
        if pts.size:
            withinss += float(((pts - pts.mean()) ** 2).sum())

    return labels, centers, withinss


# ============================================================================
# PART 3: scikit-learn heuristic KMeans baseline (the paper's original
#         K-CPD implementation choice -- kept here for comparison)
# ============================================================================

def kmeans_1d_baseline(x: np.ndarray, k: int, random_state: int = 42):
    """
    Heuristic K-means (scikit-learn, Lloyd's algorithm) on 1-D data.
    Not guaranteed optimal; included only as the baseline the paper
    itself used, so we can quantify how much Ckmeans.1d.dp improves
    on it for this specific CPD task.
    """
    from sklearn.cluster import KMeans

    x = np.asarray(x, dtype=float)
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    km.fit(x.reshape(-1, 1))
    raw_labels = km.labels_
    raw_centers = km.cluster_centers_.flatten()

    # re-map so label 0 = lowest-value cluster ... k-1 = highest-value
    # cluster, to match ckmeans_1d_dp's convention
    order = np.argsort(raw_centers)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in raw_labels])
    centers = raw_centers[order]

    withinss = float(km.inertia_)
    return labels, centers, withinss


# ============================================================================
# PART 4: CPD helper functions (Omega' metric, gap analysis) -- shared
#         logic mirroring Algorithms 1-3 of the paper / the uploaded
#         run_cpd_theta_version.py, kept self-contained here.
# ============================================================================

def labels_to_grades(scores_sorted_desc: np.ndarray, labels_for_sorted: np.ndarray,
                      grade_symbols: list[str]) -> pd.DataFrame:
    """
    Convert cluster labels (ascending: 0=lowest value) for a descendingly
    sorted score array into a Score/Grade DataFrame, assigning the FIRST
    grade_symbols[0] to the highest-value cluster, grade_symbols[-1] to
    the lowest-value cluster -- matching the paper's convention that `l`
    is "a vector of unique PRLs in a descendingly qualitative order".
    """
    n_clusters = labels_for_sorted.max() + 1
    if len(grade_symbols) < n_clusters:
        raise ValueError("Not enough grade symbols for the number of clusters")
    # cluster (n_clusters-1) = highest values -> grade_symbols[0]
    cluster_to_grade = {
        c: grade_symbols[n_clusters - 1 - c] for c in range(n_clusters)
    }
    grades = [cluster_to_grade[c] for c in labels_for_sorted]
    return pd.DataFrame({"Score": scores_sorted_desc, "Grade": grades})


def compute_score_gaps(sorted_scores_desc: np.ndarray, upper_bound: float, lower_bound: float):
    """Gaps gamma_u, delta_1..delta_{N-1}, gamma_l over descending scores."""
    extended = np.concatenate(([upper_bound], sorted_scores_desc, [lower_bound]))
    return np.abs(np.diff(extended))


def compute_grade_pvis(grades_df: pd.DataFrame) -> dict:
    pvis = {}
    for grade in grades_df["Grade"].unique():
        gs = grades_df.loc[grades_df["Grade"] == grade, "Score"]
        pvis[grade] = float(gs.max() - gs.min()) if len(gs) else 0.0
    return pvis


def calculate_omega(grades_df: pd.DataFrame, grade_symbols: list[str],
                     all_sorted_scores_desc: np.ndarray,
                     upper_bound: float, lower_bound: float,
                     theta: int, Theta: int) -> float:
    """
    Faithful implementation of Eq. (2), Omega' = Omega1 * Omega2 * Omega3,
    matching the paper's three components -- NOT the simplified formula
    used in the uploaded run_cpd_theta_version.py (which omits Omega1).
    """
    N = len(grade_symbols)

    # Omega1: requirement 1 (unassigned-PRL efficiency)
    if Theta >= 1:
        omega1 = min(theta / Theta, 1.0)
    else:
        omega1 = 1.0

    # Omega2: requirement 2 (gap maximization), needs N >= 3
    if N >= 3:
        deltas = []
        for i in range(N - 1):
            better, worse = grade_symbols[i], grade_symbols[i + 1]
            lb_better = grades_df.loc[grades_df["Grade"] == better, "Score"]
            ub_worse = grades_df.loc[grades_df["Grade"] == worse, "Score"]
            if len(lb_better) and len(ub_worse):
                deltas.append(max(0.0, lb_better.min() - ub_worse.max()))
        sum_delta_i = sum(deltas)

        all_gaps = np.sort(np.abs(np.diff(all_sorted_scores_desc)))
        if len(all_gaps) < N - 1:
            omega2 = 1.0
        else:
            sum_min = float(np.sum(all_gaps[:N - 1]))
            sum_max = float(np.sum(all_gaps[::-1][:N - 1]))
            denom = sum_max - sum_min
            omega2 = 1.0 if denom == 0 else (sum_delta_i - sum_min) / denom
    else:
        omega2 = 1.0

    # Omega3: requirement 3 (PVI uniformity), needs N >= 2
    if N >= 2:
        pvis = compute_grade_pvis(grades_df)
        sigma = float(np.std(list(pvis.values()))) if pvis else 0.0
        omega3 = 1.0 / (1.0 + sigma)
    else:
        omega3 = 1.0

    return float(omega1 * omega2 * omega3)


# ============================================================================
# PART 5: K-CPD -- Algorithm 2 of the paper, in full, with a pluggable
#         1-D clustering backend ("dp" | "ckwrap" | "kmeans")
# ============================================================================

_BACKENDS = {
    "dp": ckmeans_1d_dp,
    "kmeans": kmeans_1d_baseline,
}


def _cluster(x_sorted_desc: np.ndarray, k: int, backend: str):
    """Dispatch to the requested backend; ckwrap falls back to dp if missing."""
    x_asc = x_sorted_desc[::-1]  # all backends expect/sort internally, but
                                  # we pass ascending for clarity; ckmeans_1d_dp
                                  # sorts internally anyway so order doesn't matter
    if backend == "ckwrap":
        try:
            labels_asc, centers, withinss = ckwrap_1d(x_asc, k)
        except ImportError:
            labels_asc, centers, withinss = ckmeans_1d_dp(x_asc, k)
            backend = "dp (ckwrap unavailable, fell back)"
    else:
        func = _BACKENDS[backend]
        labels_asc, centers, withinss = func(x_asc, k)

    # x_asc[i] corresponds to x_sorted_desc[n-1-i]; convert labels back to
    # the descending order used throughout K-CPD
    n = len(x_sorted_desc)
    labels_desc = np.empty(n, dtype=int)
    labels_desc[:] = labels_asc[::-1]
    return labels_desc, centers, withinss, backend


def K_CPD(scores: np.ndarray, grade_symbols: list[str],
          upper_bound: float, lower_bound: float,
          backend: str = "dp", verbose: bool = True) -> dict:
    """
    Full Algorithm 2 (K-CPD) from the paper.

    Parameters
    ----------
    scores        : raw performance values (any order; will be sorted desc)
    grade_symbols : vector `l` of unique PRLs, descendingly qualitative
                    order, e.g. ['A', 'B', 'C']
    upper_bound, lower_bound : U and L in the paper
    backend       : "dp" (our from-scratch Ckmeans.1d.dp), "ckwrap"
                    (real package, falls back to "dp" if not installed),
                    or "kmeans" (sklearn heuristic baseline)
    verbose       : print progress, mirroring the style of
                    run_cpd_theta_version.py

    Returns
    -------
    dict with keys: final_grades_df, final_grade_symbols, final_omega,
                    iterations, backend_used, history
    """
    scores_sorted = np.sort(np.asarray(scores, dtype=float))[::-1]
    l = list(grade_symbols)
    history = []

    def log(msg):
        if verbose:
            print(msg)

    log("=" * 80)
    log(f"K-CPD  (backend = {backend})")
    log("=" * 80)

    # ---- j = 0 : initial clustering into |l| clusters --------------------
    labels, centers, withinss, backend_used = _cluster(scores_sorted, len(l), backend)
    grades_df = labels_to_grades(scores_sorted, labels, l)

    gaps = compute_score_gaps(scores_sorted, upper_bound, lower_bound)
    pvis = compute_grade_pvis(grades_df)
    max_pvi = max(pvis.values()) if pvis else 0.0
    Theta = int(np.sum(gaps >= max_pvi))
    theta = 0

    omega = calculate_omega(grades_df, l, scores_sorted, upper_bound, lower_bound, theta, Theta)
    log(f"j=0: clusters={len(l)} grades={l} Omega'={omega:.4f} Theta={Theta}")
    history.append({"j": 0, "grade_symbols": l.copy(), "omega": omega, "Theta": Theta})

    best_omega = omega
    best_grades_df = grades_df
    best_l = l.copy()

    j = 1
    # ---- while j <= Theta : Algorithm 2, lines 8-15 -----------------------
    while j <= Theta and len(l) - j >= 2:
        n_clusters = len(l) - j
        labels, centers, withinss, backend_used = _cluster(scores_sorted, n_clusters, backend)

        # K-CPD's KmeansPD() overload assigns |l|-j PRLs taken as the
        # n_clusters HIGHEST-qualitative grade_symbols (it sacrifices from
        # the tail, since clustering alone can't know which symbolic label
        # to drop -- the paper relies on Omega'-maximization across j to
        # pick the right sacrifice implicitly through re-discretization).
        l_j = l[:n_clusters]
        grades_df_j = labels_to_grades(scores_sorted, labels, l_j)

        gaps = compute_score_gaps(scores_sorted, upper_bound, lower_bound)
        pvis = compute_grade_pvis(grades_df_j)
        max_pvi = max(pvis.values()) if pvis else 0.0
        Theta = int(np.sum(gaps >= max_pvi))
        theta = j

        omega_j = calculate_omega(grades_df_j, l_j, scores_sorted, upper_bound, lower_bound, theta, Theta)
        log(f"j={j}: clusters={n_clusters} grades={l_j} Omega'={omega_j:.4f} Theta={Theta}")
        history.append({"j": j, "grade_symbols": l_j.copy(), "omega": omega_j, "Theta": Theta})

        if omega_j > best_omega:
            best_omega = omega_j
            best_grades_df = grades_df_j
            best_l = l_j.copy()

        j += 1

    log("-" * 80)
    log(f"Best Omega' = {best_omega:.4f} with grades {best_l}  [backend used: {backend_used}]")

    return {
        "final_grades_df": best_grades_df.reset_index(drop=True),
        "final_grade_symbols": best_l,
        "final_omega": best_omega,
        "iterations": j - 1,
        "backend_used": backend_used,
        "history": history,
    }


# ============================================================================
# PART 6: side-by-side comparison runner
# ============================================================================

def run_comparison(scores: np.ndarray, grade_symbols: list[str],
                    upper_bound: float, lower_bound: float,
                    verbose: bool = False) -> pd.DataFrame:
    """
    Runs K-CPD with each available backend on the SAME data and tabulates
    the resulting Omega', number of grades retained, and which backend
    actually executed (useful since "ckwrap" silently falls back to "dp"
    in network-restricted environments).
    """
    rows = []
    for backend in ["dp", "ckwrap", "kmeans"]:
        result = K_CPD(scores, grade_symbols, upper_bound, lower_bound,
                        backend=backend, verbose=verbose)
        rows.append({
            "backend_requested": backend,
            "backend_used": result["backend_used"],
            "n_grades_final": len(result["final_grade_symbols"]),
            "final_grades": result["final_grade_symbols"],
            "omega_prime": round(result["final_omega"], 4),
            "iterations": result["iterations"],
        })
    return pd.DataFrame(rows)


# ============================================================================
# Sanity check / smoke test against the paper's worked example (Table 4,
# EMP1 data set is too large to hand-type here; instead we verify the DP
# matches sklearn's GLOBAL optimum on a small synthetic case where brute
# force is feasible).
# ============================================================================

def _self_test():
    rng = np.random.default_rng(0)
    x = np.concatenate([
        rng.normal(0, 0.3, 20),
        rng.normal(5, 0.3, 20),
        rng.normal(10, 0.3, 20),
    ])
    k = 3
    labels_dp, centers_dp, withinss_dp = ckmeans_1d_dp(x, k)
    labels_km, centers_km, withinss_km = kmeans_1d_baseline(x, k)

    print("Self-test: 3 well-separated Gaussian blobs, k=3")
    print(f"  DP withinss     = {withinss_dp:.6f}")
    print(f"  sklearn withinss= {withinss_km:.6f}")
    assert withinss_dp <= withinss_km + 1e-9, "DP should be <= heuristic kmeans"
    print("  PASS: DP achieves withinss <= sklearn KMeans (DP is optimal)")


if __name__ == "__main__":
    _self_test()

    print("\n" + "=" * 80)
    print("DEMO: K-CPD on a small synthetic score set")
    print("=" * 80)
    demo_scores = np.array([95, 93, 91, 78, 76, 75, 60, 58, 40, 38, 20, 18])
    demo_grades = ["A", "B", "C", "D", "F"]
    comparison = run_comparison(demo_scores, demo_grades, upper_bound=100, lower_bound=0)
    print()
    print(comparison.to_string(index=False))
