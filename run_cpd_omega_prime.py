"""
Verification + benchmark driver for optimal_cpd_omega_prime.py

Run:  python3 run_cpd_omega_prime.py [path_to_scores.csv]

- Verifies calculate_omega_prime() against Table 3 of the PeerJ paper.
- Benchmarks EXHAUSTIVE (optimal) vs DP-based (heuristic) search of Ω′.
- If a CSV/xlsx path is given, its scores are used; otherwise the paper's
  Table 3 dataset (30 values) is used as a validated demo.
"""

import sys
import time
import numpy as np

from optimal_cpd_omega_prime import (
    calculate_omega_prime, exhaustive_optimal, dp_best,
    exhaustive_best_for_k, dp_best as _dp, describe, assign_labels,
)

# ---------------------------------------------------------------------------
# Paper Table 3 dataset (30 descending values, U=100, L=0, |L|=5: A,B,C,D,F)
# ---------------------------------------------------------------------------
TABLE3 = [82, 80, 76, 75, 72, 70, 69, 69, 68, 68, 67, 65, 65, 62, 61, 59, 58,
          57, 57, 57, 56, 56, 55, 54, 53, 52, 52, 51, 50, 50]

# A1 assigns A,B,C,D  (N=4): [82,80][76,75][72..65][62..50]  → cuts at 2,4,13
A1_CUTS = (2, 4, 13)
# A2 assigns B,C,D    (N=3): [82,80,76,75][72..65][62..50]   → cuts at 4,13
A2_CUTS = (4, 13)


def test_paper_examples():
    print("=" * 74)
    print("VERIFICATION 1 — reproduce Table 3 of the PeerJ paper")
    print("=" * 74)
    r1 = calculate_omega_prime(TABLE3, A1_CUTS, num_labels=5, U=100, L=0,
                               return_details=True)
    r2 = calculate_omega_prime(TABLE3, A2_CUTS, num_labels=5, U=100, L=0,
                               return_details=True)
    print("\nA1 (N=4, assigns A,B,C,D):")
    print(describe(r1))
    print(f"  expected Ω′ ≈ 0.08   got {r1.omega_prime:.4f}")
    print("\nA2 (N=3, assigns B,C,D):")
    print(describe(r2))
    print(f"  expected Ω′ ≈ 0.22   got {r2.omega_prime:.4f}")

    ok1 = abs(r1.omega_prime - 0.08) < 0.005
    ok2 = abs(r2.omega_prime - 0.22) < 0.005
    # component-level checks
    comp_ok = (abs(r1.omega1 - 0.5) < 1e-9 and abs(r1.omega2 - 1.0) < 1e-9 and
               abs(r1.sigma - 5.07) < 0.02 and
               abs(r2.omega1 - 1.0) < 1e-9 and abs(r2.omega2 - 6/7) < 1e-6 and
               abs(r2.sigma - 2.89) < 0.02)
    assert ok1, f"A1 Ω′ mismatch: {r1.omega_prime}"
    assert ok2, f"A2 Ω′ mismatch: {r2.omega_prime}"
    assert comp_ok, "Ω1/Ω2/σ component mismatch"
    print("\n[PASS] Table 3 reproduced: A1=0.08, A2=0.22, and Ω1/Ω2/σ match.\n")
    return True


def load_scores(path):
    if path.lower().endswith((".xlsx", ".xls")):
        import pandas as pd
        df = pd.read_excel(path)
    else:
        import pandas as pd
        df = pd.read_csv(path)
    # pick the first numeric column (or one named Score/score)
    col = None
    for c in df.columns:
        if str(c).lower() in ("score", "scores", "performance", "value"):
            col = c
            break
    if col is None:
        num = df.select_dtypes(include=[np.number])
        col = num.columns[0]
    vals = df[col].dropna().to_numpy(dtype=float)
    return vals


def benchmark(values, num_labels, U, L, grade_symbols, k_min=2, k_max=None):
    print("=" * 74)
    print("BENCHMARK — Exhaustive (optimal) vs DP-based (heuristic) on Ω′")
    print("=" * 74)
    v = np.sort(np.asarray(values, dtype=float))[::-1]     # descending
    n = len(v)
    k_max = num_labels if k_max is None else k_max
    k_max = min(k_max, n)
    print(f"n={n}, value range [{v.min():g}, {v.max():g}], "
          f"|L|={num_labels} grades={grade_symbols[:num_labels]}, "
          f"U={U}, L={L}, k∈[{k_min},{k_max}]\n")

    # ---- exhaustive (search over k) ----
    t0 = time.perf_counter()
    ex_best, ex_per_k = exhaustive_optimal(v, num_labels, U, L,
                                           k_min=k_min, k_max=k_max)
    t_ex = time.perf_counter() - t0

    # ---- DP-based (search over k) ----
    t0 = time.perf_counter()
    dp_best_res, dp_per_k = dp_best(v, num_labels, U, L,
                                    k_min=k_min, k_max=k_max)
    t_dp = time.perf_counter() - t0

    # ---- per-k comparison table ----
    print(f"{'k':>3} | {'Ω′ exhaustive':>14} | {'Ω′ DP':>10} | "
          f"{'exh cuts':>18} | {'DP cuts':>18} | match")
    print("-" * 84)
    all_ge = True
    for kk in range(k_min, k_max + 1):
        e = ex_per_k.get(kk)
        d = dp_per_k.get(kk)
        if e is None or d is None:
            continue
        match = "yes" if e.cuts == d.cuts else "no"
        if e.omega_prime + 1e-9 < d.omega_prime:
            all_ge = False
        print(f"{kk:>3} | {e.omega_prime:>14.4f} | {d.omega_prime:>10.4f} | "
              f"{str(e.cuts):>18} | {str(d.cuts):>18} | {match}")

    print("\n--- OVERALL BEST (unconstrained: k chosen freely by Ω′) ---")
    print(f"\nEXHAUSTIVE optimal  (runtime {t_ex*1000:.1f} ms):")
    print(describe(ex_best, grade_symbols))
    print(f"\nDP-based heuristic  (runtime {t_dp*1000:.1f} ms):")
    print(describe(dp_best_res, grade_symbols))

    gap = ex_best.omega_prime - dp_best_res.omega_prime
    print(f"\nΩ′ gap (exhaustive − DP) = {gap:+.4f}  "
          f"({'DP is optimal here' if abs(gap) < 1e-9 else 'DP is sub-optimal'})")

    # ---- degeneracy warning + meaningful (non-degenerate) best ----
    if ex_best.N < 3:
        print("\n[!] NOTE — Ω′ DEGENERACY: the unconstrained optimum collapses to "
              f"N={ex_best.N}.")
        print("    When N<3, Ω2 is forced to 1 and an equal-PVI 2-split gives σ=0 "
              "→ Ω′=1 trivially.")
        print("    Ω′ is a metric to COMPARE methods at a fixed label budget |L|, "
              "not an\n    objective to minimise freely over k. The paper's "
              "algorithms start at |L|\n    grades and only REDUCE per Requirement 1 "
              "— they never search down to N=2.")
        best_nd = None
        for kk in range(max(3, k_min), k_max + 1):
            e = ex_per_k.get(kk)
            if e and (best_nd is None or e.omega_prime > best_nd.omega_prime):
                best_nd = e
        if best_nd is not None:
            print("\n    Best NON-degenerate exhaustive optimum (N≥3):")
            print(describe(best_nd, grade_symbols))

    # ---- faithful comparison at the FIXED label budget k=|L| ----
    print("\n--- CONSTRAINED at fixed label budget k=|L| "
          f"(={min(num_labels,n)} grades, paper's intent) ---")
    kL = min(num_labels, n)
    exL = exhaustive_best_for_k(v, kL, num_labels, U, L)
    dpL = dp_best(v, num_labels, U, L, k=kL)
    print("\nEXHAUSTIVE optimal @k=|L|:")
    print(describe(exL, grade_symbols))
    print("\nDP-based @k=|L|:")
    print(describe(dpL, grade_symbols))
    assert exL.omega_prime + 1e-9 >= dpL.omega_prime, "DP beat exhaustive @k=|L|"

    print("\n" + "=" * 74)
    print("VERIFICATION 2 — exhaustive Ω′ ≥ DP Ω′ for every k, and overall")
    print("=" * 74)
    assert all_ge, "DP beat the exhaustive optimum at some k — bug!"
    assert ex_best.omega_prime + 1e-9 >= dp_best_res.omega_prime, \
        "DP beat the exhaustive overall optimum — bug!"
    print("[PASS] Exhaustive dominates DP everywhere (as it must).\n")

    # final grade assignment of the optimal partition
    labeled, syms = assign_labels(v, ex_best.cuts, grade_symbols)
    print("Optimal (exhaustive) grade assignment:")
    cur = None
    for val, g in labeled:
        if g != cur:
            print(f"  grade {g}: ", end="")
            cur = g
        print(f"{val:g} ", end="")
    print()
    return ex_best, dp_best_res


if __name__ == "__main__":
    # 1) verify the metric
    test_paper_examples()

    # 2) pick dataset
    if len(sys.argv) > 1:
        vals = load_scores(sys.argv[1])
        # sensible defaults; override for your grading scheme
        U, L = 100.0, 0.0
        grades = ["A", "B+", "B", "C+", "C", "D+", "D", "F"]
        num_labels = len(grades)
        print(f"Loaded {len(vals)} scores from {sys.argv[1]}\n")
    else:
        vals = TABLE3
        U, L = 100.0, 0.0
        grades = ["A", "B", "C", "D", "F"]
        num_labels = 5
        print("No data file given → using the paper's Table 3 demo dataset.\n")

    # 3) benchmark exhaustive vs DP
    benchmark(vals, num_labels, U, L, grades)

    print("Done. Attach your real WGP.csv (or pass a path) to grade actual data.")
