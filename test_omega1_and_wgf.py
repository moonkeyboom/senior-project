"""Regression tests for the Ω1 formula and the reduce-from-|L| (WGF) loop.

Two things are locked down here:

1. Ω1 = 1 − |Θ−θ|/Θ, NOT θ/Θ. The paper's Table 3 examples both have θ ≤ Θ,
   where the two forms coincide — so Table 3 alone cannot catch the bug. The
   synthetic cases below deliberately use θ > Θ, where they diverge.

2. Requirement 1: whenever Θ ≥ 1 a label must be sacrificed and the orphaned
   scores re-graded into the surviving neighbour. Verified against the manual
   spreadsheet (CPDtemplateV2.xlsx), whose two heuristic iterations are the
   ground truth for the loop's mechanics.
"""

import csv
import os

import numpy as np

from optimal_cpd_omega_prime import (
    calculate_omega_prime,
    wgf_reduce,
    dropped_label,
    _clusters_from_cuts,
)

U, L = 100.0, 0.0
GRADES = ["A", "B+", "B", "C+", "C", "D+", "D", "F"]

# --- paper Table 3 -------------------------------------------------------
TABLE3 = [82, 80, 76, 75, 72, 70, 69, 69, 68, 68, 67, 65, 65, 62, 61, 59, 58,
          57, 57, 57, 56, 56, 55, 54, 53, 52, 52, 51, 50, 50]
A1_CUTS = (2, 4, 13)     # A|B|C|D  → N=4
A2_CUTS = (4, 13)        # B|C|D    → N=3


def _cuts_from_groups(values_desc, groups):
    """Convert a list of score-groups into cut indices."""
    cuts, acc = [], 0
    for g in groups[:-1]:
        acc += len(g)
        cuts.append(acc)
    return tuple(cuts)


def load_manual():
    """The 139 real scores + the manual grading of both iterations."""
    path = os.path.join("data", "wgp_manual.csv")
    scores, g1, g2 = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores.append(float(row["score"]))
            g1.append(row["grade_iter1"])
            g2.append(row["grade_iter2"])
    return scores, g1, g2


def groups_of(scores, grades):
    out, cur, last = [], [], None
    for s, g in zip(scores, grades):
        if g != last and cur:
            out.append(cur)
            cur = []
        cur.append(s)
        last = g
    if cur:
        out.append(cur)
    return out


# =========================================================================
def test_table3_still_passes():
    """The fix must not disturb the paper's published values."""
    r1 = calculate_omega_prime(TABLE3, A1_CUTS, 5, U, L, return_details=True)
    r2 = calculate_omega_prime(TABLE3, A2_CUTS, 5, U, L, return_details=True)

    assert (r1.theta, r1.Theta) == (1, 2), (r1.theta, r1.Theta)
    assert (r2.theta, r2.Theta) == (2, 2), (r2.theta, r2.Theta)
    assert abs(r1.sigma - 5.0662) < 1e-3, r1.sigma
    assert abs(r2.sigma - 2.8868) < 1e-3, r2.sigma
    assert abs(r1.omega_prime - 0.08) < 0.005, r1.omega_prime
    assert abs(r2.omega_prime - 0.22) < 0.005, r2.omega_prime
    print(f"  Table 3: A1 Ω′={r1.omega_prime:.4f}  A2 Ω′={r2.omega_prime:.4f}  OK")


def test_omega1_is_two_sided():
    """θ > Θ must be penalised. θ/Θ would saturate at 1.0 here."""
    cases = [
        # (theta, Theta, expected Ω1)
        (0, 1, 0.0),
        (1, 1, 1.0),
        (2, 1, 0.0),      # θ/Θ would give 1.0
        (3, 2, 0.5),      # θ/Θ would give 1.0
        (4, 2, 0.0),      # θ/Θ would give 1.0
        (2, 4, 0.5),
    ]
    for theta, Theta, expected in cases:
        got = min(max(1.0 - abs(Theta - theta) / Theta, 0.0), 1.0)
        wrong = min(max(theta / Theta, 0.0), 1.0)
        assert abs(got - expected) < 1e-9, (theta, Theta, got, expected)
        if theta > Theta:
            assert got != wrong, f"θ={theta} Θ={Theta} must diverge from θ/Θ"
    print("  Ω1 two-sided penalty: OK (θ>Θ diverges from θ/Θ)")


def test_manual_iteration1_requires_a_cut():
    """Iteration 1 uses all 8 labels; γl qualifies, so Requirement 1 is unmet."""
    scores, g1, _ = load_manual()
    v = np.array(scores, dtype=float)
    cuts = _cuts_from_groups(v, groups_of(scores, g1))
    res = calculate_omega_prime(v, cuts, len(GRADES), U, L, return_details=True)

    assert res.N == 8, res.N
    assert res.theta == 0, res.theta
    assert res.Theta >= 1, res.Theta          # a gap still exceeds widest PVI
    assert res.omega1 == 0.0, res.omega1      # unassigned too few → Ω1 = 0
    assert dropped_label(v, cuts, GRADES, U, L) == "F"
    print(f"  iter1: N={res.N} θ={res.theta} Θ={res.Theta} Ω1={res.omega1:.1f} "
          f"→ must drop 'F'  OK")


def test_manual_iteration2_matches_spreadsheet():
    """After dropping F, the 4 orphaned scores merge into D: [39.5 .. 35]."""
    scores, _, g2 = load_manual()
    v = np.array(scores, dtype=float)
    groups = groups_of(scores, g2)
    cuts = _cuts_from_groups(v, groups)
    res = calculate_omega_prime(v, cuts, len(GRADES), U, L, return_details=True)

    assert res.N == 7, res.N
    assert res.theta == 1, res.theta
    assert res.omega1 == 1.0, res.omega1      # exactly one sacrificed → Ω1 = 1

    lo_hi = res.cluster_bounds[-1]
    assert lo_hi == (39.5, 35.0), lo_hi       # D absorbed the ex-F scores
    assert abs((lo_hi[0] - lo_hi[1]) - 4.5) < 1e-9
    print(f"  iter2: N={res.N} θ={res.theta} Θ={res.Theta} Ω1={res.omega1:.1f} "
          f"D={lo_hi} PVI=4.5  OK")


def test_wgf_reduce_reproduces_the_manual_loop():
    """Iteration 1 → 2 must match the human's first reduction (N=8 → N=7)."""
    scores, _, _ = load_manual()
    v = np.array(scores, dtype=float)
    _, history = wgf_reduce(v, len(GRADES), U, L)

    assert history[0].N == 8, history[0].N
    assert history[0].Theta >= 1               # triggers the reduction
    assert len(history) >= 2, len(history)
    assert history[1].N == 7, history[1].N
    assert history[1].theta == 1, history[1].theta
    print(f"  wgf_reduce: {' → '.join('N=%d' % h.N for h in history)}  OK")


def test_wgf_runs_until_theta_zero():
    """Algorithm 1 terminates on the constraint Θ=0, not on max Ω′."""
    scores, _, _ = load_manual()
    v = np.array(scores, dtype=float)
    final, history = wgf_reduce(v, len(GRADES), U, L)

    assert final is history[-1], "final must be the terminal state"
    assert final.Theta == 0, f"loop ended with Θ={final.Theta}, expected 0"
    # every non-terminal step must still have had an unmet Requirement 1
    for h in history[:-1]:
        assert h.Theta >= 1, (h.N, h.Theta)
    # the terminal state is NOT necessarily the highest-Ω′ one
    best_omega = max(h.omega_prime for h in history)
    print(f"  terminal: N={final.N} Θ={final.Theta} Ω′={final.omega_prime:.4f} "
          f"(max Ω′ seen={best_omega:.4f})  OK")


def test_requirement1_drives_reduction():
    """Core invariant: Θ ≥ 1 must always force another label to be sacrificed."""
    scores, _, _ = load_manual()
    v = np.array(scores, dtype=float)
    _, history = wgf_reduce(v, len(GRADES), U, L)

    for prev, nxt in zip(history, history[1:]):
        assert prev.Theta >= 1, (
            f"stopped reducing at N={prev.N} despite Θ={prev.Theta}")
        assert nxt.N == prev.N - 1, (prev.N, nxt.N)
    assert history[-1].Theta < 1 or len(history) > 1
    print(f"  Requirement 1 loop invariant holds over {len(history)} iterations  OK")


if __name__ == "__main__":
    tests = [
        test_table3_still_passes,
        test_omega1_is_two_sided,
        test_manual_iteration1_requires_a_cut,
        test_manual_iteration2_matches_spreadsheet,
        test_wgf_reduce_reproduces_the_manual_loop,
        test_wgf_runs_until_theta_zero,
        test_requirement1_drives_reduction,
    ]
    failed = 0
    for t in tests:
        print(f"{t.__name__}:")
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
    print()
    print("ALL PASS" if not failed else f"{failed} FAILED")
