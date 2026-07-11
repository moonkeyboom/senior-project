import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from itertools import combinations
import os

# ============================================================================
# CPD REFINEMENT ALGORITHM - K-MEANS METHOD (Algorithm 2 from Paper)
# ============================================================================

def calculate_omega(scores_grades_df, grade_symbols, all_sorted_scores):
    '''Calculate Omega fairness metric'''
    N = len(grade_symbols)
    grade_intervals = []
    for grade in grade_symbols:
        grade_scores = scores_grades_df[scores_grades_df['Grade'] == grade]['Score']
        if grade_scores.size > 0:
            interval = grade_scores.max() - grade_scores.min()
            grade_intervals.append(interval)

    sigma = np.std(grade_intervals) if grade_intervals else 0.0

    sum_delta_i = 0.0
    for i in range(N - 1):
        better_grade = grade_symbols[i]
        worse_grade = grade_symbols[i+1]
        lower_bound_better_grade_scores = scores_grades_df[scores_grades_df['Grade'] == better_grade]['Score']
        upper_bound_worse_grade_scores = scores_grades_df[scores_grades_df['Grade'] == worse_grade]['Score']
        if lower_bound_better_grade_scores.size > 0 and upper_bound_worse_grade_scores.size > 0:
            lower_bound_better = lower_bound_better_grade_scores.min()
            upper_bound_worse = upper_bound_worse_grade_scores.max()
            delta = lower_bound_better - upper_bound_worse
            sum_delta_i += max(0, delta)

    score_gaps = np.diff(all_sorted_scores)
    score_gaps = np.abs(score_gaps)

    if len(score_gaps) < N - 1:
        return 0.0

    sorted_gaps = np.sort(score_gaps)
    sum_delta_min = np.sum(sorted_gaps[:N-1])
    sum_delta_max = np.sum(sorted_gaps[::-1][:N-1])

    denominator = (1 + sigma) * (sum_delta_max - sum_delta_min)
    if denominator == 0:
        return 0.0

    omega = (sum_delta_i - sum_delta_min) / denominator
    return omega


def compute_score_gaps(sorted_scores, upper_bound=100, lower_bound=0):
    """Computes score gaps including boundary gaps."""
    if len(sorted_scores) < 1:
        return np.array([])

    extended_scores = [upper_bound] + sorted_scores.tolist() + [lower_bound]
    extended_scores = np.array(extended_scores)
    gaps = np.abs(np.diff(extended_scores))
    return gaps


def compute_grade_pvis(grades_df):
    '''Computes Performance Value Intervals (PVI) for each grade'''
    pvis = {}
    for grade in grades_df['Grade'].unique():
        grade_scores = grades_df[grades_df['Grade'] == grade]['Score']
        if len(grade_scores) > 0:
            pvi_width = grade_scores.max() - grade_scores.min()
            pvis[grade] = pvi_width
        else:
            pvis[grade] = 0.0
    return pvis


def get_max_gap_indices(gaps):
    '''Finds all indices of the maximum gap value'''
    if len(gaps) == 0:
        return []
    max_gap = np.max(gaps)
    max_indices = np.where(gaps == max_gap)[0].tolist()
    return max_indices


def get_max_pvi(pvis):
    '''Finds the maximum PVI value among all grades'''
    if not pvis:
        return 0.0
    return max(pvis.values())


def determine_label_to_remove(gap_index, extended_scores, grade_symbols, grades_df):
    """Determines which grade label to remove based on the gap position."""
    if gap_index == 0:
        top_score = extended_scores[1]
        top_student_idx = grades_df[grades_df['Score'] == top_score].index[0]
        grade_to_remove = grades_df.loc[top_student_idx, 'Grade']
        return grade_to_remove
    elif gap_index == len(extended_scores) - 2:
        bottom_score = extended_scores[-2]
        bottom_student_idx = grades_df[grades_df['Score'] == bottom_score].index[-1]
        grade_to_remove = grades_df.loc[bottom_student_idx, 'Grade']
        return grade_to_remove
    else:
        score_above = extended_scores[gap_index]
        score_below = extended_scores[gap_index + 1]
        idx_above = grades_df[grades_df['Score'] == score_above].index[0]
        idx_below = grades_df[grades_df['Score'] == score_below].index[-1]
        grade_above = grades_df.loc[idx_above, 'Grade']
        grade_below = grades_df.loc[idx_below, 'Grade']
        return grade_below


def run_kmeans_segmentation(scores, grade_symbols, random_state=42):
    """
    Algorithm 2: K-means-based CPD segmentation method.

    Steps:
    1. Apply K-means clustering with k = len(grade_symbols)
    2. Sort clusters by centroid values (descending)
    3. Map highest centroid -> best grade (A)
    4. Assign grades based on cluster membership

    Args:
        scores (np.array): Sorted scores in descending order.
        grade_symbols (list): Ordered list of grade symbols (best to worst).
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with 'Score' and 'Grade' columns.
    """
    n_clusters = len(grade_symbols)

    if n_clusters < 2:
        raise ValueError("Need at least 2 grade symbols")

    # Reshape scores for K-means (expects 2D array)
    scores_reshaped = scores.reshape(-1, 1)

    # Initialize K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')

    # Fit K-means
    kmeans.fit(scores_reshaped)

    # Get cluster labels and centroids
    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_.flatten()

    # Sort clusters by centroid values (descending)
    sorted_cluster_indices = np.argsort(centroids)[::-1]

    # Map cluster indices to grade symbols
    cluster_to_grade = {}
    for rank, cluster_idx in enumerate(sorted_cluster_indices):
        cluster_to_grade[cluster_idx] = grade_symbols[rank]

    # Assign grades based on cluster membership
    grades = [cluster_to_grade[label] for label in cluster_labels]

    # Create result DataFrame
    result_df = pd.DataFrame({
        'Score': scores,
        'Grade': grades
    })

    # Reset index for safe lookup
    result_df = result_df.reset_index(drop=True)

    return result_df


def cpd_kmeans_refinement_loop(scores, initial_grades_df, grade_symbols,
                                upper_bound=100, lower_bound=0,
                                random_state=42, max_iterations=10):
    """
    CPD refinement loop for K-means method.

    Iteratively removes grade labels until fairness constraint is satisfied.
    """
    current_grades_df = initial_grades_df.copy()
    current_grades_df = current_grades_df.reset_index(drop=True)
    current_grade_symbols = grade_symbols.copy()
    iteration = 0
    history = []

    print('\n' + '='*80)
    print('CPD K-MEANS REFINEMENT LOOP - Algorithm 2')
    print('='*80)

    while iteration < max_iterations:
        iteration += 1

        print(f'\n{"="*60}')
        print(f'ITERATION {iteration}')
        print(f'{"="*60}')

        # Display current state
        print(f'\nCurrent state:')
        print(f'  Grade symbols: {current_grade_symbols}')
        print(f'  Number of grades: {len(current_grade_symbols)}')

        # Step 1: Compute gaps including boundaries
        gaps = compute_score_gaps(scores, upper_bound, lower_bound)
        pvis = compute_grade_pvis(current_grades_df)

        # Step 2: Find max_gap and max_PVI
        max_gap_value = np.max(gaps) if len(gaps) > 0 else 0
        max_pvi_value = get_max_pvi(pvis)

        print(f'\nGap Analysis:')
        print(f'  Max gap: {max_gap_value:.4f}')
        print(f'  Max PVI: {max_pvi_value:.4f}')

        # Display PVI per grade
        print(f'\nPVI per Grade:')
        for grade in sorted(current_grade_symbols, key=lambda x: current_grade_symbols.index(x)):
            if grade in pvis:
                print(f'  {grade}: {pvis[grade]:.4f}')

        # Step 3: Check fairness constraint
        print(f'\nFairness Check:')
        if max_gap_value <= max_pvi_value:
            print(f'  SATISFIED: {max_gap_value:.4f} <= {max_pvi_value:.4f}')
            print('\n' + '='*80)
            print('CPD REFINEMENT COMPLETE - FAIRNESS ACHIEVED')
            print('='*80)
            break

        print(f'  VIOLATION: {max_gap_value:.4f} > {max_pvi_value:.4f}')
        print(f'  Action: Must remove a grade label')

        # Step 4: Find all max_gap indices
        max_gap_indices = get_max_gap_indices(gaps)
        print(f'\nMax Gap Analysis:')
        print(f'  Found {len(max_gap_indices)} gap(s) with max value {max_gap_value:.4f}')
        print(f'  Indices: {max_gap_indices}')

        # Identify gap types
        for idx in max_gap_indices:
            if idx == 0:
                print(f'    Gap {idx}: UPPER BOUNDARY gap ({upper_bound} -> {scores.max():.2f})')
            elif idx == len(gaps) - 1:
                print(f'    Gap {idx}: LOWER BOUNDARY gap ({scores.min():.2f} -> {lower_bound})')
            else:
                extended_scores = [upper_bound] + scores.tolist() + [lower_bound]
                score_above = extended_scores[idx]
                score_below = extended_scores[idx + 1]
                print(f'    Gap {idx}: Internal gap ({score_above:.2f} -> {score_below:.2f})')

        # Step 5: Try each candidate
        print(f'\nEvaluating Candidates:')
        best_candidate = None
        best_omega = -1.0

        extended_scores = [upper_bound] + scores.tolist() + [lower_bound]

        for idx, gap_idx in enumerate(max_gap_indices, 1):
            print(f'\n  Candidate {idx}:')
            print(f'    Gap index: {gap_idx}')
            print(f'    Gap value: {gaps[gap_idx]:.4f}')

            # Determine gap type
            if gap_idx == 0:
                print(f'    Type: UPPER BOUNDARY gap')
            elif gap_idx == len(gaps) - 1:
                print(f'    Type: LOWER BOUNDARY gap')
            else:
                score_above = extended_scores[gap_idx]
                score_below = extended_scores[gap_idx + 1]
                print(f'    Type: Internal gap ({score_above:.2f} -> {score_below:.2f})')

            # Determine label to remove
            label_to_remove = determine_label_to_remove(
                gap_idx, np.array(extended_scores), current_grade_symbols, current_grades_df
            )
            print(f"    Label to remove: '{label_to_remove}'")

            # Create reduced grade symbols
            reduced_grade_symbols = [g for g in current_grade_symbols if g != label_to_remove]
            print(f'    New grade symbols: {reduced_grade_symbols}')

            if len(reduced_grade_symbols) < 2:
                print(f'    Status: SKIPPED (would result in < 2 grades)')
                continue

            # Re-run K-means segmentation
            try:
                candidate_grades_df = run_kmeans_segmentation(
                    scores, reduced_grade_symbols, random_state
                )
                candidate_omega = calculate_omega(
                    candidate_grades_df, reduced_grade_symbols, scores
                )

                print(f'    Omega: {candidate_omega:.4f}')

                # Check for empty grades
                empty_grades = []
                for grade in reduced_grade_symbols:
                    count = len(candidate_grades_df[candidate_grades_df['Grade'] == grade])
                    if count == 0:
                        empty_grades.append(grade)

                if empty_grades:
                    print(f'    Warning: Empty grades: {empty_grades}')

                # Select best candidate
                if candidate_omega > best_omega:
                    best_omega = candidate_omega
                    best_candidate = {
                        'grades_df': candidate_grades_df,
                        'grade_symbols': reduced_grade_symbols,
                        'omega': candidate_omega,
                        'removed_label': label_to_remove,
                        'gap_index': gap_idx
                    }
                    print(f'    Status: NEW BEST CANDIDATE')
                else:
                    print(f'    Status: Not better than current best ({best_omega:.4f})')

            except Exception as e:
                print(f'    Status: ERROR - {e}')
                continue

        # Step 6: Select best candidate
        if best_candidate is None:
            print(f'\nNo valid candidate found. Stopping refinement.')
            break

        print(f'\n--- Best Candidate Selected ---')
        print(f"  Removed label: '{best_candidate['removed_label']}'")
        print(f"  Gap index: {best_candidate['gap_index']}")
        print(f"  Omega: {best_candidate['omega']:.4f}")

        # Update for next iteration
        current_grades_df = best_candidate['grades_df'].reset_index(drop=True)
        current_grade_symbols = best_candidate['grade_symbols']

        # Record history
        history.append({
            'iteration': iteration,
            'removed_label': best_candidate['removed_label'],
            'gap_index': best_candidate['gap_index'],
            'omega': best_candidate['omega'],
            'grade_symbols': current_grade_symbols.copy()
        })

    # Calculate final Omega
    final_omega = calculate_omega(current_grades_df, current_grade_symbols, scores)

    return {
        'final_grades_df': current_grades_df,
        'final_grade_symbols': current_grade_symbols,
        'final_omega': final_omega,
        'iterations': iteration,
        'history': history
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    '''Main execution function for K-means CPD method'''

    print('='*80)
    print('CPD K-MEANS GRADING - Algorithm 2 Implementation')
    print('='*80)

    # Step 1: Load dataset
    input_file = './file/input/221.xlsx'
    print(f'\nLoading dataset: {input_file}')

    df = pd.read_excel(input_file)
    scores = df['Score'].values
    sorted_scores = np.sort(scores)[::-1]

    print(f'Total students: {len(scores)}')
    print(f'Score range: {scores.min():.2f} - {scores.max():.2f}')
    print(f'Mean score: {scores.mean():.2f}')
    print(f'Std deviation: {scores.std():.2f}')

    # Step 2: Initialize grade symbols
    grade_symbols = ['A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F']
    print(f'\nInitial grade symbols: {grade_symbols}')
    print(f'Number of grades: {len(grade_symbols)}')

    # Step 3: Validate gap calculation
    print(f'\n{"="*80}')
    print('GAP VALIDATION')
    print(f'{"="*80}')

    gaps = compute_score_gaps(sorted_scores, upper_bound=100, lower_bound=0)
    print(f'Total gaps (including boundaries): {len(gaps)}')
    print(f'Gap at upper bound (100 -> {scores.max():.2f}): {gaps[0]:.4f}')
    print(f'Gap at lower bound ({scores.min():.2f} -> 0): {gaps[-1]:.4f}')
    print(f'Max gap: {np.max(gaps):.4f}')

    if np.max(gaps) == 35.0:
        print('[OK] Max gap is 35.0 as expected')
    else:
        print(f'[WARNING] Max gap is {np.max(gaps):.4f} (expected 35.0)')

    # Step 4: Run initial K-means segmentation
    print(f'\n{"="*80}')
    print('INITIAL K-MEANS SEGMENTATION')
    print(f'{"="*80}')

    initial_grades_df = run_kmeans_segmentation(sorted_scores, grade_symbols, random_state=42)
    initial_omega = calculate_omega(initial_grades_df, grade_symbols, sorted_scores)

    print(f'Initial K-means omega: {initial_omega:.4f}')

    # Display initial grade distribution
    print(f'\nInitial Grade Distribution:')
    initial_dist = initial_grades_df.groupby('Grade')['Score'].agg(['count', 'min', 'max'])
    print(initial_dist)

    # Step 5: Run CPD refinement loop
    print(f'\n{"="*80}')
    print('STARTING CPD REFINEMENT LOOP')
    print(f'{"="*80}')

    cpd_result = cpd_kmeans_refinement_loop(
        sorted_scores,
        initial_grades_df,
        grade_symbols,
        upper_bound=100,
        lower_bound=0,
        random_state=42,
        max_iterations=10
    )

    # Step 6: Display final results
    print(f'\n{"="*80}')
    print('FINAL RESULTS')
    print(f'{"="*80}')

    print(f'\nInitial Configuration:')
    print(f'  Grades: {grade_symbols}')
    print(f'  Number of grades: {len(grade_symbols)}')
    print(f'  Omega: {initial_omega:.4f}')

    print(f'\nFinal Configuration:')
    print(f'  Grades: {cpd_result["final_grade_symbols"]}')
    print(f'  Number of grades: {len(cpd_result["final_grade_symbols"])}')
    print(f'  Omega: {cpd_result["final_omega"]:.4f}')
    print(f'  Total iterations: {cpd_result["iterations"]}')

    # Compute improvement
    omega_improvement = cpd_result['final_omega'] - initial_omega
    print(f'\nOmega Change: {omega_improvement:+.4f}')

    if omega_improvement > 0:
        print('[OK] Omega IMPROVED')
    elif omega_improvement < 0:
        print('[INFO] Omega DECREASED')
    else:
        print('[INFO] Omega UNCHANGED')

    # Display refinement history
    if cpd_result['history']:
        print(f'\nRefinement History:')
        for h in cpd_result['history']:
            print(f"  Iteration {h['iteration']}: Removed '{h['removed_label']}' "
                  f"(Omega: {h['omega']:.4f}, Grades: {len(h['grade_symbols'])})")
    else:
        print(f'\nNo refinement performed (fairness already satisfied)')

    # Display final grade distribution
    print(f'\nFinal Grade Distribution:')
    final_dist = cpd_result['final_grades_df'].groupby('Grade')['Score'].agg(['count', 'min', 'max'])
    print(final_dist)

    # Validation
    print(f'\n{"="*80}')
    print('VALIDATION')
    print(f'{"="*80}')

    # Check for empty grades
    empty_grades = []
    for grade in cpd_result['final_grade_symbols']:
        count = len(cpd_result['final_grades_df'][cpd_result['final_grades_df']['Grade'] == grade])
        if count == 0:
            empty_grades.append(grade)

    if empty_grades:
        print(f'[WARNING] Empty grades found: {empty_grades}')
    else:
        print('[OK] No empty grades')

    # Check label reduction
    grades_removed = len(grade_symbols) - len(cpd_result['final_grade_symbols'])
    if grades_removed > 0:
        print(f'[OK] Labels removed: {grades_removed} ({len(grade_symbols)} -> {len(cpd_result["final_grade_symbols"])})')
    else:
        print(f'[INFO] No labels removed ({len(cpd_result["final_grade_symbols"])} grades)')

    # Check iterations
    if cpd_result['iterations'] > 1:
        print(f'[OK] Algorithm ran for {cpd_result["iterations"]} iterations')
    else:
        print(f'[INFO] Algorithm stopped at iteration 1')

# Save results
    output_dir = './file/output/kmean'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'kmeans_cpd_result.xlsx')
    cpd_result['final_grades_df'].to_excel(output_file, index=False)
    print(f'\n[OK] Results saved to: {output_file}')

    print(f'\n{"="*80}')
    print('CPD K-MEANS GRADING COMPLETE')
    print(f'{"="*80}')

    return cpd_result


if __name__ == "__main__":
    result = main()
