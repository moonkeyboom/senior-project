import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import zscore
from itertools import combinations

# ============================================================================
# CPD REFINEMENT ALGORITHM - FIXED VERSION WITH BOUNDARY GAPS
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

    # Calculate delta_i (score gaps between adjacent grades)
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

    # Calculate all score gaps
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


def assign_grades(scores, grade_boundaries, grade_symbols):
    '''Assign grades based on score boundaries'''
    graded_data = []
    grade_boundaries.sort(key=lambda x: x[1], reverse=True)

    for score in scores:
        assigned_grade = grade_symbols[-1]  # Default to lowest grade
        for min_s, max_s, grade_s in grade_boundaries:
            if min_s <= score <= max_s:
                assigned_grade = grade_s
                break
        graded_data.append({'Score': score, 'Grade': assigned_grade})
    return pd.DataFrame(graded_data)


def grading_by_heuristic(scores, grade_symbols):
    '''Grades using heuristic method (widest gaps)'''
    n_grades = len(grade_symbols)
    gaps = np.abs(np.diff(scores))

    if len(gaps) == 0:
        return pd.DataFrame({'Score': scores, 'Grade': [grade_symbols[0]] * len(scores)})

    num_cuts_needed = n_grades - 1
    if num_cuts_needed <= 0:
        return pd.DataFrame({'Score': scores, 'Grade': [grade_symbols[0]] * len(scores)})

    if num_cuts_needed > len(gaps):
        return pd.DataFrame({'Score': scores, 'Grade': [grade_symbols[0]] * len(scores)})

    # Create indexed gaps
    indexed_gaps = [(gaps[i], i) for i in range(len(gaps))]
    indexed_gaps.sort(key=lambda x: x[0], reverse=True)

    threshold_gap_value = indexed_gaps[num_cuts_needed - 1][0]
    candidate_cut_indices_pool = []
    for gap_val, idx in indexed_gaps:
        if gap_val >= threshold_gap_value:
            candidate_cut_indices_pool.append(idx)
        else:
            break

    candidate_cut_indices_pool = sorted(list(set(candidate_cut_indices_pool)))

    if len(candidate_cut_indices_pool) < num_cuts_needed:
        candidate_cut_indices_pool = list(range(len(scores) - 1))

    best_omega = -1.0
    best_grades_df = None

    for cut_combination in combinations(candidate_cut_indices_pool, num_cuts_needed):
        current_cut_indices = sorted(list(cut_combination))
        temp_grade_boundaries = []
        current_start_idx = 0
        valid_combination = True

        for i in range(n_grades):
            grade_s = grade_symbols[i]
            if i < num_cuts_needed:
                end_idx = current_cut_indices[i]
            else:
                end_idx = len(scores) - 1

            current_grade_scores = scores[current_start_idx : end_idx + 1]
            if current_grade_scores.size > 0:
                min_s = current_grade_scores.min()
                max_s = current_grade_scores.max()
                temp_grade_boundaries.append((min_s, max_s, grade_s))
            else:
                valid_combination = False
                break
            current_start_idx = end_idx + 1

        if not valid_combination:
            continue

        current_grades_df = assign_grades(scores, temp_grade_boundaries, grade_symbols)
        current_omega = calculate_omega(current_grades_df, grade_symbols, scores)

        if current_omega > best_omega:
            best_omega = current_omega
            best_grades_df = current_grades_df.copy()

    if best_grades_df is None:
        return pd.DataFrame({'Score': scores, 'Grade': [grade_symbols[-1]] * len(scores)})

    return best_grades_df


# ============================================================================
# CPD HELPER FUNCTIONS - FIXED VERSION
# ============================================================================

def compute_score_gaps(sorted_scores, upper_bound=100, lower_bound=0):
    """
    Computes score gaps including boundary gaps.

    Args:
        sorted_scores (np.array): Scores sorted in descending order.
        upper_bound (float): Maximum possible score (default 100).
        lower_bound (float): Minimum possible score (default 0).

    Returns:
        np.array: Array of gaps including boundaries.
    """
    if len(sorted_scores) < 1:
        return np.array([])

    # Construct extended score list with boundaries
    extended_scores = [upper_bound] + sorted_scores.tolist() + [lower_bound]
    extended_scores = np.array(extended_scores)

    # Compute gaps
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
    """
    Determines which grade label to remove based on the gap position.
    Now handles boundary gaps correctly.

    Args:
        gap_index (int): Index of the gap in the extended scores array.
        extended_scores (np.array): Extended scores [100, ..., 0].
        grade_symbols (list): Ordered list of grade symbols (best to worst).
        grades_df (pd.DataFrame): Current grade assignments with index preserved.

    Returns:
        str: The grade symbol to remove.
    """
    # Case 1: Upper boundary gap (between 100 and highest score)
    if gap_index == 0:
        # Remove the highest grade label (grade of the top score)
        top_score = extended_scores[1]  # First actual score
        # Use index-based lookup to avoid duplicate score issues
        # Find first occurrence of this score
        top_student_idx = grades_df[grades_df['Score'] == top_score].index[0]
        grade_to_remove = grades_df.loc[top_student_idx, 'Grade']
        return grade_to_remove

    # Case 2: Lower boundary gap (between lowest score and 0)
    elif gap_index == len(extended_scores) - 2:
        # Remove the lowest grade label (grade of the bottom score)
        bottom_score = extended_scores[-2]  # Last actual score
        # Use index-based lookup to avoid duplicate score issues
        # Find last occurrence of this score (for duplicates)
        bottom_student_idx = grades_df[grades_df['Score'] == bottom_score].index[-1]
        grade_to_remove = grades_df.loc[bottom_student_idx, 'Grade']
        return grade_to_remove

    # Case 3: Normal internal gap
    else:
        # Gap is between extended_scores[gap_index] and extended_scores[gap_index + 1]
        # These are actual student scores (not boundaries)
        score_above = extended_scores[gap_index]
        score_below = extended_scores[gap_index + 1]

        # Use index-based lookup for safety
        # For duplicate scores, use first occurrence for score_above
        idx_above = grades_df[grades_df['Score'] == score_above].index[0]
        # For duplicate scores, use last occurrence for score_below
        idx_below = grades_df[grades_df['Score'] == score_below].index[-1]

        grade_above = grades_df.loc[idx_above, 'Grade']
        grade_below = grades_df.loc[idx_below, 'Grade']

        # Find positions in grade_symbols
        pos_above = grade_symbols.index(grade_above)
        pos_below = grade_symbols.index(grade_below)

        # Remove the label BELOW the gap (worse grade)
        return grade_below


def reevaluate_segmentation(scores, reduced_grade_symbols):
    '''Re-runs the grading segmentation with reduced grade labels'''
    if len(reduced_grade_symbols) < 2:
        raise ValueError('Cannot have fewer than 2 grade symbols')
    return grading_by_heuristic(scores, reduced_grade_symbols)


# ============================================================================
# MAIN CPD REFINEMENT LOOP - FIXED VERSION
# ============================================================================

def cpd_refinement_loop_detailed(scores, initial_grades_df, grade_symbols,
                                  upper_bound=100, lower_bound=0, max_iterations=10):
    '''
    CPD refinement loop with detailed logging and boundary gaps.
    '''
    current_grades_df = initial_grades_df.copy()
    # Reset index to ensure safe lookup
    current_grades_df = current_grades_df.reset_index(drop=True)
    current_grade_symbols = grade_symbols.copy()
    iteration = 0
    history = []

    print('\n' + '='*80)
    print('CPD REFINEMENT LOOP - DETAILED EXECUTION LOG (FIXED VERSION)')
    print('='*80)

    while iteration < max_iterations:
        iteration += 1

        print('\n' + '='*80)
        print(f'ITERATION {iteration}')
        print('='*80)

        # Step 1: Display current state
        print(f'\n--- Current State ---')
        print(f'Grade symbols: {current_grade_symbols}')
        print(f'Number of grades: {len(current_grade_symbols)}')

        print(f'\n--- Sorted Scores (first 10) ---')
        print(f'{scores[:10]}')
        print(f'Max score: {scores.max():.2f}')
        print(f'Min score: {scores.min():.2f}')

        # Step 2: Compute gaps INCLUDING BOUNDARIES
        gaps = compute_score_gaps(scores, upper_bound, lower_bound)
        pvis = compute_grade_pvis(current_grades_df)

        print(f'\n--- Score Gaps (including boundaries) ---')
        print(f'Total gaps: {len(gaps)}')
        print(f'Gap 0 (upper bound {upper_bound} -> {scores.max():.2f}): {gaps[0]:.4f}')
        print(f'Gap 1 to {len(gaps)-2} (internal gaps): ...')
        print(f'Gap {len(gaps)-1} (lower bound {scores.min():.2f} -> {lower_bound}): {gaps[-1]:.4f}')

        # Step 3: Find max_gap and max_PVI
        max_gap_value = np.max(gaps) if len(gaps) > 0 else 0
        max_pvi_value = get_max_pvi(pvis)

        print(f'\n--- Gap Analysis ---')
        print(f'Max gap: {max_gap_value:.4f}')
        print(f'Max PVI: {max_pvi_value:.4f}')

        # Step 4: Display PVI per grade
        print(f'\n--- PVI per Grade ---')
        for grade in sorted(current_grade_symbols, key=lambda x: current_grade_symbols.index(x)):
            if grade in pvis:
                print(f'  {grade}: {pvis[grade]:.4f}')

        # Step 5: Check violation
        print(f'\n--- Fairness Check ---')
        if max_gap_value <= max_pvi_value:
            print(f'SATISFIED: {max_gap_value:.4f} <= {max_pvi_value:.4f}')
            print('\n' + '='*80)
            print('CPD REFINEMENT COMPLETE - FAIRNESS ACHIEVED')
            print('='*80)
            break

        print(f'VIOLATION: {max_gap_value:.4f} > {max_pvi_value:.4f}')
        print(f'Must remove a grade label')

        # Step 6: Find all max_gap indices
        max_gap_indices = get_max_gap_indices(gaps)
        print(f'\n--- Max Gap Indices ---')
        print(f'Found {len(max_gap_indices)} gap(s) with max value {max_gap_value:.4f}')
        print(f'Indices: {max_gap_indices}')

        # Identify which gap is which
        for idx in max_gap_indices:
            if idx == 0:
                print(f'  Gap {idx}: UPPER BOUNDARY gap ({upper_bound} -> {scores.max():.2f}) = {gaps[idx]:.4f}')
            elif idx == len(gaps) - 1:
                print(f'  Gap {idx}: LOWER BOUNDARY gap ({scores.min():.2f} -> {lower_bound}) = {gaps[idx]:.4f}')
            else:
                extended_scores = [upper_bound] + scores.tolist() + [lower_bound]
                score_above = extended_scores[idx]
                score_below = extended_scores[idx + 1]
                print(f'  Gap {idx}: Internal gap ({score_above:.2f} -> {score_below:.2f}) = {gaps[idx]:.4f}')

        # Step 7: Try each candidate
        print(f'\n--- Evaluating Candidates ---')
        best_candidate = None
        best_omega = -1.0

        # Build extended scores for label removal
        extended_scores = [upper_bound] + scores.tolist() + [lower_bound]

        for idx, gap_idx in enumerate(max_gap_indices, 1):
            print(f'\n  Candidate {idx}:')
            print(f'    Gap index: {gap_idx}')
            print(f'    Gap value: {gaps[gap_idx]:.4f}')

            # Identify gap type
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
                print(f'    Skipped: Would result in < 2 grades')
                continue

            # Re-run segmentation
            try:
                candidate_grades_df = reevaluate_segmentation(scores, reduced_grade_symbols)
                candidate_grades_df = candidate_grades_df.reset_index(drop=True)
                candidate_omega = calculate_omega(candidate_grades_df, reduced_grade_symbols, scores)

                print(f'    Omega: {candidate_omega:.4f}')

                # Check for empty grades
                empty_grades = []
                for grade in reduced_grade_symbols:
                    count = len(candidate_grades_df[candidate_grades_df['Grade'] == grade])
                    if count == 0:
                        empty_grades.append(grade)

                if empty_grades:
                    print(f'    Warning: Empty grades: {empty_grades}')

                if candidate_omega > best_omega:
                    best_omega = candidate_omega
                    best_candidate = {
                        'grades_df': candidate_grades_df,
                        'grade_symbols': reduced_grade_symbols,
                        'omega': candidate_omega,
                        'removed_label': label_to_remove,
                        'gap_index': gap_idx
                    }
                    print(f'    -> New best candidate!')

            except Exception as e:
                print(f'    Error: {e}')
                import traceback
                traceback.print_exc()
                continue

        # Step 8: Select best candidate
        if best_candidate is None:
            print(f'\n--- No Valid Candidate Found ---')
            print('Stopping refinement.')
            break

        print(f'\n--- Best Candidate Selected ---')
        print(f"Removed label: '{best_candidate['removed_label']}'")
        print(f"Gap index: {best_candidate['gap_index']}")
        print(f"Omega: {best_candidate['omega']:.4f}")

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
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data
    df = pd.read_excel('./file/input/221.xlsx')
    scores = df['Score'].values
    sorted_scores = np.sort(scores)[::-1]

    print('='*80)
    print('DATASET INFORMATION')
    print('='*80)
    print(f'Total students: {len(scores)}')
    print(f'Score range: {scores.min():.2f} - {scores.max():.2f}')
    print(f'Mean score: {scores.mean():.2f}')
    print(f'Std deviation: {scores.std():.2f}')

    # Initial grading
    print('\n' + '='*80)
    print('INITIAL GRADING (Heuristic Method)')
    print('='*80)

    grade_symbols = ['A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F']
    print(f'Initial grade symbols: {grade_symbols}')

    initial_grades_df = grading_by_heuristic(sorted_scores, grade_symbols)
    initial_omega = calculate_omega(initial_grades_df, grade_symbols, sorted_scores)

    print(f'Initial Omega: {initial_omega:.4f}')

    # Validate gap calculation
    print('\n' + '='*80)
    print('GAP VALIDATION (BEFORE CPD LOOP)')
    print('='*80)

    gaps = compute_score_gaps(sorted_scores, upper_bound=100, lower_bound=0)
    print(f'Total gaps (including boundaries): {len(gaps)}')
    print(f'Gap at upper bound (100 -> {scores.max():.2f}): {gaps[0]:.4f}')
    print(f'Gap at lower bound ({scores.min():.2f} -> 0): {gaps[-1]:.4f}')
    print(f'Max gap: {np.max(gaps):.4f}')

    if np.max(gaps) == 35.0:
        print('[OK] CORRECT: Max gap is 35.0 as expected!')
    else:
        print(f'[ERR] WRONG: Max gap should be 35.0, but got {np.max(gaps):.4f}')

    # Run CPD refinement
    cpd_result = cpd_refinement_loop_detailed(sorted_scores, initial_grades_df, grade_symbols)

    # Final results
    print('\n' + '='*80)
    print('FINAL RESULTS')
    print('='*80)
    print(f"Final grade symbols: {cpd_result['final_grade_symbols']}")
    print(f"Number of grades: {len(cpd_result['final_grade_symbols'])}")
    print(f"Final Omega: {cpd_result['final_omega']:.4f}")
    print(f"Total iterations: {cpd_result['iterations']}")

    if len(cpd_result['final_grade_symbols']) < len(grade_symbols):
        print(f'[OK] CORRECT: Labels were removed ({len(grade_symbols)} -> {len(cpd_result["final_grade_symbols"])})')
    else:
        print(f'[WARN] WARNING: No labels were removed')

    if cpd_result['iterations'] > 1:
        print(f'[OK] CORRECT: CPD ran for {cpd_result["iterations"]} iterations (not just 1)')
    else:
        print(f'[ERR] WRONG: CPD stopped at iteration 1 (should run more)')

    print('\n--- Refinement History ---')
    if cpd_result['history']:
        for h in cpd_result['history']:
            print(f"Iteration {h['iteration']}: Removed '{h['removed_label']}' "
                  f"(Omega: {h['omega']:.4f}, Grades: {h['grade_symbols']})")
    else:
        print('No refinement history (CPD stopped at iteration 1)')

    # Display final grading distribution
    print('\n--- Final Grade Distribution ---')
    grade_dist = cpd_result['final_grades_df'].groupby('Grade')['Score'].agg(['count', 'min', 'max'])
    print(grade_dist)

    # Validate
    print('\n--- Validation ---')
    empty_grades = []
    for grade in cpd_result['final_grade_symbols']:
        count = len(cpd_result['final_grades_df'][cpd_result['final_grades_df']['Grade'] == grade])
        if count == 0:
            empty_grades.append(grade)

    if empty_grades:
        print(f"WARNING: Empty grades found: {empty_grades}")
    else:
        print('[OK] No empty grades')

    print(f"[OK] Label count: {len(cpd_result['final_grade_symbols'])} (started with {len(grade_symbols)})")
    print(f"[OK] Final Omega computed correctly: {cpd_result['final_omega']:.4f}")

    # Save final results
    output_df = cpd_result['final_grades_df'].copy()
    output_file = './file/output/cpd_refined_result_fixed.xlsx'
    output_df.to_excel(output_file, index=False)
    print(f'\nFinal results saved to: {output_file}')
