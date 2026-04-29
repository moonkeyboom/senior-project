import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import zscore
from itertools import combinations

# ============================================================================
# CPD REFINEMENT ALGORITHM - THETA-BASED LOOP (PAPER ALGORITHM 1)
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
        assigned_grade = grade_symbols[-1]
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
# CPD HELPER FUNCTIONS (UNCHANGED)
# ============================================================================

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


def reevaluate_segmentation(scores, reduced_grade_symbols):
    '''Re-runs the grading segmentation with reduced grade labels'''
    if len(reduced_grade_symbols) < 2:
        raise ValueError('Cannot have fewer than 2 grade symbols')
    return grading_by_heuristic(scores, reduced_grade_symbols)


# ============================================================================
# CPD REFINEMENT LOOPS
# ============================================================================

def cpd_refinement_old_loop(scores, initial_grades_df, grade_symbols,
                            upper_bound=100, lower_bound=0, max_iterations=10):
    '''
    OLD VERSION: Fixed iteration loop (INCORRECT per paper)
    while iteration < max_iterations
    '''
    current_grades_df = initial_grades_df.copy()
    current_grades_df = current_grades_df.reset_index(drop=True)
    current_grade_symbols = grade_symbols.copy()
    iteration = 0
    history = []

    print('\n' + '='*80)
    print('OLD LOOP: Fixed Iteration (iteration < max_iterations)')
    print('='*80)

    while iteration < max_iterations:
        iteration += 1

        print(f'\n--- Iteration {iteration} ---')

        gaps = compute_score_gaps(scores, upper_bound, lower_bound)
        pvis = compute_grade_pvis(current_grades_df)

        max_gap_value = np.max(gaps) if len(gaps) > 0 else 0
        max_pvi_value = get_max_pvi(pvis)

        print(f'Max gap: {max_gap_value:.4f}, Max PVI: {max_pvi_value:.4f}')

        if max_gap_value <= max_pvi_value:
            print(f'SATISFIED: {max_gap_value:.4f} <= {max_pvi_value:.4f}')
            print('Stopping refinement.')
            break

        print(f'VIOLATION: {max_gap_value:.4f} > {max_pvi_value:.4f}')

        max_gap_indices = get_max_gap_indices(gaps)
        print(f'Max gap indices: {max_gap_indices}')

        best_candidate = None
        best_omega = -1.0
        extended_scores = [upper_bound] + scores.tolist() + [lower_bound]

        for idx, gap_idx in enumerate(max_gap_indices, 1):
            label_to_remove = determine_label_to_remove(
                gap_idx, np.array(extended_scores), current_grade_symbols, current_grades_df
            )

            reduced_grade_symbols = [g for g in current_grade_symbols if g != label_to_remove]

            if len(reduced_grade_symbols) < 2:
                continue

            try:
                candidate_grades_df = reevaluate_segmentation(scores, reduced_grade_symbols)
                candidate_grades_df = candidate_grades_df.reset_index(drop=True)
                candidate_omega = calculate_omega(candidate_grades_df, reduced_grade_symbols, scores)

                print(f'  Candidate {idx}: Remove {label_to_remove}, Omega: {candidate_omega:.4f}')

                if candidate_omega > best_omega:
                    best_omega = candidate_omega
                    best_candidate = {
                        'grades_df': candidate_grades_df,
                        'grade_symbols': reduced_grade_symbols,
                        'omega': candidate_omega,
                        'removed_label': label_to_remove
                    }
            except Exception as e:
                print(f'  Candidate {idx}: Error - {e}')
                continue

        if best_candidate is None:
            print('No valid candidate found. Stopping.')
            break

        print(f'Selected: Remove {best_candidate["removed_label"]}, Omega: {best_candidate["omega"]:.4f}')

        current_grades_df = best_candidate['grades_df'].reset_index(drop=True)
        current_grade_symbols = best_candidate['grade_symbols']

        history.append({
            'iteration': iteration,
            'removed_label': best_candidate['removed_label'],
            'omega': best_candidate['omega'],
            'grade_symbols': current_grade_symbols.copy()
        })

    final_omega = calculate_omega(current_grades_df, current_grade_symbols, scores)

    return {
        'final_grades_df': current_grades_df,
        'final_grade_symbols': current_grade_symbols,
        'final_omega': final_omega,
        'iterations': iteration,
        'history': history
    }


def cpd_refinement_theta_loop(scores, initial_grades_df, grade_symbols,
                              upper_bound=100, lower_bound=0):
    '''
    NEW VERSION: Theta-based loop (CORRECT per paper Algorithm 1)
    while j <= theta
    '''
    current_grades_df = initial_grades_df.copy()
    current_grades_df = current_grades_df.reset_index(drop=True)
    current_grade_symbols = grade_symbols.copy()

    # Initialize according to paper
    initial_omega = calculate_omega(current_grades_df, grade_symbols, scores)
    best_omega = initial_omega

    # Check if refinement is needed
    gaps = compute_score_gaps(scores, upper_bound, lower_bound)
    pvis = compute_grade_pvis(current_grades_df)
    max_gap_value = np.max(gaps) if len(gaps) > 0 else 0
    max_pvi_value = get_max_pvi(pvis)

    # Set theta based on initial fairness check
    if max_gap_value > max_pvi_value:
        theta = 1  # Allow at least one iteration to fix violation
    else:
        theta = 0  # No violation, no iteration needed

    j = 1
    history = []

    print('\n' + '='*80)
    print('NEW LOOP: Theta-based (j <= theta) - PAPER ALGORITHM 1')
    print('='*80)
    print(f'Initialization:')
    print(f'  Initial omega: {initial_omega:.4f}')
    print(f'  Max gap: {max_gap_value:.4f}, Max PVI: {max_pvi_value:.4f}')
    print(f'  Fairness satisfied: {max_gap_value <= max_pvi_value}')
    print(f'  Initial theta: {theta}')
    print(f'  Loop condition: while j <= theta')

    while j <= theta:
        print(f'\n' + '='*60)
        print(f'Iteration j={j}, theta={theta}')
        print(f'Loop condition: {j} <= {theta} = {j <= theta}')
        print('='*60)

        # Generate candidates
        gaps = compute_score_gaps(scores, upper_bound, lower_bound)
        pvis = compute_grade_pvis(current_grades_df)

        max_gap_value = np.max(gaps) if len(gaps) > 0 else 0
        max_pvi_value = get_max_pvi(pvis)

        print(f'\nCurrent state:')
        print(f'  Max gap: {max_gap_value:.4f}')
        print(f'  Max PVI: {max_pvi_value:.4f}')
        print(f'  Current omega: {best_omega:.4f}')
        print(f'  Current grades: {current_grade_symbols}')

        if max_gap_value <= max_pvi_value:
            print(f'\nFairness SATISFIED: {max_gap_value:.4f} <= {max_pvi_value:.4f}')
            print('Stopping refinement early.')
            break

        print(f'\nFairness VIOLATED: {max_gap_value:.4f} > {max_pvi_value:.4f}')
        print('Generating candidates...')

        max_gap_indices = get_max_gap_indices(gaps)
        print(f'Max gap indices: {max_gap_indices}')

        best_candidate = None
        best_candidate_omega = -1.0
        extended_scores = [upper_bound] + scores.tolist() + [lower_bound]

        for idx, gap_idx in enumerate(max_gap_indices, 1):
            label_to_remove = determine_label_to_remove(
                gap_idx, np.array(extended_scores), current_grade_symbols, current_grades_df
            )

            reduced_grade_symbols = [g for g in current_grade_symbols if g != label_to_remove]

            if len(reduced_grade_symbols) < 2:
                continue

            try:
                candidate_grades_df = reevaluate_segmentation(scores, reduced_grade_symbols)
                candidate_grades_df = candidate_grades_df.reset_index(drop=True)
                candidate_omega = calculate_omega(candidate_grades_df, reduced_grade_symbols, scores)

                print(f'  Candidate {idx}: Remove {label_to_remove}, Omega: {candidate_omega:.4f}')

                if candidate_omega > best_candidate_omega:
                    best_candidate_omega = candidate_omega
                    best_candidate = {
                        'grades_df': candidate_grades_df,
                        'grade_symbols': reduced_grade_symbols,
                        'omega': candidate_omega,
                        'removed_label': label_to_remove
                    }
            except Exception as e:
                print(f'  Candidate {idx}: Error - {e}')
                continue

        if best_candidate is None:
            print('\nNo valid candidate found. Stopping.')
            break

        # UPDATE RULE (per paper)
        print(f'\n--- UPDATE STEP ---')
        print(f'Best candidate omega: {best_candidate["omega"]:.4f}')
        print(f'Current best omega: {best_omega:.4f}')

        if best_candidate['omega'] > best_omega:
            print(f'IMPROVEMENT: {best_candidate["omega"]:.4f} > {best_omega:.4f}')
            print(f'UPDATE: theta = j = {j}')

            best_omega = best_candidate['omega']
            theta = j  # Update theta to current iteration

            current_grades_df = best_candidate['grades_df'].reset_index(drop=True)
            current_grade_symbols = best_candidate['grade_symbols']

            history.append({
                'iteration': j,
                'removed_label': best_candidate['removed_label'],
                'omega': best_candidate['omega'],
                'grade_symbols': current_grade_symbols.copy()
            })

            print(f'Selected: Remove {best_candidate["removed_label"]}')
            print(f'New grades: {current_grade_symbols}')
        else:
            print(f'NO IMPROVEMENT: {best_candidate["omega"]:.4f} <= {best_omega:.4f}')
            print(f'DO NOT UPDATE theta (remains {theta})')
            print(f'Stopping refinement.')

            # Even though we don't update, we still apply the change
            current_grades_df = best_candidate['grades_df'].reset_index(drop=True)
            current_grade_symbols = best_candidate['grade_symbols']

            history.append({
                'iteration': j,
                'removed_label': best_candidate['removed_label'],
                'omega': best_candidate['omega'],
                'grade_symbols': current_grade_symbols.copy()
            })

            break  # Stop if no improvement

        # Increment j
        j = j + 1

        print(f'\nIncrement: j = {j}')
        print(f'Next loop condition: {j} <= {theta} = {j <= theta}')

    # Check final condition
    print(f'\n' + '='*80)
    print('LOOP TERMINATED')
    print(f'Final j: {j}')
    print(f'Final theta: {theta}')
    print(f'Termination reason: j ({j}) > theta ({theta})')
    print('='*80)

    final_omega = calculate_omega(current_grades_df, current_grade_symbols, scores)

    return {
        'final_grades_df': current_grades_df,
        'final_grade_symbols': current_grade_symbols,
        'final_omega': final_omega,
        'iterations': j - 1,  # Actual iterations performed
        'theta': theta,
        'history': history
    }


# ============================================================================
# COMPARISON TEST
# ============================================================================

def compare_cpd_loops():
    '''Compare old loop vs new theta-based loop'''

    # Load data
    df = pd.read_excel('./file/input/221.xlsx')
    scores = df['Score'].values
    sorted_scores = np.sort(scores)[::-1]

    print('='*80)
    print('CPD REFINEMENT LOOP COMPARISON TEST')
    print('='*80)
    print(f'\nDataset: 221.xlsx')
    print(f'Total students: {len(scores)}')
    print(f'Score range: {scores.min():.2f} - {scores.max():.2f}')

    # Validate gap calculation
    gaps = compute_score_gaps(sorted_scores, upper_bound=100, lower_bound=0)
    print(f'\n--- Gap Validation ---')
    print(f'Max gap: {np.max(gaps):.4f}')
    if np.max(gaps) == 35.0:
        print('[OK] Max gap is 35.0 as expected')
    else:
        print(f'[ERR] Max gap should be 35.0, got {np.max(gaps):.4f}')

    # Initial grading
    grade_symbols = ['A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F']
    print(f'\nInitial grading with {len(grade_symbols)} grades')

    initial_grades_df = grading_by_heuristic(sorted_scores, grade_symbols)
    initial_omega = calculate_omega(initial_grades_df, grade_symbols, sorted_scores)

    print(f'Initial omega: {initial_omega:.4f}')

    # Run OLD loop
    print('\n' + '='*80)
    print('RUNNING OLD LOOP (Fixed Iteration)')
    print('='*80)

    old_result = cpd_refinement_old_loop(
        sorted_scores,
        initial_grades_df.copy(),
        grade_symbols.copy(),
        max_iterations=10
    )

    # Run NEW loop
    print('\n' + '='*80)
    print('RUNNING NEW LOOP (Theta-based)')
    print('='*80)

    new_result = cpd_refinement_theta_loop(
        sorted_scores,
        initial_grades_df.copy(),
        grade_symbols.copy()
    )

    # Compare results
    print('\n' + '='*80)
    print('COMPARISON RESULTS')
    print('='*80)

    print(f'\nOLD LOOP (Fixed Iteration):')
    print(f'  Total iterations: {old_result["iterations"]}')
    print(f'  Final omega: {old_result["final_omega"]:.4f}')
    print(f'  Final grades: {old_result["final_grade_symbols"]}')
    print(f'  Number of grades: {len(old_result["final_grade_symbols"])}')

    print(f'\nNEW LOOP (Theta-based):')
    print(f'  Total iterations: {new_result["iterations"]}')
    print(f'  Final theta: {new_result["theta"]}')
    print(f'  Final omega: {new_result["final_omega"]:.4f}')
    print(f'  Final grades: {new_result["final_grade_symbols"]}')
    print(f'  Number of grades: {len(new_result["final_grade_symbols"])}')

    print(f'\n--- Comparison ---')
    if new_result['iterations'] > old_result['iterations']:
        print(f'[OK] New loop ran more iterations: {new_result["iterations"]} vs {old_result["iterations"]}')
    elif new_result['iterations'] < old_result['iterations']:
        print(f'[INFO] New loop ran fewer iterations: {new_result["iterations"]} vs {old_result["iterations"]}')
    else:
        print(f'[SAME] Both loops ran {old_result["iterations"]} iterations')

    omega_diff = new_result['final_omega'] - old_result['final_omega']
    if omega_diff > 0.001:
        print(f'[OK] New loop has BETTER omega: {new_result["final_omega"]:.4f} vs {old_result["final_omega"]:.4f}')
    elif omega_diff < -0.001:
        print(f'[INFO] New loop has LOWER omega: {new_result["final_omega"]:.4f} vs {old_result["final_omega"]:.4f}')
    else:
        print(f'[SAME] Both loops have similar omega: ~{old_result["final_omega"]:.4f}')

    if len(new_result['final_grade_symbols']) < len(old_result['final_grade_symbols']):
        print(f'[INFO] New loop has fewer grades: {len(new_result["final_grade_symbols"])} vs {len(old_result["final_grade_symbols"])}')
    elif len(new_result['final_grade_symbols']) > len(old_result['final_grade_symbols']):
        print(f'[INFO] New loop has more grades: {len(new_result["final_grade_symbols"])} vs {len(old_result["final_grade_symbols"])}')
    else:
        print(f'[SAME] Both loops have {len(old_result["final_grade_symbols"])} grades')

    # Display history comparison
    print(f'\n--- Refinement History Comparison ---')

    print(f'\nOLD LOOP History:')
    if old_result['history']:
        for h in old_result['history']:
            print(f"  Iteration {h['iteration']}: Removed '{h['removed_label']}' "
                  f"(Omega: {h['omega']:.4f}, Grades: {len(h['grade_symbols'])})")
    else:
        print('  No refinement performed')

    print(f'\nNEW LOOP History:')
    if new_result['history']:
        for h in new_result['history']:
            print(f"  Iteration j={h['iteration']}: Removed '{h['removed_label']}' "
                  f"(Omega: {h['omega']:.4f}, Grades: {len(h['grade_symbols'])})")
    else:
        print('  No refinement performed')

    # Display final distributions
    print(f'\n--- Final Grade Distribution Comparison ---')

    print(f'\nOLD LOOP Distribution:')
    old_dist = old_result['final_grades_df'].groupby('Grade')['Score'].agg(['count', 'min', 'max'])
    print(old_dist)

    print(f'\nNEW LOOP Distribution:')
    new_dist = new_result['final_grades_df'].groupby('Grade')['Score'].agg(['count', 'min', 'max'])
    print(new_dist)

    # Save NEW loop result
    output_df = new_result['final_grades_df'].copy()
    output_file = './file/output/heuristic/cpd_refined_theta_version.xlsx'
    output_df.to_excel(output_file, index=False)
    print(f'\n[OK] NEW LOOP result saved to: {output_file}')

    return {
        'old_result': old_result,
        'new_result': new_result
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = compare_cpd_loops()

    print('\n' + '='*80)
    print('TEST COMPLETE')
    print('='*80)
    print('\nSummary:')
    print('  - Gap validation: PASSED' if np.max(compute_score_gaps(np.sort(pd.read_excel('./file/input/221.xlsx')['Score'].values)[::-1])) == 35.0 else '  - Gap validation: FAILED')
    print(f"  - Old loop iterations: {results['old_result']['iterations']}")
    print(f"  - New loop iterations: {results['new_result']['iterations']}")
    print(f"  - Old loop omega: {results['old_result']['final_omega']:.4f}")
    print(f"  - New loop omega: {results['new_result']['final_omega']:.4f}")
