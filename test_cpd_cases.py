import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optimal_cpd_omega_prime import (
    exhaustive_best_for_k, dp_best, dp_best as _dp,
    exhaustive_optimal
)

def generate_data():
    os.makedirs('data', exist_ok=True)
    np.random.seed(42)
    
    test_cases = {
        'tc1_uniform': np.linspace(10, 90, 30),
        'tc2_bimodal': np.concatenate([np.random.normal(30, 5, 15), np.random.normal(80, 5, 15)]),
        'tc3_outlier_high': np.concatenate([np.random.normal(40, 5, 29), [99]]),
        'tc4_outlier_low': np.concatenate([[10], np.random.normal(80, 5, 29)]),
        'tc5_skewed': np.clip(np.random.exponential(scale=15, size=30) + 20, 0, 100),
        'tc6_normal': np.random.normal(50, 15, 30),
        'tc7_narrow': np.random.normal(50, 0.5, 30),
        'tc8_small_n': np.array([20, 40, 60, 80, 95]),
        'tc9_five_groups': np.concatenate([np.random.normal(loc, 2, 6) for loc in [20, 40, 60, 80, 95]]),
        'tc10_gaps': np.array([10, 12, 40, 42, 70, 72, 98, 100])
    }
    
    files = {}
    for name, data in test_cases.items():
        data = np.clip(data, 0, 100).round(2)
        path = f'data/{name}.csv'
        pd.DataFrame({'score': data}).to_csv(path, index=False)
        files[name] = data
        
    return files

def run_tests_and_plot(files):
    U, L = 100.0, 0.0
    grades = ["A", "B", "C", "D", "F"]
    num_labels = len(grades)
    
    results = []
    
    for name, data in files.items():
        v = np.sort(np.asarray(data, dtype=float))[::-1]
        n = len(v)
        kL = min(num_labels, n)
        
        # Test 1: Constrained (Fixed k = |L|)
        exL = exhaustive_best_for_k(v, kL, num_labels, U, L)
        dpL = dp_best(v, num_labels, U, L, k=kL)
        
        ex_score_L = exL.omega_prime if exL else 0.0
        dp_score_L = dpL.omega_prime if dpL else 0.0
        
        # Test 2: Unconstrained (Best overall)
        ex_opt, _ = exhaustive_optimal(v, num_labels, U, L, k_min=2, k_max=kL)
        dp_opt, _ = dp_best(v, num_labels, U, L, k_min=2, k_max=kL)
        
        ex_score_opt = ex_opt.omega_prime if ex_opt else 0.0
        dp_score_opt = dp_opt.omega_prime if dp_opt else 0.0
        
        results.append({
            'Test Case': name,
            'Exhaustive (Fixed k)': ex_score_L,
            'DP (Fixed k)': dp_score_L,
            'Exhaustive (Overall)': ex_score_opt,
            'DP (Overall)': dp_score_opt
        })
        
    df = pd.DataFrame(results)
    
    # Generate Visual Output (Bar Chart)
    labels = df['Test Case'].str.replace('tc', '').str.replace('_', ' ')
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, df['Exhaustive (Overall)'], width, label='Exhaustive (Optimal)')
    rects2 = ax.bar(x + width/2, df['DP (Overall)'], width, label='DP (Heuristic)')
    
    ax.set_ylabel('Ω′ Score')
    ax.set_title('Ω′ Score Comparison: Exhaustive vs DP-based approach')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    
    fig.tight_layout()
    plt.savefig('summary_chart.png', dpi=300)
    print("Saved visual output to summary_chart.png")
    
    # Save text summary
    with open('test_summary.txt', 'w') as f:
        f.write("# CPD Algorithm Test Summary\n\n")
        f.write("This table compares the Exhaustive (Ground Truth Optimal) approach against the DP-based heuristic.\n\n")
        f.write(df.to_string(index=False))
        
    print(df.to_string(index=False))
    df.to_csv('results.csv', index=False)

if __name__ == '__main__':
    print("Generating data...")
    files = generate_data()
    print(f"Generated {len(files)} test cases in 'data' directory.")
    print("Running tests and generating visual output...")
    run_tests_and_plot(files)
