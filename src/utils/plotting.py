import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from sklearn.metrics import auc
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Plot aggregated results')
    parser.add_argument('--dataset', type=str, default='fried',
                       help='Dataset name (default: fried)')
    return parser.parse_args()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from collections import defaultdict

def plot_results_with_ci(dataset_name="fried", confidence=0.90):
    """Plot aggregated results with confidence intervals for multiple seeds"""
    # Set much larger font sizes
    plt.rcParams.update({
        'font.size': 28,  # Base font size
        'axes.labelsize': 32,  # Axis labels
        'axes.titlesize': 36,  # Title
        'xtick.labelsize': 28,  # X-axis ticks
        'ytick.labelsize': 28,  # Y-axis ticks
        'legend.fontsize': 20,  # Legend
        'lines.linewidth': 5,  # Much thicker lines
        'grid.linewidth': 2,  # Thicker grid
        'axes.linewidth': 2,  # Thicker axes lines
        'font.weight': 'bold',  # Bold font
        'axes.labelweight': 'bold',  # Bold axis labels
        'axes.titleweight': 'bold'  # Bold title
    })
    
    results = []
    #seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    method_aucs = defaultdict(list)
    
    # Read results for all seeds
    for seed in seeds:
        try:
            df = pd.read_csv(f"./results/{dataset_name}/seed_{seed}/addition_experiment_results.csv")
            print(f"Found data for seed {seed}")
            
            # Calculate AUC for each method
            for method in df['Unnamed: 0'].unique():
                method_name = str(method).split('(')[0]
                method_data = df[df['Unnamed: 0'] == method]
                
                y = method_data['remove_least_influential_first_Metrics.ACCURACY'].values
                auc_value = np.sum(y)/(len(y)-1) 
                method_aucs[method_name].append(auc_value)
            
            df['seed'] = seed
            results.append(df)
            
        except Exception as e:
            print(f"Error processing seed {seed}: {e}")
            continue
    
    if not results:
        print("No results found to plot")
        return

    # Calculate AUC statistics for each method
    method_auc_stats = {}
    for method, aucs in method_aucs.items():
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ci_auc = std_auc * stats.t.ppf((1 + confidence) / 2., len(aucs)-1) / np.sqrt(len(aucs))
        method_auc_stats[method] = {
            'mean': mean_auc,
            'ci': ci_auc
        }

    # Combine results
    all_results = pd.concat(results)
    
    # Get unique methods (remove seed-specific numbering)
    methods = all_results.iloc[:, 0].apply(lambda x: str(x).split('(')[0]).unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # First plot - Most valuable points first
    plt.figure(figsize=(16, 16))  # Slightly larger square figure
    
    for idx, method in enumerate(methods):
        method_mask = all_results.iloc[:, 0].apply(lambda x: str(x).split('(')[0]) == method
        method_data = all_results[method_mask]
        
        grouped_data = method_data.groupby('axis')['remove_least_influential_first_Metrics.ACCURACY']
        means = grouped_data.mean()
        std = grouped_data.std()
        ci = std * stats.t.ppf((1 + confidence) / 2., len(seeds)-1) / np.sqrt(len(seeds))
        
        x_values = np.concatenate(([0], (means.index) + 1))
        y_values = np.concatenate(([0], means.values[::-1]))
        
        if len(ci) > 0:
            ci_values = np.concatenate(([0], ci.values[::-1]))
        else:
            ci_values = np.zeros_like(y_values)
        
        plt.plot(x_values, y_values,
                label=f'{method} (AUC={method_auc_stats[method]["mean"]:.4f})',
                color=colors[idx],
                linewidth=5)  # Much thicker lines
        plt.fill_between(x_values, 
                        y_values - ci_values,
                        y_values + ci_values,
                        alpha=0.2,  # More transparent confidence intervals
                        color=colors[idx])
    
    plt.xlabel('Number of Data Points Added', labelpad=20)
    plt.ylabel('Test Accuracy', labelpad=20)
    plt.title(f'Impact of Data Point Addition\n(Most Valuable First) - {dataset_name}')
    plt.ylim(0.1, None)
    
    # Enhanced legend
    plt.legend(loc='lower right', 
              frameon=True,
              framealpha=0.95,
              edgecolor='black',
              borderpad=1.5,
              handlelength=3)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.margins(x=0.02)
    
    # Make tick labels bold
    plt.tick_params(width=2, length=10)  # Thicker and longer ticks
    
    plt.tight_layout()
    
    # Save first plot
    out_dir = f"./results/{dataset_name}/aggregate"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{out_dir}/data_addition_full_most.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Second plot - Least valuable points first
    plt.figure(figsize=(16, 16))
    
    for idx, method in enumerate(methods):
        method_mask = all_results.iloc[:, 0].apply(lambda x: str(x).split('(')[0]) == method
        method_data = all_results[method_mask]
        
        grouped_data = method_data.groupby('axis')['remove_most_influential_first_Metrics.ACCURACY']
        means = grouped_data.mean()
        std = grouped_data.std()
        ci = std * stats.t.ppf((1 + confidence) / 2., len(seeds)-1) / np.sqrt(len(seeds))
        
        x_values = np.concatenate(([0], (means.index) + 1))
        y_values = np.concatenate(([0], means.values[::-1]))
        
        if len(ci) > 0:
            ci_values = np.concatenate(([0], ci.values[::-1]))
        else:
            ci_values = np.zeros_like(y_values)
        
        plt.plot(x_values, y_values,
                label=f'{method} (AUC={method_auc_stats[method]["mean"]:.4f})',
                color=colors[idx],
                linestyle='--',
                linewidth=5)  # Much thicker lines
        plt.fill_between(x_values,
                        y_values - ci_values,
                        y_values + ci_values,
                        alpha=0.2,  # More transparent intervals
                        color=colors[idx])
    
    plt.xlabel('Number of Data Points Added', labelpad=20)
    plt.ylabel('Test Accuracy', labelpad=20)
    plt.title(f'Impact of Data Point Addition\n(Least Valuable First) - {dataset_name}')
    plt.ylim(0.1, None)
    
    # Enhanced legend
    plt.legend(loc='lower right',
              frameon=True,
              framealpha=0.95,
              edgecolor='black',
              borderpad=1.5,
              handlelength=3)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.margins(x=0.02)
    
    # Make tick labels bold
    plt.tick_params(width=2, length=10)  # Thicker and longer ticks
    
    plt.tight_layout()
    
    # Save second plot
    plt.savefig(f"{out_dir}/data_addition_full_least.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    plot_results_with_ci(dataset_name=args.dataset)