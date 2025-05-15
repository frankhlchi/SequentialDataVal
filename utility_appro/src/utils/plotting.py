import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.stats import sem

def aggregate_results(dataset_name):
    """Aggregate results from multiple seed runs and create visualizations"""
    results_dir = f"results/{dataset_name}"
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    # List all seed directories
    seed_dirs = [d for d in os.listdir(results_dir) if d.startswith("seed_")]
    
    # Combine all result CSVs
    all_results = []
    for seed_dir in seed_dirs:
        csv_path = os.path.join(results_dir, seed_dir, "correlation_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_results.append(df)
    
    if not all_results:
        print(f"No result files found for dataset: {dataset_name}")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Create summary dataframe with mean and standard error
    summary_df = combined_df.groupby(['method', 'metric']).agg({
        'value': ['mean', 'std', 'count', lambda x: sem(x)]
    }).reset_index()
    
    # Rename columns
    summary_df.columns = ['method', 'metric', 'mean', 'std', 'count', 'sem']
    
    # Save summary statistics
    os.makedirs("aggregate_results", exist_ok=True)
    summary_df.to_csv(f"aggregate_results/{dataset_name}_summary.csv", index=False)
    
    # Create bar plots for each metric
    metrics = ['spearman_corr', 'kendall_corr', 'ordering_preservation']
    methods = sorted(summary_df['method'].unique())
    
    plt.figure(figsize=(15, 12))
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i+1)
        
        # Filter data for this metric
        metric_data = summary_df[summary_df['metric'] == metric]
        
        # Calculate positions for bars and error bars
        x_pos = np.arange(len(methods))
        
        # Plot bars
        bars = plt.bar(x_pos, 
                     metric_data['mean'], 
                     yerr=metric_data['sem'],
                     align='center',
                     alpha=0.7,
                     ecolor='black',
                     capsize=10)
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}',
                   ha='center', va='bottom', rotation=0)
        
        # Customizing the plot
        plt.ylabel(f"{metric.replace('_', ' ').title()}")
        plt.xticks(x_pos, methods)
        plt.title(f"{metric.replace('_', ' ').title()} by Method for {dataset_name}")
        plt.ylim(0, 1.05)  # All metrics are between 0 and 1
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"aggregate_results/{dataset_name}_metrics_comparison.png", dpi=300, bbox_inches='tight')
    
    # Create line plot showing performance across seeds
    pivot_df = combined_df.pivot_table(
        index=['seed', 'method'], 
        columns='metric', 
        values='value'
    ).reset_index()