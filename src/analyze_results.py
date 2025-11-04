"""
Analyze and visualize results from automated experiments.
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

def load_results(summary_path=None):
    """
    Load experiment summary.
    If summary_path is None, finds the most recent run directory.
    """
    if summary_path is None:
        # Find most recent run directory
        run_dirs = glob.glob("output/run_*")
        if not run_dirs:
            raise FileNotFoundError("No run directories found in output/")
        latest_run = max(run_dirs, key=os.path.getmtime)
        summary_path = os.path.join(latest_run, "experiment_summary.pkl")
        print(f"Loading results from: {summary_path}")
    
    with open(summary_path, 'rb') as f:
        results = pickle.load(f)
    return results, summary_path


def analyze_results(results, summary_path=None):
    """
    Analyze experiment results and create visualizations.
    
    Args:
        results: List of experiment results
        summary_path: Path to the summary file (for saving outputs in same directory)
    """
    # Filter out failed experiments
    successful_results = [r for r in results if 'error' not in r]
    
    if len(successful_results) == 0:
        print("No successful experiments to analyze!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(successful_results)
    
    # Extract epoch-specific test accuracies if available
    checkpoint_epochs = [120, 200, 240]
    for epoch in checkpoint_epochs:
        df[f'test_acc_epoch_{epoch}'] = df['test_acc_at_epochs'].apply(
            lambda x: x.get(epoch) if isinstance(x, dict) else None
        )
    
    # Add extended validation columns if available
    df['best_extended_val_acc'] = df.apply(
        lambda x: x.get('best_extended_val_acc', np.nan), axis=1
    )
    df['test_acc_best_ext_val'] = df.apply(
        lambda x: x.get('test_acc_best_ext_val', np.nan), axis=1
    )
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    # Group by optimizer and adaptive pooling
    for optimizer in df['optimizer'].unique():
        print(f"\n{'='*80}")
        print(f"OPTIMIZER: {optimizer}")
        print(f"{'='*80}")
        for use_adap in [False, True]:
            subset = df[(df['use_adap_pool'] == use_adap) & (df['optimizer'] == optimizer)]
            if len(subset) == 0:
                continue
            print(f"\n{'WITH' if use_adap else 'WITHOUT'} Adaptive Pooling:")
            print(f"  Test Accuracy (best loss):  {subset['test_acc'].mean():.4f} ± {subset['test_acc'].std():.4f}")
            if 'test_acc_best_ext_val' in subset.columns and subset['test_acc_best_ext_val'].notna().any():
                print(f"  Test Accuracy (best ext val): {subset['test_acc_best_ext_val'].mean():.4f} ± {subset['test_acc_best_ext_val'].std():.4f}")
            print(f"  Val Accuracy (quick):        {subset['best_val_acc'].mean():.4f} ± {subset['best_val_acc'].std():.4f}")
            if 'best_extended_val_acc' in subset.columns and subset['best_extended_val_acc'].notna().any():
                print(f"  Extended Val Accuracy:       {subset['best_extended_val_acc'].mean():.4f} ± {subset['best_extended_val_acc'].std():.4f}")
            print(f"  Avg Time:                    {subset['time_minutes'].mean():.1f} ± {subset['time_minutes'].std():.1f} minutes")
            print(f"  Avg Epochs:                  {subset['training_epochs'].mean():.1f}")
    
    # Statistical comparison per optimizer
    for optimizer in df['optimizer'].unique():
        subset = df[df['optimizer'] == optimizer]
        no_adap = subset[subset['use_adap_pool'] == False]['test_acc'].values
        with_adap = subset[subset['use_adap_pool'] == True]['test_acc'].values
        
        if len(no_adap) > 0 and len(with_adap) > 0:
            print(f"\n\nTest Accuracy Improvement for {optimizer} (Adaptive vs Non-Adaptive):")
            print(f"  Mean difference: {with_adap.mean() - no_adap.mean():.4f}")
            print(f"  Relative improvement: {((with_adap.mean() - no_adap.mean()) / no_adap.mean() * 100):.2f}%")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    # Get unique optimizers
    optimizers = df['optimizer'].unique()
    n_optimizers = len(optimizers)
    
    # Determine if we have extended validation data
    has_ext_val = 'test_acc_best_ext_val' in df.columns and df['test_acc_best_ext_val'].notna().any()
    
    # Create figure with 3 or 4 rows depending on extended validation availability
    n_rows = 4 if has_ext_val else 3
    fig, axes = plt.subplots(n_rows, n_optimizers, figsize=(7*n_optimizers, 4*n_rows))
    if n_optimizers == 1:
        axes = axes.reshape(-1, 1)
    
    for opt_idx, optimizer in enumerate(optimizers):
        subset = df[df['optimizer'] == optimizer]
        
        # Check what configurations we have
        has_no_adap = len(subset[subset['use_adap_pool'] == False]) > 0
        has_with_adap = len(subset[subset['use_adap_pool'] == True]) > 0
        
        # 1. Test Accuracy Comparison
        ax = axes[0, opt_idx]
        
        if has_no_adap and has_with_adap:
            # Both configurations available - side-by-side bars
            seeds = subset[subset['use_adap_pool'] == False]['seed'].values
            no_adap_acc = subset[subset['use_adap_pool'] == False]['test_acc'].values
            with_adap_acc = subset[subset['use_adap_pool'] == True]['test_acc'].values
            
            x = np.arange(len(seeds))
            width = 0.35
            
            ax.bar(x - width/2, no_adap_acc, width, label='No Adaptive', color='steelblue', alpha=0.8)
            ax.bar(x + width/2, with_adap_acc, width, label='With Adaptive', color='coral', alpha=0.8)
            
            ax.set_xticks(x)
            ax.set_xticklabels(seeds)
        elif has_no_adap:
            # Only non-adaptive available
            seeds = subset['seed'].values
            test_accs = subset['test_acc'].values
            x = np.arange(len(seeds))
            ax.bar(x, test_accs, 0.6, label='No Adaptive', color='steelblue', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(seeds)
        else:
            # Only adaptive available
            seeds = subset['seed'].values
            test_accs = subset['test_acc'].values
            x = np.arange(len(seeds))
            ax.bar(x, test_accs, 0.6, label='With Adaptive', color='coral', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(seeds)
        
        ax.set_xlabel('Seed', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'{optimizer}: Test Accuracy by Seed', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
        
        # 2. Training Time Comparison
        ax = axes[1, opt_idx]
        
        if has_no_adap and has_with_adap:
            # Both configurations available
            no_adap_time = subset[subset['use_adap_pool'] == False]['time_minutes'].values
            with_adap_time = subset[subset['use_adap_pool'] == True]['time_minutes'].values
            
            x = np.arange(len(seeds))
            width = 0.35
            
            ax.bar(x - width/2, no_adap_time, width, label='No Adaptive', color='steelblue', alpha=0.8)
            ax.bar(x + width/2, with_adap_time, width, label='With Adaptive', color='coral', alpha=0.8)
        elif has_no_adap:
            # Only non-adaptive available
            train_times = subset['time_minutes'].values
            x = np.arange(len(seeds))
            ax.bar(x, train_times, 0.6, label='No Adaptive', color='steelblue', alpha=0.8)
        else:
            # Only adaptive available
            train_times = subset['time_minutes'].values
            x = np.arange(len(seeds))
            ax.bar(x, train_times, 0.6, label='With Adaptive', color='coral', alpha=0.8)
        
        ax.set_xlabel('Seed', fontsize=12)
        ax.set_ylabel('Training Time (minutes)', fontsize=12)
        ax.set_title(f'{optimizer}: Training Time by Seed', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seeds)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Extended Validation Comparison (if available)
        row_idx = 2
        if has_ext_val:
            ax = axes[row_idx, opt_idx]
            
            if has_no_adap and has_with_adap:
                # Both configurations available
                no_adap_ext_val = subset[subset['use_adap_pool'] == False]['test_acc_best_ext_val'].values
                with_adap_ext_val = subset[subset['use_adap_pool'] == True]['test_acc_best_ext_val'].values
                no_adap_best_loss = subset[subset['use_adap_pool'] == False]['test_acc'].values
                with_adap_best_loss = subset[subset['use_adap_pool'] == True]['test_acc'].values
                
                x = np.arange(len(seeds))
                width = 0.2
                
                ax.bar(x - 1.5*width, no_adap_best_loss, width, label='No Adap (Best Loss)', 
                       color='steelblue', alpha=0.8)
                ax.bar(x - 0.5*width, no_adap_ext_val, width, label='No Adap (Best Ext Val)', 
                       color='lightblue', alpha=0.8)
                ax.bar(x + 0.5*width, with_adap_best_loss, width, label='Adap (Best Loss)', 
                       color='coral', alpha=0.8)
                ax.bar(x + 1.5*width, with_adap_ext_val, width, label='Adap (Best Ext Val)', 
                       color='lightsalmon', alpha=0.8)
            elif has_no_adap:
                # Only non-adaptive available
                ext_val = subset['test_acc_best_ext_val'].values
                best_loss = subset['test_acc'].values
                
                x = np.arange(len(seeds))
                width = 0.35
                
                ax.bar(x - width/2, best_loss, width, label='Best Loss', color='steelblue', alpha=0.8)
                ax.bar(x + width/2, ext_val, width, label='Best Ext Val', color='lightblue', alpha=0.8)
            else:
                # Only adaptive available
                ext_val = subset['test_acc_best_ext_val'].values
                best_loss = subset['test_acc'].values
                
                x = np.arange(len(seeds))
                width = 0.35
                
                ax.bar(x - width/2, best_loss, width, label='Best Loss', color='coral', alpha=0.8)
                ax.bar(x + width/2, ext_val, width, label='Best Ext Val', color='lightsalmon', alpha=0.8)
            
            ax.set_xlabel('Seed', fontsize=12)
            ax.set_ylabel('Test Accuracy', fontsize=12)
            ax.set_title(f'{optimizer}: Best Loss vs Extended Val Model', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(seeds)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1.0])
            
            row_idx = 3
        
        # 4 (or 3). Test Accuracy at Different Epochs
        ax = axes[row_idx, opt_idx]
        
        # Prepare data for line plot
        epoch_labels = ['Epoch 120', 'Epoch 200', 'Epoch 240', 'Best Loss']
        
        if has_no_adap and has_with_adap:
            # Both configurations available
            for idx, seed in enumerate(seeds):
                seed_no_adap = subset[(subset['seed'] == seed) & (subset['use_adap_pool'] == False)].iloc[0]
                seed_with_adap = subset[(subset['seed'] == seed) & (subset['use_adap_pool'] == True)].iloc[0]
                
                # Collect test accuracies at different epochs
                no_adap_values = []
                with_adap_values = []
                
                for epoch in checkpoint_epochs:
                    col_name = f'test_acc_epoch_{epoch}'
                    no_adap_val = seed_no_adap[col_name] if pd.notna(seed_no_adap[col_name]) else None
                    with_adap_val = seed_with_adap[col_name] if pd.notna(seed_with_adap[col_name]) else None
                    no_adap_values.append(no_adap_val)
                    with_adap_values.append(with_adap_val)
                
                # Add best loss model accuracy
                no_adap_values.append(seed_no_adap['test_acc'])
                with_adap_values.append(seed_with_adap['test_acc'])
                
                # Plot lines for this seed
                x_pos = np.arange(len(epoch_labels))
                ax.plot(x_pos, no_adap_values, 'o-', label=f'Seed {seed} (No Adap)', alpha=0.6, linewidth=2)
                ax.plot(x_pos, with_adap_values, 's--', label=f'Seed {seed} (Adap)', alpha=0.6, linewidth=2)
        else:
            # Only one configuration available
            for idx, seed in enumerate(seeds):
                seed_data = subset[subset['seed'] == seed].iloc[0]
                
                # Collect test accuracies at different epochs
                values = []
                for epoch in checkpoint_epochs:
                    col_name = f'test_acc_epoch_{epoch}'
                    val = seed_data[col_name] if pd.notna(seed_data[col_name]) else None
                    values.append(val)
                
                # Add best loss model accuracy
                values.append(seed_data['test_acc'])
                
                # Plot lines for this seed
                x_pos = np.arange(len(epoch_labels))
                config_label = 'Adap' if has_with_adap else 'No Adap'
                ax.plot(x_pos, values, 'o-', label=f'Seed {seed} ({config_label})', alpha=0.6, linewidth=2)
        
        ax.set_xlabel('Checkpoint', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'{optimizer}: Test Accuracy Over Training', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(epoch_labels, rotation=15, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    # Save figure to the same directory as the summary file
    output_dir = os.path.dirname(summary_path) if isinstance(summary_path, str) else "output"
    output_path = os.path.join(output_dir, "experiment_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n\nVisualization saved to: {output_path}")
    
    plt.show()
    
    return df


if __name__ == "__main__":
    print("Loading experiment results...")
    results, summary_path = load_results()
    df = analyze_results(results, summary_path)
    
    # Export to CSV in the same directory as the summary
    output_dir = os.path.dirname(summary_path)
    csv_path = os.path.join(output_dir, "experiment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results exported to: {csv_path}")
