import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def find_log_files(root_dir):
    """Find all training_log.json files in a directory."""
    log_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "training_log.json":
                log_files.append(os.path.join(dirpath, filename))
    return log_files

def load_log_data(log_files):
    """Load data from a list of log files into a pandas DataFrame."""
    all_data = []
    for log_file in log_files:
        run_name = os.path.basename(os.path.dirname(log_file))
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            entry['run_name'] = run_name
            all_data.append(entry)
            
    return pd.DataFrame(all_data)

def plot_individual_run(df, run_name, output_dir):
    """Generate and save plots for a single run."""
    run_df = df[df['run_name'] == run_name]
    if run_df.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(f'Training Analysis for {run_name}', fontsize=16)

    # Plot losses
    axes[0].plot(run_df['step'], run_df['train_loss'], label='Train Loss')
    axes[0].plot(run_df['step'], run_df['eval_loss'], label='Eval Loss', marker='o')
    axes[0].set_title('Losses vs. Steps')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Ghost Activations
    activations = pd.DataFrame(run_df['ghost_activations'].tolist(), index=run_df['step'])
    activations.columns = [f'ghost_{i}' for i in range(activations.shape[1])]
    activations.plot(ax=axes[1])
    axes[1].set_title('Ghost Expert Activations vs. Steps')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Activation Level')
    axes[1].grid(True)

    # Plot Saturation Metrics
    axes[2].plot(run_df['step'], run_df['saturation_level'], label='Saturation Level', color='r')
    ax2_twin = axes[2].twinx()
    ax2_twin.plot(run_df['step'], run_df['orthogonality_score'], label='Orthogonality Score', color='b')
    axes[2].set_title('Saturation and Orthogonality vs. Steps')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Saturation', color='r')
    ax2_twin.set_ylabel('Orthogonality', color='b')
    axes[2].legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(output_dir, f'{run_name}_analysis.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved individual plot to {plot_path}")

def plot_comparison(df, sweep_dir):
    """Generate and save comparison plots for all runs."""
    if df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle('Sweep Comparison', fontsize=16)

    # Plot Eval Loss Comparison
    sns.lineplot(data=df, x='step', y='eval_loss', hue='run_name', ax=axes[0], marker='o')
    axes[0].set_title('Evaluation Loss Comparison')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Evaluation Loss')
    axes[0].grid(True)
    axes[0].legend(title='Run')

    # Plot Average Ghost Activation Comparison
    df['avg_ghost_activation'] = df['ghost_activations'].apply(lambda x: sum(x) / len(x) if x else 0)
    sns.lineplot(data=df, x='step', y='avg_ghost_activation', hue='run_name', ax=axes[1])
    axes[1].set_title('Average Ghost Activation Comparison')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Average Activation')
    axes[1].grid(True)
    axes[1].legend(title='Run')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(sweep_dir, 'sweep_comparison.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved comparison plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results.")
    parser.add_argument('sweep_dir', type=str, help="The root directory of the sweep to analyze.")
    args = parser.parse_args()

    print(f"Analyzing sweep results in: {args.sweep_dir}")
    
    log_files = find_log_files(args.sweep_dir)
    if not log_files:
        print("No training_log.json files found. Exiting.")
        return

    df = load_log_data(log_files)
    
    # Generate individual plots for each run
    for run_name in df['run_name'].unique():
        plot_individual_run(df, run_name, os.path.join(args.sweep_dir, run_name))

    # Generate comparison plot
    plot_comparison(df, args.sweep_dir)

    print("Analysis complete.")

if __name__ == "__main__":
    main()
