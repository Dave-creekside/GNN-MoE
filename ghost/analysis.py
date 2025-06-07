import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_log_data(log_path):
    """Load data from a single training_log.json file."""
    with open(log_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_losses(df, output_path):
    """Plot training and evaluation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['train_loss'], label='Train Loss (Batch)', alpha=0.7)
    plt.plot(df['step'], df['eval_loss'], label='Eval Loss', marker='o', linestyle='--')
    plt.title('Loss vs. Training Steps')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_perplexity(df, output_path):
    """Plot evaluation perplexity."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['eval_perplexity'], label='Perplexity', marker='o', color='green')
    plt.title('Perplexity vs. Training Steps')
    plt.xlabel('Step')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_learning_rates(df, output_path):
    """Plot primary and ghost learning rates."""
    plt.figure(figsize=(10, 6))
    
    # Plot primary LR
    plt.plot(df['step'], df['primary_lr'], label='Primary LR', color='blue')

    # Plot ghost LRs
    ghost_lrs_df = pd.DataFrame(df['ghost_lrs'].tolist(), index=df['step'])
    for i, col in enumerate(ghost_lrs_df.columns):
        plt.plot(ghost_lrs_df.index, ghost_lrs_df[col], label=f'Ghost {i} LR', linestyle=':', alpha=0.8)

    plt.title('Learning Rates vs. Training Steps')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_ghost_metrics(df, output_path):
    """Plot ghost activations and saturation level."""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot saturation level on the first y-axis
    ax1.plot(df['step'], df['saturation_level'], label='Saturation Level', color='red', linestyle='--')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Saturation Level', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)

    # Create a second y-axis for activations
    ax2 = ax1.twinx()
    activations_df = pd.DataFrame(df['ghost_activations'].tolist(), index=df['step'])
    for i, col in enumerate(activations_df.columns):
        ax2.plot(activations_df.index, activations_df[col], label=f'Ghost {i} Activation', alpha=0.7)
    
    ax2.set_ylabel('Activation Level')
    ax2.legend(loc='upper right')
    
    plt.title('Ghost Activations and Saturation vs. Steps')
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def run_analysis(log_path):
    """
    Main function to load a log and generate all plots.
    """
    if not os.path.exists(log_path):
        print(f"❌ ERROR: Log file not found at {log_path}")
        return

    output_dir = os.path.dirname(log_path)
    print(f"📊 Analyzing log file: {log_path}")
    print(f"   Saving plots to: {output_dir}")

    df = load_log_data(log_path)
    if df.empty:
        print("   ⚠️ Log file is empty. Skipping analysis.")
        return

    plot_losses(df, os.path.join(output_dir, "plot_losses.png"))
    plot_perplexity(df, os.path.join(output_dir, "plot_perplexity.png"))
    plot_learning_rates(df, os.path.join(output_dir, "plot_learning_rates.png"))
    plot_ghost_metrics(df, os.path.join(output_dir, "plot_ghost_metrics.png"))
    
    print("   ✅ Analysis and plotting complete.")
