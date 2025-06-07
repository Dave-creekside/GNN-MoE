import os
import subprocess
import itertools
import shutil
from datetime import datetime

# --- Sweep Configuration ---

# Define the base command for running a single training instance
# We use `python -m ghost.run_gnn_moe` to ensure Python treats `ghost` as a package
BASE_COMMAND = "python -m ghost.run_gnn_moe"

# Define hyperparameters to sweep over
# Format: { 'arg_name': [list_of_values], 'prefix': 'short_name' }
SWEEP_PARAMS = {
    'ghost_activation_threshold': {
        'values': [0.6, 0.75, 0.9],
        'prefix': 'gat'
    },
    'ghost_learning_rate': {
        'values': [1e-4, 5e-5],
        'prefix': 'glr'
    }
}

# Define static parameters that will be the same for all runs in the sweep
# These are chosen to make the runs fast for testing purposes
STATIC_PARAMS = {
    'epochs': 1,
    'max_batches_per_epoch': 50, # Limit to 50 batches
    'num_train_samples': 1000,
    'num_eval_samples': 200,
    'eval_every': 25, # Evaluate twice per run
    'quiet': True # Suppress verbose output from the training script
}

# --- Sweep Execution ---

def run_sweep():
    """
    Executes the hyperparameter sweep.
    """
    # Create a root directory for this entire sweep session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root_dir = os.path.join('ghost', 'tests', 'sweeps', f'sweep_{timestamp}')
    os.makedirs(sweep_root_dir, exist_ok=True)
    print(f"📁 Created root directory for this sweep: {sweep_root_dir}")

    # Get all parameter names and their value lists
    param_names = list(SWEEP_PARAMS.keys())
    param_values = [SWEEP_PARAMS[name]['values'] for name in param_names]

    # Generate all combinations of hyperparameters
    combinations = list(itertools.product(*param_values))
    print(f"🔬 Starting sweep with {len(combinations)} combinations...")

    for i, combo in enumerate(combinations):
        run_name_parts = []
        
        # --- Build the command for this specific run ---
        cmd = list(BASE_COMMAND.split())
        
        # Add sweep parameters
        for j, value in enumerate(combo):
            param_name = param_names[j]
            prefix = SWEEP_PARAMS[param_name]['prefix']
            cmd.append(f"--{param_name}")
            cmd.append(str(value))
            run_name_parts.append(f"{prefix}_{value:.2e}" if isinstance(value, float) else f"{prefix}_{value}")

        # Add static parameters
        for name, value in STATIC_PARAMS.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{name}")
            else:
                cmd.append(f"--{name}")
                cmd.append(str(value))
            
        # Create a unique run name and output directory
        run_name = "_".join(run_name_parts)
        run_output_dir = os.path.join(sweep_root_dir, run_name)
        os.makedirs(run_output_dir, exist_ok=True)
        
        # The training script will save checkpoints to a subdir of 'checkpoints'
        # We need to tell it where to put them.
        checkpoint_base_dir = "checkpoints"
        cmd.append("--checkpoint_dir")
        cmd.append(checkpoint_base_dir)
        cmd.append("--run_name")
        cmd.append(run_name)

        print(f"\n--- Running Combination {i+1}/{len(combinations)}: {run_name} ---")
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            # Execute the training run
            subprocess.run(cmd, check=True)
            
            # --- Post-processing: Move results ---
            source_dir = os.path.join(checkpoint_base_dir, run_name)
            
            # Move summary file
            summary_src = os.path.join(source_dir, "run_summary.json")
            summary_dst = os.path.join(run_output_dir, "run_summary.json")
            if os.path.exists(summary_src):
                shutil.move(summary_src, summary_dst)
                print(f"   ✅ Moved summary to {summary_dst}")

            # Move plots (if any were created)
            # Note: The base run script doesn't create plots, but this is here for future use
            plots_src_dir = "plots"
            if os.path.exists(plots_src_dir):
                for plot_file in os.listdir(plots_src_dir):
                    if run_name in plot_file:
                        shutil.move(os.path.join(plots_src_dir, plot_file), run_output_dir)
                print(f"   ✅ Moved any relevant plots to {run_output_dir}")

            # Clean up the checkpoint directory
            if os.path.exists(source_dir):
                shutil.rmtree(source_dir)
                print(f"   ✅ Cleaned up checkpoint directory: {source_dir}")

        except subprocess.CalledProcessError as e:
            print(f"   ❌ ERROR: Run failed for combination {run_name}")
            print(f"   Return code: {e.returncode}")
            print(f"   Output:\n{e.stdout}\n{e.stderr}")
        except Exception as e:
            print(f"   ❌ An unexpected error occurred: {e}")

    print("\n🎉 Sweep finished!")

if __name__ == "__main__":
    run_sweep()
