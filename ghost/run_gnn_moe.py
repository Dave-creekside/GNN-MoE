#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_gnn_moe.py

Main executable script for Ghost MoE model training and experimentation.
"""

import torch
import os
import argparse
import numpy as np
import random
import json

# --- Conditional Print Function ---
_VERBOSE = True
def verbose_print(*args, **kwargs):
    if _VERBOSE:
        print(*args, **kwargs)

# Import from modules
from gnn_moe_config import GhostMoEConfig
from gnn_moe_architecture import GhostMoEModel, create_dynamic_optimizer, PrimaryGhostLRScheduler
from gnn_moe_data import load_data # Assuming gnn_moe_data.py will be created
from gnn_moe_training import load_checkpoint, train_ghost_moe_model

def setup_environment(config: GhostMoEConfig):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    return device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ghost MoE Training Script")
    
    temp_cfg = GhostMoEConfig()

    # Add arguments for all fields in GhostMoEConfig
    for field in fields(GhostMoEConfig):
        if field.type == bool:
            parser.add_argument(f'--{field.name}', action='store_true')
            parser.add_argument(f'--no_{field.name}', action='store_false', dest=field.name)
        else:
            parser.add_argument(f'--{field.name}', type=field.type, default=getattr(temp_cfg, field.name))

    args = parser.parse_args()
    cfg = GhostMoEConfig(**vars(args))

    if args.quiet:
        _VERBOSE = False

    if cfg.run_name:
        cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.run_name)
    
    if not os.path.exists(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir)

    selected_device = setup_environment(cfg)
    train_loader, eval_loader, tokenizer, data_mode = load_data(cfg)
    
    model = GhostMoEModel(cfg).to(selected_device)
    
    optimizer = create_dynamic_optimizer(model, cfg)
    scheduler = PrimaryGhostLRScheduler(cfg, optimizer)

    start_epoch, current_step, best_eval_loss_resumed = 0, 0, float('inf')

    if cfg.resume_checkpoint:
        if os.path.isfile(cfg.resume_checkpoint):
            start_epoch, current_step, best_eval_loss_resumed = load_checkpoint(cfg.resume_checkpoint, model, optimizer, scheduler.primary_scheduler)
            model.to(selected_device)

    training_stats, final_best_loss = train_ghost_moe_model(
        model, train_loader, eval_loader, selected_device, cfg, 
        resume_from_epoch=start_epoch,
        resume_step=current_step,
        initial_best_loss=best_eval_loss_resumed
    )
    
    summary_data = {
        "run_name": cfg.run_name,
        "best_eval_loss": float(f"{final_best_loss:.4f}") if final_best_loss != float('inf') else None,
    }

    summary_file_path = os.path.join(cfg.checkpoint_dir, "run_summary.json")
    with open(summary_file_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
