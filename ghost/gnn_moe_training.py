#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnn_moe_training.py

Training loop, evaluation, and checkpointing utilities for Ghost MoE models.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import shutil
from tqdm import tqdm
from collections import defaultdict
import time

# Assuming GhostMoEConfig and model classes will be imported in the main script
from .gnn_moe_config import GhostMoEConfig
from .gnn_moe_architecture import GhostMoEModel, create_dynamic_optimizer, PrimaryGhostLRScheduler

# --- Checkpoint Helper Functions (can be shared or copied) ---
def save_checkpoint(state, is_best, checkpoint_dir="checkpoints", filename="checkpoint.pth.tar"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth.tar")
        shutil.copyfile(filepath, best_model_path)
        if 'config' in state:
            config_dict = state['config']
            config_json_path = os.path.join(checkpoint_dir, "config.json")
            try:
                import json
                with open(config_json_path, 'w') as f:
                    json.dump(config_dict, f, indent=4)
            except Exception as e:
                print(f"Error saving config.json: {e}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        return 0, 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
    step = checkpoint.get('step', 0)
    
    return start_epoch, step, best_eval_loss

# --- Training Utilities ---
def prepare_batch(batch, device):
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    labels = input_ids.clone()
    labels[~attention_mask.bool()] = 0 
    return input_ids, attention_mask, labels

def evaluate_model(model, eval_loader, device, config, max_batches=-1):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_eval_batches = len(eval_loader) if max_batches == -1 else min(len(eval_loader), max_batches)

    if num_eval_batches == 0:
        return float('inf'), float('inf')

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= num_eval_batches:
                break
            input_ids, attention_mask, labels = prepare_batch(batch, device)
            outputs = model(input_ids, step=0, attention_mask=attention_mask, labels=labels)
            
            mask = (labels != 0)
            if mask.sum().item() > 0:
                total_loss += outputs['loss'].item() * mask.sum().item()
                total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity

def update_parameter_group_lrs(optimizer, primary_lr, ghost_lrs):
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'primary_experts':
            param_group['lr'] = primary_lr
        elif param_group['name'] == 'ghost_experts':
            # This assumes one LR for all ghosts, need to adjust if per-ghost LR is needed
            param_group['lr'] = ghost_lrs[0] if ghost_lrs else 0

def log_ghost_training_metrics(step, total_loss, base_loss, orthogonality_loss, ghost_activations, primary_lr, ghost_lrs, saturation_metrics):
    print(f"Step {step}: Total Loss: {total_loss.item():.4f}, Base Loss: {base_loss.item():.4f}, Ortho Loss: {orthogonality_loss.item():.4f}")
    print(f"  LRs - Primary: {primary_lr:.2e}, Ghosts: {[f'{lr:.2e}' for lr in ghost_lrs]}")
    print(f"  Ghost Activations: {[f'{act:.2f}' for act in ghost_activations]}")
    if saturation_metrics:
        print(f"  Saturation: {saturation_metrics.get('saturation_level', 0):.4f}, Ortho Score: {saturation_metrics.get('orthogonality_score', 0):.4f}")


def train_ghost_moe_model(model, train_loader, eval_loader, device, config,
                          resume_from_epoch=0, resume_step=0, initial_best_loss=float('inf')):
    
    optimizer = create_dynamic_optimizer(model, config)
    
    actual_batches_per_epoch = len(train_loader) if config.max_batches_per_epoch == -1 else min(len(train_loader), config.max_batches_per_epoch)
    if not hasattr(config, 'max_steps') or config.max_steps is None:
        config.max_steps = config.epochs * actual_batches_per_epoch
    
    lr_scheduler = PrimaryGhostLRScheduler(config, optimizer)

    total_steps = config.max_steps
    
    if total_steps == 0:
        return [], float('inf')

    stats = [] # Change to a list of dicts
    best_eval_loss = initial_best_loss
    current_step = resume_step
    start_time = time.time()

    for epoch in range(resume_from_epoch, config.epochs):
        model.train()
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", total=actual_batches_per_epoch)

        for batch_idx, batch in enumerate(pbar_train):
            if batch_idx >= actual_batches_per_epoch:
                break

            input_ids, attention_mask, labels = prepare_batch(batch, device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, step=current_step, attention_mask=attention_mask, return_loss=True, labels=labels)
            
            base_loss = outputs['loss']
            orthogonality_loss = model.get_total_orthogonality_loss(current_step)
            ghost_saturation_loss = model.get_ghost_saturation_loss()
            
            total_loss = base_loss + orthogonality_loss + ghost_saturation_loss
            
            total_loss.backward()
            
            ghost_activations = model.get_current_ghost_activations()
            primary_lr, ghost_lrs = lr_scheduler.step(ghost_activations)
            
            update_parameter_group_lrs(optimizer, primary_lr, ghost_lrs)
            
            optimizer.step()
            
            current_step += 1
            
            pbar_train.set_postfix({
                'loss': f'{total_loss.item():.3f}',
                'lm': f'{base_loss.item():.3f}',
                'lr': f"{primary_lr:.1e}"
            })

            if current_step % config.eval_every == 0:
                eval_loss, perplexity = evaluate_model(model, eval_loader, device, config)
                
                saturation_metrics = model.get_last_saturation_metrics()
                
                log_ghost_training_metrics(current_step, total_loss, base_loss, orthogonality_loss, ghost_activations, primary_lr, ghost_lrs, saturation_metrics)
                print(f"  Evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

                # Append a snapshot of all metrics
                stats.append({
                    'step': current_step,
                    'train_loss': total_loss.item(),
                    'lm_loss': base_loss.item(),
                    'orthogonality_loss': orthogonality_loss.item(),
                    'eval_loss': eval_loss,
                    'eval_perplexity': perplexity,
                    'primary_lr': primary_lr,
                    'ghost_lrs': ghost_lrs,
                    'ghost_activations': ghost_activations.cpu().numpy().tolist(),
                    'saturation_level': saturation_metrics.get('saturation_level', 0),
                    'orthogonality_score': saturation_metrics.get('orthogonality_score', 0)
                })
                
                is_best = eval_loss < best_eval_loss
                if is_best:
                    best_eval_loss = eval_loss
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.primary_scheduler.state_dict(),
                    'best_eval_loss': best_eval_loss,
                    'config': vars(config)
                }, is_best, checkpoint_dir=config.checkpoint_dir)
                model.train()

    return stats, best_eval_loss
