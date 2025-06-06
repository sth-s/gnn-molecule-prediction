#!/usr/bin/env python3
"""
Grid Search Wrapper for GIN Hyperparameter Optimization on Tox21 Dataset.

This script performs grid search over specified hyperparameters using the train.py script.
Results are logged to JSON format for easy analysis.

Usage:
    python scripts/hyp_search.py
    python scripts/hyp_search.py --output_dir grid_search_results
    python scripts/hyp_search.py --fast_search --max_experiments 10
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Any


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Grid search for GIN hyperparameter optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--output_dir', type=str, default='grid_search_results',
                        help='Directory to save grid search results')
    parser.add_argument('--fast_search', action='store_true',
                        help='Use fast search mode (fewer epochs, early stopping)')
    parser.add_argument('--max_experiments', type=int, default=None,
                        help='Maximum number of experiments to run (for testing)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--data_root', type=str, default='data/Tox21',
                        help='Root directory for dataset')
    parser.add_argument('--split_method', type=str, default='random',
                        choices=['random', 'scaffold', 'index'],
                        help='Method for splitting the dataset')
    
    return parser.parse_args()


def get_hyperparameter_grid() -> Dict[str, List]:
    """Define the hyperparameter grid to search over."""
    return {
        'hidden_channels': [32, 128],
        'num_layers': [3, 4],
        'dropout': [0.2, 0.5],
        'batch_size': [32, 64],
        'lr': [1e-2, 5e-3, 1e-3]
    }


def generate_experiment_configs(grid: Dict[str, List], max_experiments: int = None) -> List[Dict]:
    """Generate all combinations of hyperparameters."""
    # Get all parameter names and their possible values
    param_names = list(grid.keys())
    param_values = list(grid.values())
    
    # Generate all combinations
    all_combinations = list(product(*param_values))
    
    # Limit experiments if specified
    if max_experiments is not None:
        all_combinations = all_combinations[:max_experiments]
    
    # Convert to list of dictionaries
    configs = []
    for i, combination in enumerate(all_combinations):
        config = {param_names[j]: combination[j] for j in range(len(param_names))}
        config['experiment_id'] = i + 1
        configs.append(config)
    
    return configs


def run_single_experiment(config: Dict, base_args: argparse.Namespace, experiment_dir: Path) -> Dict:
    """Run a single experiment with given hyperparameters."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {config['experiment_id']}")
    print(f"{'='*60}")
    print("Configuration:")
    for key, value in config.items():
        if key != 'experiment_id':
            print(f"  {key}: {value}")
    
    # Create experiment directory
    exp_name = f"exp_{config['experiment_id']:03d}"
    exp_dir = experiment_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command arguments
    cmd = [
        sys.executable, 'scripts/train.py',
        '--hidden_channels', str(config['hidden_channels']),
        '--num_layers', str(config['num_layers']),
        '--dropout', str(config['dropout']),
        '--batch_size', str(config['batch_size']),
        '--lr', str(config['lr']),
        '--output_dir', str(exp_dir),
        '--exp_name', 'model',
        '--seed', str(base_args.seed),
        '--device', base_args.device,
        '--data_root', base_args.data_root,
        '--split_method', base_args.split_method,
        '--save_best_only',  # Only save best model to save space
    ]
    
    # Add fast search flag if enabled
    if base_args.fast_search:
        cmd.append('--fast_search')
    
    print(f"\nRunning command: {' '.join(cmd)}")
    
    # Run the experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        success = True
        error_message = None
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.CalledProcessError as e:
        success = False
        error_message = str(e)
        stdout = e.stdout
        stderr = e.stderr
        print(f"ERROR: Experiment failed with code {e.returncode}")
        print(f"Error message: {error_message}")
    
    end_time = time.time()
    experiment_time = end_time - start_time
    
    # Load results if successful
    results = None
    if success:
        results_file = exp_dir / 'model' / 'results.json'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                print(f"SUCCESS: Experiment completed in {experiment_time:.1f}s")
                if 'final_results' in results and 'best_validation' in results['final_results']:
                    val_roc = results['final_results']['best_validation']['roc_auc']
                    print(f"Best validation ROC AUC: {val_roc:.4f}")
            except Exception as e:
                print(f"WARNING: Could not load results file: {e}")
        else:
            print(f"WARNING: Results file not found: {results_file}")
    
    # Create experiment summary
    experiment_summary = {
        'experiment_id': config['experiment_id'],
        'hyperparameters': {k: v for k, v in config.items() if k != 'experiment_id'},
        'success': success,
        'error_message': error_message,
        'experiment_time_seconds': experiment_time,
        'experiment_time_minutes': experiment_time / 60,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    # Save experiment summary
    summary_file = exp_dir / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    return experiment_summary


def save_grid_search_results(all_results: List[Dict], output_dir: Path, args: argparse.Namespace):
    """Save comprehensive grid search results."""
    
    # Calculate summary statistics
    successful_experiments = [r for r in all_results if r['success']]
    failed_experiments = [r for r in all_results if not r['success']]
    
    # Extract best results
    best_results = []
    for result in successful_experiments:
        if result['results'] and 'final_results' in result['results']:
            final_results = result['results']['final_results']
            if 'best_validation' in final_results:
                val_roc = final_results['best_validation']['roc_auc']
                test_roc = final_results.get('test', {}).get('roc_auc', None)
                
                best_results.append({
                    'experiment_id': result['experiment_id'],
                    'hyperparameters': result['hyperparameters'],
                    'val_roc_auc': val_roc,
                    'test_roc_auc': test_roc,
                    'experiment_time_minutes': result['experiment_time_minutes'],
                    'best_epoch': final_results['best_validation'].get('epoch', None),
                    'total_epochs': result['results']['training_info'].get('total_epochs', None),
                    'early_stopped': result['results']['training_info'].get('early_stopped', False)
                })
    
    # Sort by validation ROC AUC
    best_results.sort(key=lambda x: x['val_roc_auc'], reverse=True)
    
    # Calculate total time
    total_time = sum(r['experiment_time_seconds'] for r in all_results)
    
    # Create comprehensive summary
    summary = {
        'grid_search_info': {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(all_results),
            'successful_experiments': len(successful_experiments),
            'failed_experiments': len(failed_experiments),
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'total_time_hours': total_time / 3600,
            'fast_search_mode': args.fast_search,
            'hyperparameter_grid': get_hyperparameter_grid(),
            'other_settings': {
                'seed': args.seed,
                'device': args.device,
                'data_root': args.data_root,
                'split_method': args.split_method
            }
        },
        'best_results': best_results[:10],  # Top 10 results
        'all_experiments': all_results,
        'summary_statistics': {
            'best_val_roc_auc': best_results[0]['val_roc_auc'] if best_results else None,
            'worst_val_roc_auc': best_results[-1]['val_roc_auc'] if best_results else None,
            'mean_val_roc_auc': sum(r['val_roc_auc'] for r in best_results) / len(best_results) if best_results else None,
            'mean_experiment_time_minutes': sum(r['experiment_time_minutes'] for r in all_results) / len(all_results),
            'mean_epochs_completed': sum(r.get('total_epochs', 0) for r in best_results) / len(best_results) if best_results else None
        }
    }
    
    # Save main results file
    results_file = output_dir / 'grid_search_results.json'
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETED")
    print(f"{'='*60}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Successful: {len(successful_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    print(f"Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
    
    if best_results:
        print(f"\nTOP 5 RESULTS:")
        print("-" * 40)
        for i, result in enumerate(best_results[:5]):
            print(f"{i+1}. Exp {result['experiment_id']:3d}: ROC AUC = {result['val_roc_auc']:.4f}")
            hyperparams = result['hyperparameters']
            print(f"   Hidden: {hyperparams['hidden_channels']}, Layers: {hyperparams['num_layers']}, "
                  f"Dropout: {hyperparams['dropout']}, Batch: {hyperparams['batch_size']}, LR: {hyperparams['lr']}")
            print(f"   Time: {result['experiment_time_minutes']:.1f}min, Epochs: {result['total_epochs']}")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return summary


def main():
    """Main grid search function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GNN MOLECULE PREDICTION - HYPERPARAMETER GRID SEARCH")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Fast search mode: {args.fast_search}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {args.device}")
    
    # Get hyperparameter grid and generate experiments
    grid = get_hyperparameter_grid()
    experiment_configs = generate_experiment_configs(grid, args.max_experiments)
    
    print(f"\nHyperparameter grid:")
    for param, values in grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = len(list(product(*grid.values())))
    actual_experiments = len(experiment_configs)
    
    print(f"\nTotal possible combinations: {total_combinations}")
    print(f"Experiments to run: {actual_experiments}")
    
    if args.max_experiments and actual_experiments < total_combinations:
        print(f"(Limited to {args.max_experiments} for testing)")
    
    # Create experiments directory
    experiments_dir = output_dir / 'experiments'
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all experiments
    all_results = []
    start_time = time.time()
    
    for i, config in enumerate(experiment_configs):
        print(f"\n[{i+1}/{len(experiment_configs)}] Starting experiment {config['experiment_id']}")
        result = run_single_experiment(config, args, experiments_dir)
        all_results.append(result)
        
        # Show progress
        elapsed_time = time.time() - start_time
        avg_time_per_exp = elapsed_time / (i + 1)
        remaining_experiments = len(experiment_configs) - (i + 1)
        estimated_remaining_time = avg_time_per_exp * remaining_experiments
        
        print(f"Progress: {i+1}/{len(experiment_configs)} "
              f"(Elapsed: {elapsed_time/60:.1f}min, "
              f"Est. remaining: {estimated_remaining_time/60:.1f}min)")
    
    # Save comprehensive results
    summary = save_grid_search_results(all_results, output_dir, args)
    
    return summary


if __name__ == "__main__":
    main()
