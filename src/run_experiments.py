"""
Automated experiment runner for 3x4 reset feedforward QCNN experiments.
Runs multiple seeds with and without adaptive pooling.

Logging:
- run_log.txt: Concise progress updates
- full_log.txt: Detailed execution logs
"""

import numpy as np
import os
from qiskit_aer import Aer, AerSimulator
import torch
import pickle

# Use non-interactive backend to avoid Tkinter threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_machine_learning.optimizers import SPSA, COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import sys
import time
from datetime import datetime

from utils import *
from prebuilt_cicuits import get_circuit
from callbacks import EarlyStoppingCallback, SPSACallbackWrapper
from param_init import init_params_aligned

from amp_encode.utils_amp import images_to_angles_batch_dual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# LOGGING SETUP
# ============================================================================

class DualLogger:
    """Logger that writes to both console and file"""
    def __init__(self, run_log_path, full_log_path):
        self.run_log = open(run_log_path, 'w', encoding='utf-8')
        self.full_log = open(full_log_path, 'w', encoding='utf-8')
        self.console = sys.stdout
        
    def log_progress(self, message):
        """Log concise progress to console and run_log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        self.run_log.write(msg + "\n")
        self.run_log.flush()
        
    def log_detail(self, message):
        """Log detailed info to full_log only"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.full_log.write(f"[{timestamp}] {message}\n")
        self.full_log.flush()
        
    def close(self):
        self.run_log.close()
        self.full_log.close()

# Global logger
logger = None

# ================================
# EXPERIMENT CONFIGURATION
# ================================

SEEDS = [2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]
USE_ADAP_POOL_OPTIONS = [False]
OPTIMIZERS = ['COBYLA'] # SPSA
FEATURE_MAP_TYPES = ['angle']  # Options: 'angle', 'amplitude'

# Experiment order configuration
# Options: 'optimizer_first', 'feature_map_first', 'seed_first', 'adaptive_first'
# Default: 'optimizer_first' -> optimizer > feature_map > seed > adaptive
# Example: 'adaptive_first' -> optimizer > feature_map > adaptive > seed
EXPERIMENT_ORDER = 'adaptive_first'  # Run all seeds with adaptive=False first, then adaptive=True

# Fixed parameters
EPOCHS = 500
NUM_TRAIN = 100
NUM_VAL = 74                    # Extended validation size
NUM_VAL_QUICK = 15              # Quick validation size (subset of NUM_VAL)
NUM_VAL_EXTENDED_INTERVAL = 5   # Run extended validation every N epochs
NUM_TEST = 250
PATIENCE = -1
SAVE_EVERY_X_EPOCH = 40
NOISE_SCALE_ON_IMAGES = 0.5

# Circuit parameters
N = 4
M = 4
STRIPE_LEN = 3
NUM_QUBITS_IN = 16
NUM_QUBITS_OUT = 1
SECRET_STRING="noReuse"

# =================================
# EXPERIMENT RUNNER
# =================================

def estimate_remaining_time(completed_results, remaining_experiments):
    """
    Estimate remaining time based on completed experiments.
    
    Args:
        completed_results: List of completed experiment results
        remaining_experiments: Number of experiments remaining
    
    Returns:
        str: Formatted time estimate
    """
    if not completed_results or remaining_experiments == 0:
        return "Unknown"
    
    # Filter successful results
    successful = [r for r in completed_results if 'time_minutes' in r]
    if not successful:
        return "Unknown"
    
    avg_time = np.mean([r['time_minutes'] for r in successful])
    remaining_minutes = avg_time * remaining_experiments
    
    if remaining_minutes < 60:
        return f"~{remaining_minutes:.0f} minutes"
    else:
        hours = remaining_minutes / 60
        return f"~{hours:.1f} hours"


def run_single_experiment(seed, use_adap_pool, optimizer_type, feature_map_type, experiment_num, total_experiments, output_base):
    """
    Run a single experiment with given seed, adaptive pooling configuration, optimizer, and feature map type.
    
    Args:
        seed: Random seed for reproducibility
        use_adap_pool: Whether to use adaptive pooling
        optimizer_type: 'SPSA' or 'COBYLA'
        feature_map_type: 'angle' or 'amplitude'
        experiment_num: Current experiment number (for progress tracking)
        total_experiments: Total number of experiments
        output_base: Base output directory for this run
    
    Returns:
        dict: Results including final accuracy and training history
    """
    global logger
    
    # Concise progress header
    logger.log_progress("="*80)
    logger.log_progress(f"Experiment {experiment_num}/{total_experiments}: Seed={seed}, Adaptive={use_adap_pool}, Optimizer={optimizer_type}, FeatureMap={feature_map_type}")
    logger.log_progress("="*80)
    
    # Detailed logging
    logger.log_detail(f"Starting experiment {experiment_num}/{total_experiments}")
    logger.log_detail(f"  Seed: {seed}, Adaptive: {use_adap_pool}, Optimizer: {optimizer_type}, FeatureMap: {feature_map_type}")
    
    start_time = time.time()
    
    # Set seeds
    algorithm_globals.random_seed = seed
    
    # Setup output directory
    saved_model_name = f"{seed}_{use_adap_pool}adap_{optimizer_type}_{feature_map_type}.pkl"
    output_dir = os.path.join(output_base, f"{seed}_{use_adap_pool}adap_{optimizer_type}_{feature_map_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # 1. BUILD CIRCUIT
    # ========================================================================
    logger.log_detail("[1/6] Building quantum circuit...")
    
    full_circuit, input_params, weight_params, observable = get_circuit(
        n=N, m=M, 
        num_qubits_in=NUM_QUBITS_IN, 
        num_qubits_out=NUM_QUBITS_OUT,
        use_adap_pool=use_adap_pool,
        feature_map_type=feature_map_type,
        secret_string=SECRET_STRING,
    )
    
    logger.log_detail(f"   Circuit depth: {full_circuit.depth()}, size: {full_circuit.size()}")
    
    # ========================================================================
    # 2. SETUP BACKEND
    # ========================================================================
    logger.log_detail("[2/6] Setting up quantum backend...")
    
    backend = AerSimulator(
        method='statevector',
        device='CPU',
        max_parallel_threads=0
    )
    
    # ========================================================================
    # 3. CREATE QNN
    # ========================================================================
    logger.log_detail("[3/6] Creating QNN...")
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(full_circuit)
    isa_observables = observable.apply_layout(isa_circuit.layout)
    
    logger.log_detail(f"   After transpile: depth={isa_circuit.depth()}, gates={isa_circuit.size()}")
    
    estimator = Estimator(mode=backend)
    
    qnn = EstimatorQNN(
        circuit=isa_circuit,
        observables=isa_observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator
    )
    
    logger.log_detail(f"   QNN created with {len(qnn.weight_params)} parameters")
    
    # ========================================================================
    # 4. GENERATE DATASET
    # ========================================================================
    logger.log_detail("[4/6] Generating dataset...")
    
    train_images, train_labels = generate_dataset(
        NUM_TRAIN, noise_scale=NOISE_SCALE_ON_IMAGES, n=N, m=M, 
        stripe_length=STRIPE_LEN, seed=seed
    )
    val_images, val_labels = generate_dataset(
        NUM_VAL, noise_scale=NOISE_SCALE_ON_IMAGES, n=N, m=M, 
        stripe_length=STRIPE_LEN, seed=seed+1
    )
    test_images, test_labels = generate_dataset(
        NUM_TEST, noise_scale=NOISE_SCALE_ON_IMAGES, n=N, m=M, 
        stripe_length=STRIPE_LEN, seed=seed+2
    )
    
    logger.log_detail(f"   Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Preprocess images based on feature map type
    if feature_map_type == 'amplitude':
        logger.log_detail("   Preprocessing for amplitude encoding...")
        # Reshape to 2D arrays (required by images_to_angles_batch_dual)
        imgs_train_2d = [np.asarray(img, dtype=float).reshape(N, M) for img in train_images]
        imgs_val_2d = [np.asarray(img, dtype=float).reshape(N, M) for img in val_images]
        imgs_test_2d = [np.asarray(img, dtype=float).reshape(N, M) for img in test_images]
        
        # Convert to amplitude angles (dual encoding: rows + columns)
        target_len = 16  # 2^4 for padding to next power of 2
        train_images = images_to_angles_batch_dual(imgs_train_2d, target_length=target_len)
        val_images = images_to_angles_batch_dual(imgs_val_2d, target_length=target_len)
        test_images = images_to_angles_batch_dual(imgs_test_2d, target_length=target_len)
        logger.log_detail(f"   Amplitude encoding: train_images shape = {train_images.shape}")
    else:
        # Angle encoding: use images as-is (flat vectors)
        train_images = np.asarray(train_images)
        val_images = np.asarray(val_images)
        test_images = np.asarray(test_images)
        logger.log_detail(f"   Angle encoding: train_images shape = {train_images.shape}")
    
    # ========================================================================
    # 5. TRAIN MODEL
    # ========================================================================
    logger.log_detail("[5/6] Training model...")
    logger.log_progress(f"  Training started...")
    
    initial_point = init_params_aligned(
        qnn, 
        is_adap_pooled=use_adap_pool, 
        adaptive_pool_gates=["a1_0_p"], 
        seed=seed
    )
    
    training_callback = EarlyStoppingCallback(
        qnn=qnn,
        val_images=val_images,
        val_labels=val_labels,
        output_dir=output_dir,
        saved_model_name=saved_model_name,
        save_every_x_epoch=SAVE_EVERY_X_EPOCH,
        patience=PATIENCE,
        plot=False,  # Disable interactive plotting for automated runs
        quick_val_size=NUM_VAL_QUICK,
        extended_val_interval=NUM_VAL_EXTENDED_INTERVAL,
    )
    
    if optimizer_type == 'SPSA':
        logger.log_detail(f"   Using SPSA optimizer with maxiter={EPOCHS}")
        spsa_callback_wrapper = SPSACallbackWrapper(training_callback)
        classifier = NeuralNetworkClassifier(
            qnn,
            optimizer=SPSA(maxiter=EPOCHS, callback=spsa_callback_wrapper),
            callback=None,
            initial_point=initial_point,
        )
    elif optimizer_type == 'COBYLA':
        logger.log_detail(f"   Using COBYLA optimizer with maxiter={EPOCHS}")
        classifier = NeuralNetworkClassifier(
            qnn,
            optimizer=COBYLA(maxiter=EPOCHS),
            callback=training_callback,
            initial_point=initial_point,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    x = np.asarray(train_images)
    y = np.asarray(train_labels)
    
    try:
        classifier.fit(x, y)
    except RuntimeError as e:
        if str(e) == "EARLY_STOPPING_TRIGGERED":
            logger.log_detail(f"   Early stopping triggered")
        else:
            raise
    
    logger.log_progress(f"  Training complete")
    
    # ========================================================================
    # 6. EVALUATE
    # ========================================================================
    logger.log_detail("[6/6] Evaluating on test set...")
    
    best_weights, best_obj, best_val = training_callback.get_best()
    best_ext_val_weights, best_ext_val_acc = training_callback.get_best_extended_val()
    
    # Test evaluation with best loss model
    x_test = np.asarray(test_images)
    y_test = np.asarray(test_labels)
    
    def evaluate_test_accuracy(weights):
        """Helper to evaluate test accuracy for given weights"""
        raw = qnn.forward(x_test, weights)
        raw = np.asarray(raw).reshape(-1)
        y_predict = np.where(raw >= 0.0, 1, -1)
        return float(np.mean(y_predict == y_test))
    
    # Evaluate best loss model
    test_acc_best_loss = evaluate_test_accuracy(best_weights)
    logger.log_detail(f"   Test accuracy (best loss model): {test_acc_best_loss:.4f}")
    
    # Evaluate best extended validation model
    test_acc_best_ext_val = None
    if best_ext_val_weights is not None:
        test_acc_best_ext_val = evaluate_test_accuracy(best_ext_val_weights)
        logger.log_detail(f"   Test accuracy (best extended val model): {test_acc_best_ext_val:.4f}")
        logger.log_detail(f"   (Extended val acc: {best_ext_val_acc:.4f})")
    
    # Evaluate specific epoch checkpoints (120, 200, 240)
    checkpoint_epochs = [120, 200, 240]
    test_acc_at_epochs = {}
    
    for epoch in checkpoint_epochs:
        checkpoint_file = os.path.join(output_dir, f"epoch_{epoch}_{saved_model_name}")
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    checkpoint_weights = checkpoint_data['weights']
                test_acc_epoch = evaluate_test_accuracy(checkpoint_weights)
                test_acc_at_epochs[epoch] = test_acc_epoch
                logger.log_detail(f"   Test accuracy (epoch {epoch}): {test_acc_epoch:.4f}")
            except Exception as e:
                logger.log_detail(f"   Warning: Could not load checkpoint for epoch {epoch}: {e}")
                test_acc_at_epochs[epoch] = None
        else:
            logger.log_detail(f"   Warning: Checkpoint for epoch {epoch} not found")
            test_acc_at_epochs[epoch] = None
    
    # Save results
    trained_params = {
        'weights': best_weights if best_weights is not None else classifier._fit_result.x,
        'objective_values': training_callback.get_histories()[0],
        'val_acc_history': training_callback.get_histories()[1],
        'extended_val_acc_history': training_callback.get_extended_val_history(),
        'best_objective_value': best_obj,
        'best_val_accuracy': best_val,
        'best_extended_val_accuracy': best_ext_val_acc,
        'best_extended_val_weights': best_ext_val_weights,
        'test_accuracy': test_acc_best_loss,  # Best loss model
        'test_acc_best_ext_val': test_acc_best_ext_val,  # Best extended val model
        'test_acc_at_epochs': test_acc_at_epochs,  # Checkpoint evaluations
        'seed': seed,
        'use_adap_pool': use_adap_pool,
        'optimizer': optimizer_type,
        'feature_map': feature_map_type,
    }
    
    with open(os.path.join(output_dir, f"final_{saved_model_name}"), 'wb') as f:
        pickle.dump(trained_params, f)
    
    # ========================================================================
    # SAVE TRAINING FIGURES
    # ========================================================================
    logger.log_detail("   Saving training plots...")
    try:
        fig = training_callback.plot_histories(show=False)
        if fig is not None:
            fig_path = os.path.join(output_dir, f"training_curves_{saved_model_name.replace('.pkl', '.png')}")
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.log_detail(f"   Training plots saved to: {fig_path}")
    except Exception as e:
        logger.log_detail(f"   Warning: Could not save training plots: {e}")
    
    elapsed_time = time.time() - start_time
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.log_progress(f"  Completed: Val={best_val:.4f}, ExtVal={best_ext_val_acc:.4f} Test={test_acc_best_loss:.4f}, TestExtVal={test_acc_best_ext_val if test_acc_best_ext_val is not None else 'N/A'}, Time={elapsed_time/60:.1f} min")
    
    logger.log_detail("="*80)
    logger.log_detail(f"EXPERIMENT {experiment_num}/{total_experiments} SUMMARY")
    logger.log_detail(f"  Seed: {seed}")
    logger.log_detail(f"  Adaptive: {use_adap_pool}")
    logger.log_detail(f"  Optimizer: {optimizer_type}")
    logger.log_detail(f"  Best Val Acc (quick): {best_val:.4f}")
    logger.log_detail(f"  Best Extended Val Acc: {best_ext_val_acc:.4f}")
    logger.log_detail(f"  Test Acc (best loss): {test_acc_best_loss:.4f}")
    logger.log_detail(f"  Test Acc (best ext val): {test_acc_best_ext_val if test_acc_best_ext_val is not None else 'N/A'}")
    for epoch in checkpoint_epochs:
        if epoch in test_acc_at_epochs and test_acc_at_epochs[epoch] is not None:
            logger.log_detail(f"  Test Acc (epoch {epoch}): {test_acc_at_epochs[epoch]:.4f}")
    logger.log_detail(f"  Training Epochs: {len(training_callback.get_histories()[0])}")
    logger.log_detail(f"  Time: {elapsed_time/60:.1f} minutes")
    logger.log_detail("="*80)
    
    return {
        'seed': seed,
        'use_adap_pool': use_adap_pool,
        'optimizer': optimizer_type,
        'feature_map': feature_map_type,
        'best_val_acc': best_val,
        'best_extended_val_acc': best_ext_val_acc,
        'test_acc': test_acc_best_loss,  # Best loss model
        'test_acc_best_ext_val': test_acc_best_ext_val,  # Best extended val model
        'test_acc_at_epochs': test_acc_at_epochs,  # Epoch checkpoints
        'training_epochs': len(training_callback.get_histories()[0]),'time_minutes': elapsed_time / 60,
        'output_dir': output_dir,
    }


def run_all_experiments():
    """
    Run all experiment combinations.
    """
    global logger
    
    # Create unique output directory for this run
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(script_dir, "output", f"run_{timestamp}")
    os.makedirs(output_base, exist_ok=True)
    
    # Setup logging
    logger = DualLogger(
        os.path.join(output_base, "run_log.txt"),
        os.path.join(output_base, "full_log.txt")
    )
    
    # Calculate total experiments
    total_experiments = len(SEEDS) * len(USE_ADAP_POOL_OPTIONS) * len(OPTIMIZERS) * len(FEATURE_MAP_TYPES)
    
    logger.log_progress("="*80)
    logger.log_progress("STARTING AUTOMATED EXPERIMENT SUITE")
    logger.log_progress("="*80)
    logger.log_progress(f"Total: {total_experiments} experiments")
    logger.log_progress(f"Seeds: {SEEDS}")
    logger.log_progress(f"Adaptive pooling: {USE_ADAP_POOL_OPTIONS}")
    logger.log_progress(f"Optimizers: {OPTIMIZERS}")
    logger.log_progress(f"Feature maps: {FEATURE_MAP_TYPES}")
    logger.log_progress(f"Experiment order: {EXPERIMENT_ORDER}")
    logger.log_progress(f"Epochs: {EPOCHS}")
    logger.log_progress(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log_progress("="*80)
    
    logger.log_detail(f"Configuration details:")
    logger.log_detail(f"  NUM_TRAIN={NUM_TRAIN}, NUM_VAL={NUM_VAL}, NUM_TEST={NUM_TEST}")
    logger.log_detail(f"  PATIENCE={PATIENCE}, SAVE_EVERY_X_EPOCH={SAVE_EVERY_X_EPOCH}")
    logger.log_detail(f"  NOISE_SCALE={NOISE_SCALE_ON_IMAGES}")
    logger.log_detail(f"  Circuit: {N}x{M}, {NUM_QUBITS_IN} qubits")
    logger.log_detail(f"  Output directory: {output_base}")
    
    results = []
    total_start = time.time()
    experiment_num = 0
    
    # Generate experiment list based on EXPERIMENT_ORDER
    experiments = []
    if EXPERIMENT_ORDER == 'optimizer_first':
        # optimizer > feature_map > seed > adaptive
        for optimizer_type in OPTIMIZERS:
            for feature_map_type in FEATURE_MAP_TYPES:
                for seed in SEEDS:
                    for use_adap_pool in USE_ADAP_POOL_OPTIONS:
                        experiments.append((optimizer_type, feature_map_type, seed, use_adap_pool))
    elif EXPERIMENT_ORDER == 'feature_map_first':
        # optimizer > feature_map > adaptive > seed
        for optimizer_type in OPTIMIZERS:
            for feature_map_type in FEATURE_MAP_TYPES:
                for use_adap_pool in USE_ADAP_POOL_OPTIONS:
                    for seed in SEEDS:
                        experiments.append((optimizer_type, feature_map_type, seed, use_adap_pool))
    elif EXPERIMENT_ORDER == 'seed_first':
        # optimizer > seed > feature_map > adaptive
        for optimizer_type in OPTIMIZERS:
            for seed in SEEDS:
                for feature_map_type in FEATURE_MAP_TYPES:
                    for use_adap_pool in USE_ADAP_POOL_OPTIONS:
                        experiments.append((optimizer_type, feature_map_type, seed, use_adap_pool))
    elif EXPERIMENT_ORDER == 'adaptive_first':
        # optimizer > feature_map > adaptive > seed (all seeds with adaptive=False first, then all with adaptive=True)
        for optimizer_type in OPTIMIZERS:
            for feature_map_type in FEATURE_MAP_TYPES:
                for use_adap_pool in USE_ADAP_POOL_OPTIONS:
                    for seed in SEEDS:
                        experiments.append((optimizer_type, feature_map_type, seed, use_adap_pool))
    else:
        raise ValueError(f"Unknown EXPERIMENT_ORDER: {EXPERIMENT_ORDER}. Valid options: 'optimizer_first', 'feature_map_first', 'seed_first', 'adaptive_first'")
    
    logger.log_detail(f"Experiment order: {EXPERIMENT_ORDER}")
    logger.log_detail(f"Total experiments to run: {len(experiments)}")
    
    # Run experiments in the configured order
    for optimizer_type, feature_map_type, seed, use_adap_pool in experiments:
                    experiment_num += 1
                    
                    # Estimate remaining time
                    remaining = total_experiments - (experiment_num - 1)
                    est_time = estimate_remaining_time(results, remaining)
                    
                    logger.log_progress("")
                    logger.log_progress(f"PROGRESS: {experiment_num-1}/{total_experiments} complete, {est_time} remaining")
                    logger.log_progress("")
                    
                    try:
                        result = run_single_experiment(
                            seed, use_adap_pool, optimizer_type, feature_map_type,
                            experiment_num, total_experiments, output_base
                        )
                        results.append(result)
                    except Exception as e:
                        logger.log_progress(f"  ❌ ERROR: {e}")
                        logger.log_detail(f"ERROR in experiment {experiment_num}")
                        logger.log_detail(f"  Seed: {seed}, Adaptive: {use_adap_pool}, Optimizer: {optimizer_type}, FeatureMap: {feature_map_type}")
                        logger.log_detail(f"  Exception: {e}")
                        import traceback
                        logger.log_detail(traceback.format_exc())
                        results.append({
                            'seed': seed,
                            'use_adap_pool': use_adap_pool,
                            'optimizer': optimizer_type,
                            'feature_map': feature_map_type,
                            'error': str(e)
                        })
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    total_time = time.time() - total_start
    
    logger.log_progress("")
    logger.log_progress("="*80)
    logger.log_progress("ALL EXPERIMENTS COMPLETED")
    logger.log_progress("="*80)
    logger.log_progress(f"Total time: {total_time/3600:.2f} hours")
    logger.log_progress(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log_progress("")
    
    # Create detailed results table
    create_results_table(results, total_time, output_base)
    
    # Save summary
    with open(os.path.join(output_base, "experiment_summary.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    logger.log_progress(f"Summary saved to: {os.path.join(output_base, 'experiment_summary.pkl')}")
    logger.log_progress(f"Results table saved to: {os.path.join(output_base, 'final_results_table.txt')}")
    logger.log_progress("")
    logger.log_progress(f"All results in: {output_base}")
    logger.log_progress("Run 'python analyze_results.py' to generate visualizations")
    logger.log_progress("="*80)
    
    logger.close()


def create_results_table(results, total_time, output_base):
    """Create a detailed human-readable results table"""
    with open(os.path.join(output_base, "final_results_table.txt"), 'w') as f:
        f.write("="*140 + "\n")
        f.write(" "*50 + "FINAL EXPERIMENT RESULTS\n")
        f.write("="*140 + "\n")
        f.write(f"{'Seed':<6} {'Adaptive':<9} {'Optimizer':<10} {'FeatureMap':<11} {'ValAcc':<8} {'ExtValAcc':<10} "
                f"{'Test(Best)':<11} {'Test(ExtVal)':<13} {'Test(E120)':<11} {'Test(E200)':<11} {'Test(E240)':<11} "
                f"{'Time(min)':<10}\n")
        f.write("-"*140 + "\n")
        
        for result in results:
            if 'error' in result:
                f.write(f"{result['seed']:<6} {str(result['use_adap_pool']):<9} "
                       f"{result.get('optimizer', 'N/A'):<10} {result.get('feature_map', 'N/A'):<11} {'ERROR':<8} {'---':<10} "
                       f"{'---':<11} {'---':<13} {'---':<11} {'---':<11} {'---':<11} {'---':<10}\n")
            else:
                test_120 = result['test_acc_at_epochs'].get(120)
                test_200 = result['test_acc_at_epochs'].get(200)
                test_240 = result['test_acc_at_epochs'].get(240)
                test_ext_val = result.get('test_acc_best_ext_val')
                
                f.write(f"{result['seed']:<6} {str(result['use_adap_pool']):<9} "
                       f"{result['optimizer']:<10} {result['feature_map']:<11} {result['best_val_acc']:<8.4f} "
                       f"{result.get('best_extended_val_acc', 0.0):<10.4f} "
                       f"{result['test_acc']:<11.4f} "
                       f"{test_ext_val if test_ext_val is not None else '---':<13} "
                       f"{test_120 if test_120 else '---':<11} "
                       f"{test_200 if test_200 else '---':<11} "
                       f"{test_240 if test_240 else '---':<11} "
                       f"{result['time_minutes']:<10.1f}\n")
        
        f.write("-"*140 + "\n")
        f.write("STATISTICS:\n")
        
        # Calculate statistics per configuration
        successful = [r for r in results if 'error' not in r]
        
        for optimizer in OPTIMIZERS:
            for feature_map in FEATURE_MAP_TYPES:
                for use_adap in [False, True]:
                    subset = [r for r in successful 
                             if r['optimizer'] == optimizer and r['use_adap_pool'] == use_adap and r['feature_map'] == feature_map]
                    if subset:
                        test_accs = [r['test_acc'] for r in subset]
                        test_ext_val_accs = [r.get('test_acc_best_ext_val') for r in subset 
                                            if r.get('test_acc_best_ext_val') is not None]
                        times = [r['time_minutes'] for r in subset]
                        adap_str = "With Adaptive" if use_adap else "No Adaptive"
                        f.write(f"  {optimizer} + {feature_map} + {adap_str:<15}: "
                               f"Mean={np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}   ")
                        if test_ext_val_accs:
                            f.write(f"ExtVal={np.mean(test_ext_val_accs):.4f} ± {np.std(test_ext_val_accs):.4f}   ")
                        f.write(f"(Avg time: {np.mean(times):.1f} min)\n")
        
        # Improvement analysis
        f.write("\n")
        for optimizer in OPTIMIZERS:
            for feature_map in FEATURE_MAP_TYPES:
                no_adap = [r['test_acc'] for r in successful 
                          if r['optimizer'] == optimizer and r['use_adap_pool'] == False and r['feature_map'] == feature_map]
                with_adap = [r['test_acc'] for r in successful 
                            if r['optimizer'] == optimizer and r['use_adap_pool'] == True and r['feature_map'] == feature_map]
                if no_adap and with_adap:
                    improvement = (np.mean(with_adap) - np.mean(no_adap)) / np.mean(no_adap) * 100
                    f.write(f"  Improvement ({optimizer} + {feature_map}): {improvement:+.2f}% with adaptive pooling\n")
        
        f.write("="*140 + "\n")
        f.write(f"Total execution time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)\n")
        f.write(f"Failed experiments: {len([r for r in results if 'error' in r])}/{len(results)}\n")
        f.write("="*140 + "\n")


if __name__ == "__main__":
    results = run_all_experiments()
