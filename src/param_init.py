"""
Parameter initialization utilities for QCNN experiments.
Ensures fair comparison between adaptive and non-adaptive pooling by keeping
matching gates aligned to the same parameter values.
"""

import numpy as np
from qiskit_machine_learning.utils import algorithm_globals


def init_params_aligned(qnn, is_adap_pooled=False, adaptive_pool_gates=None, seed=None, do_print=False):
    """
    Initialize QNN parameters with optional alignment for adaptive pooling.
    
    Args:
        qnn: The quantum neural network
        is_adap_pooled: Whether the model uses adaptive pooling
        adaptive_pool_gates: List of gate prefixes to identify adaptive params (e.g., ["a1_0_p"])
        seed: Random seed for reproducibility
    
    Returns:
        Array of initialized parameters
    """
    if seed is not None:
        algorithm_globals.random_seed = seed
    
    num_params = len(qnn.weight_params)
    param_names = [str(p) for p in qnn.weight_params]
    
    # If not using adaptive pooling, return standard random initialization
    if not is_adap_pooled:
        print(f"Non-adaptive model: returning {num_params} random params")
        return algorithm_globals.random.uniform(0, 2*np.pi, num_params)
    
    # Identify which parameters are adaptive pooling params (every 4th: index % 4 == 3)
    adaptive_indices = []
    adaptive_names = []
    
    for i, name in enumerate(param_names):
        for adap_prefix in adaptive_pool_gates:
            if adap_prefix in name:
                # Extract index from parameter name like "a1_0_p[3]"
                try:
                    if '[' in name and ']' in name:
                        idx_str = name[name.rfind('[') + 1 : name.rfind(']')]
                        idx = int(idx_str)
                        if idx % 4 == 3:  # Every 4th parameter (positions 3, 7, 11, 15, etc.)
                            adaptive_indices.append(i)
                            adaptive_names.append(name)
                            break
                except (ValueError, IndexError):
                    continue
    
    num_adaptive_params = len(adaptive_indices)
    num_base_params = num_params - num_adaptive_params
    
    if do_print:
        print(f"Adaptive pooling parameter indices: {adaptive_indices}")
        print(f"Adaptive pooling parameter names: {adaptive_names}")
        print(f"Parameter breakdown:")
        print(f"  - Base gates: {num_base_params} params")
        print(f"  - Adaptive pooling: {num_adaptive_params} params")
        print(f"  - Total: {num_params} params")
    
    # Generate base parameters (all non-adaptive)
    base_params = algorithm_globals.random.uniform(0, 2*np.pi, num_base_params)
    
    # Generate adaptive parameters
    #adaptive_params = algorithm_globals.random.uniform(0, 2*np.pi, num_adaptive_params)
    adaptive_params = np.zeros(num_adaptive_params) 
    
    # Build result by inserting adaptive params at their correct positions
    result = np.zeros(num_params)
    base_idx = 0
    adap_idx = 0
    
    for i in range(num_params):
        if i in adaptive_indices:
            result[i] = adaptive_params[adap_idx]
            adap_idx += 1
        else:
            result[i] = base_params[base_idx]
            base_idx += 1
    
    print(f"Adaptive model: inserted {num_adaptive_params} adaptive params at positions {adaptive_indices}")
    
    return result
