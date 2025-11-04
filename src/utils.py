# type: ignore
# This is a file conatining helping functions building my qubit circuits

import numpy as np
import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
import math
import matplotlib.pyplot as plt
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


## -- CIRCUIT BUILDING BLOCKS

# Feature map
def create_feature_map(num_qubits, param_prefix=""):
    # A simple built-in feature map from Qiskit
    return ZFeatureMap(feature_dimension=num_qubits, reps=2, parameter_prefix=param_prefix)

# Concolutionay layers
def conv_circuit(params, enable_static_angles=True):
    target = QuantumCircuit(2)
    if enable_static_angles: target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    if enable_static_angles: target.rz(np.pi / 2, 0)
    return target


def conv_layer(num_qubits, param_prefix, shifted_repeat=True):
    """The conv layer enatngles every second, then again shifted by 1 in a circular pattern
    this currently needs even number of qubits to work."""
    assert num_qubits % 2 == 0
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    num_params = num_qubits * 3 if shifted_repeat else int(num_qubits * 1.5)
    params = ParameterVector(param_prefix, length=num_qubits * num_params)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        #qc.barrier() # Keep this barier in for better seperation of the circuits
        param_index += 3
    if shifted_repeat:
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            #qc.barrier()
            param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


# Pooling layers

def pool_circuit(qc, q0, q1, theta, cbit=None, reset_source=True, adaptive_pooling=False, adaptive_angle=None, enable_static_angles=True):
    """Apply pooling to existing circuit `qc`, qubit q0 will be pooled into q1 with an optional measurement into cbit."""
    if enable_static_angles: qc.rz(-np.pi / 2, q1)
    qc.cx(q1, q0)
    qc.rz(theta[0], q0)
    qc.ry(theta[1], q1)
    qc.cx(q0, q1)
    qc.ry(theta[2], q1)

    if reset_source:
        if cbit is None:
            raise ValueError("Must provide cbit if reset_source is True")
        qc.measure(q0, cbit)
        qc.reset(q0)
        if adaptive_pooling:
            # If we measured 1 just before the reset, add a ry on q1 with a little rotation
            angle = adaptive_angle if adaptive_angle is not None else 0.25 # there is a possibility to set the static value of the apadtive pooling here
            qc.ry(angle, q1).c_if(cbit, 1)  # Apply rotation only if measurement was 1
            # else do nothing


def pool_layer(sources, sinks, param_prefix, reset_source=False, creg=None, adaptive_pooling=False):
    "We pool sources into sinks, sources will be ignored and can be reset"
    num_qubits = max(max(sources), max(sinks)) + 1
    qreg = QuantumRegister(num_qubits, 'q')

    if creg is None:
        creg = ClassicalRegister(len(sources), 'c')  # Only as many bits as source qubits

    qc = QuantumCircuit(qreg, creg, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * (3 + (1 if adaptive_pooling else 0)))

    param_index = 0
    for i, (source, sink) in enumerate(zip(sources, sinks)):
        theta = params[param_index:param_index + 3]
        adaptive_angle = params[param_index + 3] if adaptive_pooling else None
        cbit = creg[i] if reset_source else None
        #qc.barrier() # Keep the barrier for cleaner separation of conv circuits
        pool_circuit(qc, qreg[source], qreg[sink], theta, cbit, reset_source, adaptive_pooling, adaptive_angle)
        param_index += (3 + (1 if adaptive_pooling else 0))

    return qc


# --- Simple quantum dense / fully-connected layer (no observables) --- #
from qiskit.circuit import QuantumCircuit, ParameterVector

def dense_layer(num_qubits, param_prefix="fc", reps=1, entanglement="chain"):
    """
    Build a simple fully-connected quantum layer using rotation + entanglement blocks.

    Args:
        num_qubits (int): Number of qubits this layer acts on.
        param_prefix (str): Prefix for the trainable parameters.
        reps (int): Number of repetition blocks (depth of layer).
        entanglement (str): 'chain' or 'ring' — how qubits are entangled.

    Returns:
        qc (QuantumCircuit): The parameterized quantum dense layer circuit.
    """
    qc = QuantumCircuit(num_qubits, name=f"{param_prefix}_dense")
    params = ParameterVector(param_prefix, length=reps * num_qubits * 2)

    pidx = 0
    for r in range(reps):
        # Single-qubit rotations
        for q in range(num_qubits):
            qc.rz(params[pidx], q); pidx += 1
            qc.ry(params[pidx], q); pidx += 1

        # Entangling layer
        if num_qubits > 1:
            for q in range(num_qubits - 1):
                qc.cx(q, q + 1)
            if entanglement == "ring":
                qc.cx(num_qubits - 1, 0)
        #qc.barrier()  # optional, keep if you like visual separation
    return qc


# Putting it all together to a bigger circuit

def build_qcnn_circuit(num_qubits_in, num_qubits_out=1, param_prefix="", reset_source=False, adaptive_pooling=False, ignore_last_x_adap_n_reset=0, shift_repeat_conv=True, conv_repeats=1):
    """Builds a QCNN circuit with alternating convolution and pooling layers.
    Args:
        reset_source: If True, pooled qubits will be reset to |0> after disposal.
        adaptive_pooling: If True, measurement outcome from disposed pooled qubits will be fed back into the network.
        ignore_last_x_adap_pool: If > 0, adaptive pooling will be ignored in the last N pooling layers.
    
    """

    assert (num_qubits_in & (num_qubits_in - 1)) == 0, "Number of qubits in must be a power of 2" # holy bit alge brah
    assert (num_qubits_out & (num_qubits_out - 1)) == 0, "Number of qubits out must be a power of 2"

    if not reset_source and adaptive_pooling:
        print(f"WARNING!\n Warning: adaptive pooling requires reset_source=True. Adaptive pooling will not be applied with these settings. Circuit: {param_prefix}")
        adaptive_pooling = False

    # Create quantum and classical registers.
    qreg = QuantumRegister(num_qubits_in, 'q')
    creg = ClassicalRegister(num_qubits_in, 'c')
    ansatz = QuantumCircuit(qreg, creg, name="Quantum Convolutional Neural Network")

    # Start with all qubits.
    current_qubits = num_qubits_in
    layer_num = 0

    # Apply alternating convolution and pooling layers until one qubit remains.
    while current_qubits > num_qubits_out or (current_qubits == num_qubits_out and num_qubits_in == num_qubits_out):
        for conv_rep in range(conv_repeats):
            # Convolutional Layer
            prefix = f"{param_prefix}_{layer_num}/{conv_rep}_c" if conv_repeats > 1 else f"{param_prefix}_{layer_num}_c"
            conv_circ = conv_layer(current_qubits, prefix, shifted_repeat=shift_repeat_conv)
            ansatz.compose(conv_circ, list(range((num_qubits_in - current_qubits), num_qubits_in)), inplace=True)

        if current_qubits == num_qubits_out: # in case of num in = num out we still add a conv layer
            break

        # Pooling Layer
        do_adaptive_pooling = adaptive_pooling  and (current_qubits > math.pow(2, ignore_last_x_adap_n_reset) * num_qubits_out)
        do_reset            = reset_source      and (current_qubits > math.pow(2, ignore_last_x_adap_n_reset) * num_qubits_out)
        pool = pool_layer(list(range(current_qubits//2)), list(range(current_qubits//2, current_qubits)), f"{param_prefix}_{layer_num}_p", reset_source=do_reset, creg=creg, adaptive_pooling=do_adaptive_pooling)
        ansatz.compose(pool, list(range((num_qubits_in - current_qubits), num_qubits_in)), inplace=True)
        current_qubits = current_qubits // 2
        layer_num += 1
        
    return ansatz



# -- DATASET

def generate_dataset(num_images, noise_scale=0.5, n=3, m=4, stripe_length=2, seed=None):
    """Generates a dataset of images with horizontal and vertical stripes. Always return even amount of both classes."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    # Dynamically crop stripe_length if it exceeds dimensions
    max_horizontal_stripe = min(stripe_length, m)
    max_vertical_stripe = min(stripe_length, n)
    
    # Precompute all unique horizontal and vertical patterns
    horizontal_patterns = []
    vertical_patterns = []

    # Horizontal lines
    for i in range(n):  # n rows
        for j in range(m - max_horizontal_stripe + 1):  # possible horizontal positions
            hor_array = np.zeros((n, m))
            hor_array[i, j:j+max_horizontal_stripe] = np.pi / 2
            horizontal_patterns.append(hor_array.flatten())

    # Vertical lines
    for i in range(n - max_vertical_stripe + 1):  # possible vertical positions
        for j in range(m):  # m columns
            ver_array = np.zeros((n, m))
            ver_array[i:i+max_vertical_stripe, j] = np.pi / 2
            vertical_patterns.append(ver_array.flatten())

    # Balanced sampling: half horizontal, half vertical
    half = num_images // 2
    h_indices = rng.choice(len(horizontal_patterns), size=half, replace=half > len(horizontal_patterns))
    v_indices = rng.choice(len(vertical_patterns), size=half, replace=half > len(vertical_patterns))

    sampled_images = []
    sampled_labels = []

    # Add noise after sampling, then label and collect
    for idx in h_indices:
        base = horizontal_patterns[idx]
        noisy = np.array([
            pixel if pixel != 0 else rng.uniform(0, np.pi / 2 * noise_scale)
            for pixel in base
        ])
        sampled_images.append(noisy)
        sampled_labels.append(-1)

    for idx in v_indices:
        base = vertical_patterns[idx]
        noisy = np.array([
            pixel if pixel != 0 else rng.uniform(0, np.pi / 2 * noise_scale)
            for pixel in base
        ])
        sampled_images.append(noisy)
        sampled_labels.append(1)

    # Shuffle to mix classes
    perm = rng.permutation(len(sampled_images))
    sampled_images = [sampled_images[i] for i in perm]
    sampled_labels = [sampled_labels[i] for i in perm]

    return sampled_images, sampled_labels


# -- MISC

def no_op_minimizer(fun, x0, jac=None, bounds=None, **kwargs): # This function makes sure that the weight are not changed by acciedent
    loss = float(fun(x0))
    return x0, loss, {}