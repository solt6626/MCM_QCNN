"""
This file containg prebuilt circuits for various configurations.
Feel free to add you own prebuilt circuit here and import it by get_circuit()
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
import sys, os

from utils import *
from amp_encode.utils_amp import *


def get_circuit(n, m, num_qubits_in, num_qubits_out, reset_source=False, use_adap_pool=False, return_params=True, secret_string="", feature_map_type="angle"):
    """
    Factory function to get the appropriate prebuilt circuit based on dimensions and configuration.
    
    Args:
        n, m: Image dimensions (n rows, m columns)
        num_qubits_in: Number of input qubits
        num_qubits_out: Number of output qubits
        reset_source: Whether to use qubit reset in pooling layers
        use_adap_pool: Whether to use adaptive pooling
        return_params: Whether to return parameter lists
        secret_string: Version selector for alternative circuit implementations
        feature_map_type: 'angle' (default) or 'amplitude' for dual amplitude encoding
    
    Returns:
        Circuit (and optionally input_params, weight_params, observable)
    """
    config_key = (n, m, num_qubits_in, num_qubits_out)     # Match configuration and return appropriate circuit
    
    # 3x4 image, 8 qubits, 1 output qubit - AMPLITUDE ENCODING
    if config_key == (3, 4, 8, 1) and feature_map_type == "amplitude":
        print("Using circuit_3x4_8in_amp")
        return circuit_3x4_8in_amp(use_adap_pool, return_params)
    
    # 3x4 image, 8 qubits, 2 output qubits
    if config_key == (3, 4, 8, 2):
        print("Using circuit_3x4_8in_2out")
        return circuit_3x4_8in_2out(use_adap_pool, return_params)

    # 3x4 image, 8 qubits, 1 output qubit (V2 variant)
    if config_key == (3, 4, 8, 1) and secret_string == "v2":
        print("Using circuit_3x4_8inV2")
        return circuit_3x4_8inV2(use_adap_pool, return_params)
    
    # 3x4 image, 12 qubits, 2 output qubits (noreuse by default but can use adap)
    if config_key == (3, 4, 12, 2) and feature_map_type == "angle":
        print("Using circuit_3x4_12in_2out")
        return circuit_3x4_12in_2out(use_adap_pool, return_params)
    
    # 4x4 image, 8 qubits - AMPLITUDE ENCODING
    if config_key == (4, 4, 8, 1) and feature_map_type == "amplitude":
        print("Using circuit_4x4_8in_amp")
        return circuit_4x4_8in_amp(use_adap_pool, return_params)
    
    # 4x4 image, 8 qubits with reuse (angle encoding)
    if config_key == (4, 4, 8, 1):
        print("Using circuit_4x4_8in_horizontal")
        return circuit_4x4_8in_horizontal(use_adap_pool, return_params)
    
    # 4x4 image, 16 qubits (V0 variant - original version)
    if config_key == (4, 4, 16, 1) and not use_adap_pool and secret_string == "v0":
        print("Using circuit_4x4_16inV0")
        return circuit_4x4_16inV0(return_params=return_params)

    # 4x4 image, 16 qubits without reuse (expansion of circuit_4x4_8in to 16 qubits)
    if config_key == (4, 4, 16, 1) and secret_string == "noReuse" and feature_map_type == "angle":
        print("Using circuit_4x4_16in_noReuse_horizontal")
        return circuit_4x4_16in_noReuse_horizontal(return_params=return_params)

    # No matching circuit found
    raise ValueError(
        f"No prebuilt circuit for configuration:\n"
        f"  Dimensions: {n}x{m}\n"
        f"  Qubits: {num_qubits_in} in, {num_qubits_out} out\n"
        f"  Reset source: {reset_source}\n"
        f"  Adaptive pooling: {use_adap_pool}\n"
        f"  Feature map type: '{feature_map_type}'\n"
        f"  Version: '{secret_string}' (use 'v0', 'v2', or '' for default)"
    )


def circuit_3x4_8in_2out(use_adap_pool=False, return_params=False):
    """This circuit uses 8 qubits to classify 12 inputs, with 4 qubits reused after the first pooling."""
    num_qubits_in = 8
    qreg = QuantumRegister(num_qubits_in, 'q')
    creg = ClassicalRegister(num_qubits_in, 'c')
    full_circuit = QuantumCircuit(qreg, creg)

    # Encode pixels 0-3 into qubits 0-3
    feature_map_0_3 = create_feature_map(num_qubits=4, param_prefix="f_1")
    full_circuit.compose(feature_map_0_3, [0, 1, 2, 3], inplace=True)

    # Encode pixels 8-11 into qubits 4-7
    feature_map_8_11 = create_feature_map(num_qubits=4, param_prefix="f_3")
    full_circuit.compose(feature_map_8_11, [4, 5, 6, 7], inplace=True)

    # Build the QCNN circuit.
    ansatz_1 = build_qcnn_circuit(num_qubits_in, num_qubits_out=4, param_prefix="a1", reset_source=True, adaptive_pooling=use_adap_pool)

    full_circuit.compose(ansatz_1, range(num_qubits_in), inplace=True)

    # the 4 pixels coming in late with the reuse - this is the one and last reuse
    feature_map_4_7 = create_feature_map(num_qubits=4, param_prefix="f_2")
    full_circuit.compose(feature_map_4_7, [0, 1, 2, 3], inplace=True)

    ansatz_2 = build_qcnn_circuit(num_qubits_in=8, num_qubits_out=2, param_prefix="a2", reset_source=False, adaptive_pooling=False)
    full_circuit.compose(ansatz_2, range(num_qubits_in), inplace=True)

    observable = SparsePauliOp.from_list([("Z"*2 + "I"*6, 1)])

    if return_params:
        # Return circuit along with parameter groups for QNN construction
        input_params = (list(feature_map_0_3.parameters) + 
                       list(feature_map_4_7.parameters) + 
                       list(feature_map_8_11.parameters))
        weight_params = list(ansatz_1.parameters) + list(ansatz_2.parameters)
        return full_circuit, input_params, weight_params, observable
    
    return full_circuit


def circuit_4x4_8in_horizontal(use_adap_pool=False, return_params=False):
    """Pixels mapped to qubits:
    0, 1, 2, 3
    4, 5, 6, 7
    0, 1, 4, 5
    2, 7, 4, 5
    
    Where the image is:
    0, 1, 2, 3
    4, 5, 6, 7
    8, 9,10,11
    12,13,14,15
    """

    num_qubits_in = 8
    qreg = QuantumRegister(num_qubits_in, 'q')
    creg = ClassicalRegister(num_qubits_in, 'c')
    full_circuit = QuantumCircuit(qreg, creg)


    # Feature maps for the first two rows
    feature_map_0_1 = create_feature_map(num_qubits=2, param_prefix="f1") 
    full_circuit.compose(feature_map_0_1, [0, 1], inplace=True)

    feature_map_2_3 = create_feature_map(num_qubits=2, param_prefix="f2")
    full_circuit.compose(feature_map_2_3, [2, 3], inplace=True)

    feature_map_4_5 = create_feature_map(num_qubits=2, param_prefix="f3")
    full_circuit.compose(feature_map_4_5, [4, 5], inplace=True)

    feature_map_6_7 = create_feature_map(num_qubits=2, param_prefix="f4")
    full_circuit.compose(feature_map_6_7, [6, 7], inplace=True)

    #---
    # Horizontal convolution for the first two rows
    ansatz_1 = build_qcnn_circuit(4, 4, "a1", shift_repeat_conv=True, conv_repeats=2)
    full_circuit.compose(ansatz_1, [0, 1, 2, 3], inplace=True)

    ansatz_2 = build_qcnn_circuit(4, 4, "a2", shift_repeat_conv=True, conv_repeats=2)
    full_circuit.compose(ansatz_2, [4, 5, 6, 7], inplace=True)

    # Then they are pooled to the right
    pool_1 = pool_layer([0, 1], [2, 3], "p1", reset_source=True)
    full_circuit.compose(pool_1, [0, 1, 2, 3], inplace=True)

    pool_2 = pool_layer([0, 1], [2, 3], "p2", reset_source=True)
    full_circuit.compose(pool_2, [4, 5, 6, 7], inplace=True)

    # qubits 0, 1, 4, 5 are now free and reusable
    #---

    # Third row encoding and convolutions
    feature_map_8_9 = create_feature_map(num_qubits=2, param_prefix="f5")
    full_circuit.compose(feature_map_8_9, [0, 1], inplace=True)

    feature_map_10_11 = create_feature_map(num_qubits=2, param_prefix="f6")
    full_circuit.compose(feature_map_10_11, [4, 5], inplace=True)

    ansatz_3 = build_qcnn_circuit(4, 4, "a3", shift_repeat_conv=True, conv_repeats=2)
    full_circuit.compose(ansatz_3, [0, 1, 4, 5], inplace=True)

    # Third row is pooled to the left to avoid detecting vertical stripes
    pool_3 = pool_layer([0, 1], [2, 3], "p3", reset_source=True)
    full_circuit.compose(pool_3, [4, 5, 0, 1], inplace=True)
    # qubits 4, 5 (10, 11) are now free and reusable again
    ansatz_4 = conv_layer(4, "a4", shifted_repeat=False) # top right convolutions for the top to rows
    full_circuit.compose(ansatz_4, [2, 3, 6, 7], inplace=True)

    # pooling the first row into 3 and the second to 6
    # qubits 2 and 7 are now reusable
    pool_4 = pool_layer([0, 1], [2, 3], "p4", reset_source=True)
    full_circuit.compose(pool_4, [2, 7, 3, 6], inplace=True)
    #---
    # We have 4 free qubits to encode the last row
    feature_map_12_13 = create_feature_map(num_qubits=2, param_prefix="f7")
    full_circuit.compose(feature_map_12_13, [2, 7], inplace=True)

    feature_map_14_15 = create_feature_map(num_qubits=2, param_prefix="f8")
    full_circuit.compose(feature_map_14_15, [4, 5], inplace=True)

    ansatz_5 = build_qcnn_circuit(4, 4, "a5", shift_repeat_conv=True, conv_repeats=2)
    full_circuit.compose(ansatz_5, [2, 7, 4, 5], inplace=True)

    pool_5 = pool_layer([0, 1], [2, 3], "p5") # pooled to the left
    full_circuit.compose(pool_5, [4, 5, 2, 7], inplace=True)

    ansatz_6 = conv_layer(4, "a6", shifted_repeat=False) # bottom left convolution
    full_circuit.compose(ansatz_6, [0, 1, 2, 7], inplace=True)
    
    pool_6 = pool_layer([0, 1], [2, 3], "p6") # pooling to the final diagonal to avoid vertical stripe detections
    full_circuit.compose(pool_6, [0, 7, 1, 2], inplace=True)

    # final convolutions to make all the information flow to the measured qubit
    ansatz_7 = build_qcnn_circuit(num_qubits_in=4, num_qubits_out=4, param_prefix="a7", reset_source=False, shift_repeat_conv=True) 
    full_circuit.compose(ansatz_7, [3, 6, 1, 2], inplace=True)
    ansatz_8 = conv_layer(4, "a8", shifted_repeat=False)
    full_circuit.compose(ansatz_8, [3, 1, 6, 2], inplace=True)

    # Select the qubits to measure (should be from the final diagonal)
    qubits_to_measure = [2]
    ob_list = ["I"] * num_qubits_in
    for qubit_idx in qubits_to_measure:        
        ob_list[num_qubits_in - qubit_idx - 1] = "Z"
    ob_string = "".join(ob_list)
    observable = SparsePauliOp.from_list([(ob_string, 1)])

    if return_params:
        input_params = (list(feature_map_0_1.parameters) + list(feature_map_2_3.parameters) + 
                        list(feature_map_4_5.parameters) + list(feature_map_6_7.parameters) + 
                        list(feature_map_8_9.parameters) + list(feature_map_10_11.parameters) + 
                        list(feature_map_12_13.parameters) + list(feature_map_14_15.parameters))
        weight_params = (list(ansatz_1.parameters) +
                         list(ansatz_2.parameters) + 
                         list(pool_1.parameters) + list(pool_2.parameters) + 
                         list(ansatz_3.parameters) + 
                         list(pool_3.parameters) + 
                         list(ansatz_4.parameters) + list(pool_4.parameters) + 
                         list(ansatz_5.parameters) + 
                         list(pool_5.parameters) + 
                         list(ansatz_6.parameters) + list(pool_6.parameters) + 
                         list(ansatz_7.parameters) + list(ansatz_8.parameters)
                         )
        return full_circuit, input_params, weight_params, observable
    
    return full_circuit


def circuit_4x4_16in_noReuse_horizontal(return_params=False):
    num_qubits_in = 16
    qreg = QuantumRegister(num_qubits_in, 'q')
    creg = ClassicalRegister(num_qubits_in, 'c')
    full_circuit = QuantumCircuit(qreg, creg)

    # Feature maps for the first two rows
    feature_map_0_1 = create_feature_map(num_qubits=2, param_prefix="f1") 
    full_circuit.compose(feature_map_0_1, [0, 1], inplace=True)

    feature_map_2_3 = create_feature_map(num_qubits=2, param_prefix="f2")
    full_circuit.compose(feature_map_2_3, [2, 3], inplace=True)

    feature_map_4_5 = create_feature_map(num_qubits=2, param_prefix="f3")
    full_circuit.compose(feature_map_4_5, [4, 5], inplace=True)

    feature_map_6_7 = create_feature_map(num_qubits=2, param_prefix="f4")
    full_circuit.compose(feature_map_6_7, [6, 7], inplace=True)

    #---
    # Horizontal convolution for the first two rows
    ansatz_1 = build_qcnn_circuit(4, 4, "a1", shift_repeat_conv=True, conv_repeats=2)
    full_circuit.compose(ansatz_1, [0, 1, 2, 3], inplace=True)

    ansatz_2 = build_qcnn_circuit(4, 4, "a2", shift_repeat_conv=True, conv_repeats=2)
    full_circuit.compose(ansatz_2, [4, 5, 6, 7], inplace=True)

    # Then they are pooled to the right
    pool_1 = pool_layer([0, 1], [2, 3], "p1", reset_source=False)
    full_circuit.compose(pool_1, [0, 1, 2, 3], inplace=True)

    pool_2 = pool_layer([0, 1], [2, 3], "p2", reset_source=False)
    full_circuit.compose(pool_2, [4, 5, 6, 7], inplace=True)

    # qubits 0, 1, 4, 5 are now free and reusable
    #---

    # Third row encoding and convolutions
    feature_map_8_9 = create_feature_map(num_qubits=2, param_prefix="f5")
    full_circuit.compose(feature_map_8_9, [8, 9], inplace=True)

    feature_map_10_11 = create_feature_map(num_qubits=2, param_prefix="f6")
    full_circuit.compose(feature_map_10_11, [10, 11], inplace=True)

    ansatz_3 = build_qcnn_circuit(4, 4, "a3", shift_repeat_conv=True, conv_repeats=2)
    full_circuit.compose(ansatz_3, [8, 9, 10, 11], inplace=True)

    # Third row is pooled to the left to avoid detecting vertical stripes
    pool_3 = pool_layer([0, 1], [2, 3], "p3")
    full_circuit.compose(pool_3, [10, 11, 8, 9], inplace=True)
    # qubits 4, 5 (10, 11) are now free and reusable again
    ansatz_4 = conv_layer(4, "a4", shifted_repeat=False) # top right convolutions for the top to rows
    full_circuit.compose(ansatz_4, [2, 3, 6, 7], inplace=True)

    # pooling the first row into 3 and the second to 6
    # qubits 2 and 7 are now reusable
    pool_4 = pool_layer([0, 1], [2, 3], "p4")
    full_circuit.compose(pool_4, [2, 7, 3, 6], inplace=True)
    #---
    # We have 4 free qubits to encode the last row
    feature_map_12_13 = create_feature_map(num_qubits=2, param_prefix="f7")
    full_circuit.compose(feature_map_12_13, [12, 13], inplace=True)

    feature_map_14_15 = create_feature_map(num_qubits=2, param_prefix="f8")
    full_circuit.compose(feature_map_14_15, [14, 15], inplace=True)

    ansatz_5 = build_qcnn_circuit(4, 4, "a5", shift_repeat_conv=True, conv_repeats=2)
    full_circuit.compose(ansatz_5, [12, 13, 14, 15], inplace=True)

    pool_5 = pool_layer([0, 1], [2, 3], "p5") # pooled to the left
    full_circuit.compose(pool_5, [14, 15, 12, 13], inplace=True)

    ansatz_6 = conv_layer(4, "a6", shifted_repeat=False) # bottom left convolution
    full_circuit.compose(ansatz_6, [8, 9, 12, 13], inplace=True)
    
    pool_6 = pool_layer([0, 1], [2, 3], "p6") # pooling to the final diagonal to avoid vertical stripe detections
    full_circuit.compose(pool_6, [8, 13, 9, 12], inplace=True)

    # final convolutions to make all the information flow to the measured qubit
    ansatz_7 = build_qcnn_circuit(num_qubits_in=4, num_qubits_out=4, param_prefix="a7", reset_source=False, shift_repeat_conv=True) 
    full_circuit.compose(ansatz_7, [3, 6, 9, 12], inplace=True)
    ansatz_8 = conv_layer(4, "a8", shifted_repeat=False)
    full_circuit.compose(ansatz_8, [3, 9, 6, 12], inplace=True)

    # Select the qubits to measure (should be from the final diagonal)
    qubits_to_measure = [12]
    ob_list = ["I"] * num_qubits_in
    for qubit_idx in qubits_to_measure:        
        ob_list[num_qubits_in - qubit_idx - 1] = "Z"
    ob_string = "".join(ob_list)
    observable = SparsePauliOp.from_list([(ob_string, 1)])
    print(ob_string)

    if return_params:
        input_params = (list(feature_map_0_1.parameters) + list(feature_map_2_3.parameters) + 
                        list(feature_map_4_5.parameters) + list(feature_map_6_7.parameters) + 
                        list(feature_map_8_9.parameters) + list(feature_map_10_11.parameters) + 
                        list(feature_map_12_13.parameters) + list(feature_map_14_15.parameters))
        weight_params = (list(ansatz_1.parameters) + #list(ansatz_1_2.parameters) + 
                         list(ansatz_2.parameters) + #list(ansatz_2_2.parameters) +
                         list(pool_1.parameters) + list(pool_2.parameters) + 
                         list(ansatz_3.parameters) + #list(ansatz_3_2.parameters) + 
                         list(pool_3.parameters) + 
                         list(ansatz_4.parameters) + list(pool_4.parameters) + 
                         list(ansatz_5.parameters) + #list(ansatz_5_2.parameters) + 
                         list(pool_5.parameters) + 
                         list(ansatz_6.parameters) + list(pool_6.parameters) + 
                         list(ansatz_7.parameters) + list(ansatz_8.parameters)
                         )
        return full_circuit, input_params, weight_params, observable
    
    return full_circuit


def circuit_4x4_16in_noReuse_squares(return_params=False):
    num_qubits_in = 16
    qreg = QuantumRegister(num_qubits_in, 'q')
    creg = ClassicalRegister(num_qubits_in, 'c')
    full_circuit = QuantumCircuit(qreg, creg)

    # Mapping the two first columns of pixels to the 8 qubits that we have
    feature_map_0_1 = create_feature_map(num_qubits=2, param_prefix="f1") # Pixel 0, 1 goes to qubits 0, 1
    full_circuit.compose(feature_map_0_1, [0, 1], inplace=True)

    feature_map_2_3 = create_feature_map(num_qubits=2, param_prefix="f2")
    full_circuit.compose(feature_map_2_3, [2, 3], inplace=True)

    feature_map_4_5 = create_feature_map(num_qubits=2, param_prefix="f3")
    full_circuit.compose(feature_map_4_5, [4, 5], inplace=True)

    feature_map_6_7 = create_feature_map(num_qubits=2, param_prefix="f4")
    full_circuit.compose(feature_map_6_7, [6, 7], inplace=True)

    feature_map_8_9 = create_feature_map(num_qubits=2, param_prefix="f5")
    full_circuit.compose(feature_map_8_9, [8, 9], inplace=True)

    feature_map_10_11 = create_feature_map(num_qubits=2, param_prefix="f6")
    full_circuit.compose(feature_map_10_11, [10, 11], inplace=True)

    feature_map_12_13 = create_feature_map(num_qubits=2, param_prefix="f7")
    full_circuit.compose(feature_map_12_13, [12, 13], inplace=True)

    feature_map_14_15 = create_feature_map(num_qubits=2, param_prefix="f8")
    full_circuit.compose(feature_map_14_15, [14, 15], inplace=True)


    #---
    ansatz_1 = conv_layer(4, "a1", shifted_repeat=False)
    ansatz_1_2 = conv_layer(4, "a1_2", shifted_repeat=False)
    full_circuit.compose(ansatz_1, [0, 1, 4, 5], inplace=True)
    full_circuit.compose(ansatz_1_2, [0, 4, 1, 5], inplace=True)

    ansatz_2 = conv_layer(4, "a2", shifted_repeat=False)
    ansatz_2_2 = conv_layer(4, "a2_2", shifted_repeat=False)
    full_circuit.compose(ansatz_2, [8, 9, 12, 13], inplace=True)
    full_circuit.compose(ansatz_2_2, [8, 12, 9, 13], inplace=True)

    pool_1 = pool_layer([0, 1], [2, 3], "p1", reset_source=False)
    full_circuit.compose(pool_1, [0, 1, 4, 5], inplace=True)

    pool_2 = pool_layer([0, 1], [2, 3], "p2", reset_source=False)
    full_circuit.compose(pool_2, [8, 12, 9, 13], inplace=True) # In the reuse case now 0,1,4,5 are reusable now, in this case discarded
    #---
    
    ansatz_3 = conv_layer(4, "a3", shifted_repeat=False) # this should be the conv layer that came in with reuse
    ansatz_3_2 = conv_layer(4, "a3_2", shifted_repeat=False)
    full_circuit.compose(ansatz_3, [2, 3, 6, 7], inplace=True)
    full_circuit.compose(ansatz_3_2, [2, 6, 3, 7], inplace=True)

    pool_3 = pool_layer([0, 1], [2, 3], "p3", reset_source=False) # 8, 9 which are 0, 1 in the reuse case are now reusable
    full_circuit.compose(pool_3, [2, 3, 6, 7], inplace=True)

    #---

    ansatz_4 = conv_layer(4, "a4", shifted_repeat=False) # all from reuse here
    ansatz_4_2 = conv_layer(4, "a4_2", shifted_repeat=False)
    full_circuit.compose(ansatz_4, [10, 11, 14, 15], inplace=True)
    full_circuit.compose(ansatz_4_2, [10, 14, 11, 15], inplace=True)

    pool_4 = pool_layer([0, 1], [2, 3], "p4")
    full_circuit.compose(pool_4, [10, 11, 14, 15], inplace=True)

    ansatz_5 = conv_layer(4, "a5", shifted_repeat=False)
    full_circuit.compose(ansatz_5, [8, 9, 12, 13], inplace=True)

    pool_5 = pool_layer([0, 1], [2, 3], "p5")
    full_circuit.compose(pool_5, [8, 13, 9, 12], inplace=True)

    ansatz_6 = build_qcnn_circuit(num_qubits_in=8, num_qubits_out=2, param_prefix="a6", reset_source=False, shift_repeat_conv=True)
    full_circuit.compose(ansatz_6, [4, 5, 6, 7, 12, 13, 14, 15], inplace=True)

    # Select the qubits to measure
    qubits_to_measure = [14, 15]
    ob_list = ["I"] * num_qubits_in
    for qubit_idx in qubits_to_measure:        
        ob_list[num_qubits_in - qubit_idx - 1] = "Z"
    ob_string = "".join(ob_list)
    observable = SparsePauliOp.from_list([(ob_string, 1)])
    print(ob_string)

    if return_params:
        input_params = (list(feature_map_0_1.parameters) + list(feature_map_2_3.parameters) + 
                        list(feature_map_4_5.parameters) + list(feature_map_6_7.parameters) + 
                        list(feature_map_8_9.parameters) + list(feature_map_10_11.parameters) + 
                        list(feature_map_12_13.parameters) + list(feature_map_14_15.parameters))
        weight_params = (list(ansatz_1.parameters) + list(ansatz_1_2.parameters) + list(ansatz_2.parameters) + list(ansatz_2_2.parameters) +
                         list(pool_1.parameters) + list(pool_2.parameters) + 
                         list(ansatz_3.parameters) + list(ansatz_3_2.parameters) + list(pool_3.parameters) + 
                         list(ansatz_4.parameters) + list(ansatz_4_2.parameters) + list(pool_4.parameters) +
                         list(ansatz_5.parameters) + list(pool_5.parameters) +
                         list(ansatz_6.parameters)
                        )
        return full_circuit, input_params, weight_params, observable
    
    return full_circuit


def circuit_3x4_8inV2(use_adap_pool=False, return_params=False):
    """This circuit uses 8 qubits to classify 12 inputs, with 4 qubits reused after the first pooling."""
    num_qubits_in = 8
    qreg = QuantumRegister(num_qubits_in, 'q')
    creg = ClassicalRegister(num_qubits_in, 'c')
    full_circuit = QuantumCircuit(qreg, creg)

    # Encode pixels 0-3 into qubits 0-3
    feature_map_0_3 = create_feature_map(num_qubits=4, param_prefix="f_1")
    full_circuit.compose(feature_map_0_3, [0, 1, 2, 3], inplace=True)

    # Encode pixels 8-11 into qubits 4-7
    feature_map_8_11 = create_feature_map(num_qubits=4, param_prefix="f_3")
    full_circuit.compose(feature_map_8_11, [4, 5, 6, 7], inplace=True)

    conv1upper1 = conv_layer(num_qubits=4, param_prefix="c1u1", shifted_repeat=False) # upper num1
    full_circuit.compose(conv1upper1, [0, 1, 2, 3], inplace=True)
    conv1upper2 = conv_layer(num_qubits=4, param_prefix="c1u2", shifted_repeat=False) # upper num2
    full_circuit.compose(conv1upper2, [0, 2, 1, 3], inplace=True)

    conv1bottom1 = conv_layer(num_qubits=4, param_prefix="c1b1", shifted_repeat=False) # bottom num1
    full_circuit.compose(conv1bottom1, [4, 5, 6, 7], inplace=True)
    conv1bottom2 = conv_layer(num_qubits=4, param_prefix="c1b2", shifted_repeat=False) # bottom num2
    full_circuit.compose(conv1bottom2, [4, 6, 5, 7], inplace=True)

    pool1 = pool_layer(sources=[0, 1, 2, 3], sinks=[4, 5, 6, 7], param_prefix="p1", reset_source=True, adaptive_pooling=use_adap_pool) # this is the one and only reuse here
    full_circuit.compose(pool1, [0, 1, 2, 3, 4, 5, 6, 7], inplace=True)

    # the 4 pixels coming in late with the reuse - this is the one and last reuse
    feature_map_4_7 = create_feature_map(num_qubits=4, param_prefix="f_2")
    full_circuit.compose(feature_map_4_7, [0, 1, 2, 3], inplace=True)
    
    conv2upper1 = conv_layer(num_qubits=4, param_prefix="c2u1", shifted_repeat=False) # upper num3
    full_circuit.compose(conv2upper1, [0, 1, 2, 3], inplace=True)
    conv2upper2 = conv_layer(num_qubits=4, param_prefix="c2u2", shifted_repeat=False) # upper num4
    full_circuit.compose(conv2upper2, [0, 2, 1, 3], inplace=True)

    pool2 = pool_layer(sources=[0, 1, 2, 3], sinks=[4, 5, 6, 7], param_prefix="p2", reset_source=False, adaptive_pooling=False)
    full_circuit.compose(pool2, [0, 1, 4, 5, 2, 3, 6, 7], inplace=True)

    conv3 = conv_layer(num_qubits=4, param_prefix="c3", shifted_repeat=False)
    full_circuit.compose(conv3, [2, 3, 6, 7], inplace=True)

    pool3 = pool_layer(sources=[0, 1], sinks=[2, 3], param_prefix="p3", reset_source=False)
    full_circuit.compose(pool3, [2, 3, 6, 7], inplace=True)

    conv4 = conv_layer(num_qubits=2, param_prefix="c4", shifted_repeat=False)
    full_circuit.compose(conv4, [6, 7], inplace=True)

    pool4 = pool_layer(sources=[0], sinks=[1], param_prefix="p4", reset_source=False)
    full_circuit.compose(pool4, [6, 7], inplace=True)

    observable = SparsePauliOp.from_list([("Z"*1 + "I"*7, 1)])

    if return_params:
        # Return circuit along with parameter groups for QNN construction
        input_params = (list(feature_map_0_3.parameters) + 
                       list(feature_map_4_7.parameters) + 
                       list(feature_map_8_11.parameters))
        weight_params = (list(conv1upper1.parameters) + list(conv1upper2.parameters) + 
                        list(conv1bottom1.parameters) + list(conv1bottom2.parameters)
                        + list(pool1.parameters) +
                        list(conv2upper1.parameters) + list(conv2upper2.parameters) +
                        list(pool2.parameters) + list(conv3.parameters) + list(pool3.parameters) +
                        list(conv4.parameters) + list(pool4.parameters))
        return full_circuit, input_params, weight_params, observable
    
    return full_circuit


def circuit_4x4_16inV0(return_params=False):
    num_qubits_in = 16
    shift_repeat_conv = True
    qreg = QuantumRegister(num_qubits_in, 'q')
    creg = ClassicalRegister(num_qubits_in, 'c')
    full_circuit = QuantumCircuit(qreg, creg)

    feature_map = create_feature_map(num_qubits=num_qubits_in, param_prefix="f1")
    full_circuit.compose(feature_map, inplace=True)

    ansatz = build_qcnn_circuit(num_qubits_in, 1, "a_0", False, False, shift_repeat_conv=shift_repeat_conv)
    full_circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I"*15, 1)])

    if return_params:
        input_params = list(feature_map.parameters)
        weight_params = list(ansatz.parameters)
        return full_circuit, input_params, weight_params, observable
    
    return full_circuit


def circuit_3x4_12in_2out(use_adap_pool=False, return_params=False):
    """This circuit uses 12 qubits to classify 12 inputs should be equivalent to circuit_3x4_8in_2out but without reuse."""
    num_qubits_in = 12
    qreg = QuantumRegister(num_qubits_in, 'q')
    creg = ClassicalRegister(num_qubits_in, 'c')
    full_circuit = QuantumCircuit(qreg, creg)

    # Encode pixels 0-3 into qubits 0-3
    feature_map_0_3 = create_feature_map(num_qubits=4, param_prefix="f_1")
    full_circuit.compose(feature_map_0_3, [0, 1, 2, 3], inplace=True)

    # Encode pixels 8-11 into qubits 4-7
    feature_map_8_11 = create_feature_map(num_qubits=4, param_prefix="f_3")
    full_circuit.compose(feature_map_8_11, [8, 9, 10, 11], inplace=True)

    # Build the QCNN circuit.
    ansatz_1 = build_qcnn_circuit(num_qubits_in=8, num_qubits_out=4, param_prefix="a1", reset_source=use_adap_pool, adaptive_pooling=use_adap_pool)

    full_circuit.compose(ansatz_1, [0, 1, 2, 3, 8, 9, 10, 11], inplace=True)

    # the 4 pixels coming in late with the reuse - this is the one and last reuse
    feature_map_4_7 = create_feature_map(num_qubits=4, param_prefix="f_2")
    full_circuit.compose(feature_map_4_7, [4, 5, 6, 7], inplace=True)

    ansatz_2 = build_qcnn_circuit(num_qubits_in=8, num_qubits_out=2, param_prefix="a2", reset_source=False, adaptive_pooling=False)
    full_circuit.compose(ansatz_2, [4, 5, 6, 7, 8, 9, 10, 11], inplace=True)

    observable = SparsePauliOp.from_list([("Z"*2 + "I"*10, 1)])

    if return_params:
        # Return circuit along with parameter groups for QNN construction
        input_params = (list(feature_map_0_3.parameters) + 
                       list(feature_map_4_7.parameters) + 
                       list(feature_map_8_11.parameters))
        weight_params = list(ansatz_1.parameters) + list(ansatz_2.parameters)
        return full_circuit, input_params, weight_params, observable
    
    return full_circuit


def circuit_3x4_8in_amp(use_adap_pool=False, return_params=False):
    """
    3x4 image classification using 8-qubit amplitude encoding.
    - Qubits 0-3: row-major flattened pixels (3x4=12 pixels)
    - Qubits 4-7: column-major flattened pixels
    - QCNN ansatz reduces 8→1
    - Input: 30 amplitude angles per image (15 for rows + 15 for cols)
    
    Note: Images must be preprocessed with images_to_angles_batch_dual() before passing to QNN.
    """
    num_pixels = 3 * 4  # 12 pixels
    num_qubits_in = 8
    
    # Build dual amplitude feature map (rows on q[0-3], cols on q[4-7])
    fm_circ, input_params_list, fm_info = build_dual_amp_feature_map(
        num_pixels, 
        name_rows="amp_rows_3x4",
        name_cols="amp_cols_3x4"
    )
    
    # Build QCNN ansatz: 8→1 reduction
    ansatz = build_qcnn_circuit(
        num_qubits_in=num_qubits_in,
        num_qubits_out=1,
        param_prefix="a1",
        reset_source=use_adap_pool,
        adaptive_pooling=use_adap_pool,
        ignore_last_x_adap_n_reset=2,
    )
    
    # Create full circuit
    qreg = QuantumRegister(num_qubits_in, 'q')
    creg = ClassicalRegister(num_qubits_in, 'c')
    full_circuit = QuantumCircuit(qreg, creg)
    
    # Compose: feature map then ansatz
    full_circuit.compose(fm_circ, range(num_qubits_in), inplace=True)
    full_circuit.compose(ansatz, range(num_qubits_in), inplace=True)
    
    # Observable on output qubit (q[0] after reduction)
    observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])
    
    if return_params:
        input_params = input_params_list  # List of Parameter objects from dual feature map
        weight_params = list(ansatz.parameters)
        return full_circuit, input_params, weight_params, observable
    
    return full_circuit


def circuit_4x4_8in_ampV0(use_adap_pool=False, return_params=False):
    """
    4x4 image classification using 8-qubit amplitude encoding.
    - Qubits 0-3: row-major flattened pixels (4x4=16 pixels)
    - Qubits 4-7: column-major flattened pixels
    - QCNN ansatz reduces 8→1
    - Input: 30 amplitude angles per image (15 for rows + 15 for cols)
    
    Note: Images must be preprocessed with images_to_angles_batch_dual() before passing to QNN.
    This just simple conv and pools.
    """
    num_pixels = 4 * 4  # 16 pixels
    num_qubits_in = 8
    
    # Build dual amplitude feature map (rows on q[0-3], cols on q[4-7])
    fm_circ, input_params_list, _ = build_dual_amp_feature_map(
        num_pixels, 
        name_rows="amp_rows_4x4",
        name_cols="amp_cols_4x4"
    )
    
    # Build QCNN ansatz: 8→1 reduction
    ansatz = build_qcnn_circuit(
        num_qubits_in=num_qubits_in,
        num_qubits_out=1,
        param_prefix="ans_amp",
        reset_source=use_adap_pool,
        adaptive_pooling=use_adap_pool,
        ignore_last_x_adap_n_reset=1,
    )
    
    # Create full circuit
    qreg = QuantumRegister(num_qubits_in, 'q')
    creg = ClassicalRegister(num_qubits_in, 'c')
    full_circuit = QuantumCircuit(qreg, creg)
    
    # Compose: feature map then ansatz
    full_circuit.compose(fm_circ, range(num_qubits_in), inplace=True)
    full_circuit.compose(ansatz, range(num_qubits_in), inplace=True)
    
    # Observable on output qubit (q[0] after reduction)
    observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])
    
    if return_params:
        input_params = input_params_list  # List of Parameter objects from dual feature map
        weight_params = list(ansatz.parameters)
        return full_circuit, input_params, weight_params, observable
    
    return full_circuit


def circuit_4x4_8in_amp(use_adap_pool=False, return_params=False):
    """
    4x4 classification with dual amplitude feature map (rows⊗cols on 8 qubits).
    Improvements:
      - force non-adaptive pooling for amplitude path
      - strong cross-half entangler before first pooling
      - extra conv repeats inside QCNN reducer
      - trainable readout bias on final qubit (q[7])
    """
    num_pixels = 16
    num_qubits_in = 8

    # Dual feature map (rows on q[0..3], cols on q[4..7])
    fm_circ, input_params_list, _ = build_dual_amp_feature_map(
        num_pixels,
        name_rows="amp_rows_4x4",
        name_cols="amp_cols_4x4"
    )

    # QCNN reducer (8→1). Disable adaptive pooling for amplitude runs.
    ansatz = build_qcnn_circuit(
        num_qubits_in=num_qubits_in,
        num_qubits_out=1,
        param_prefix="a1",
        reset_source=use_adap_pool,
        adaptive_pooling=use_adap_pool,          # force deterministic pooling
        shift_repeat_conv=True,
        conv_repeats=2,                         # more mixing between pools
        ignore_last_x_adap_n_reset=2,
    )

    # Readout bias on the final qubit (q[7] after reduction in your setup)
    bias = Parameter("ro_bias")

    # Compose full circuit
    qreg = QuantumRegister(num_qubits_in, "q")
    creg = ClassicalRegister(num_qubits_in, "c")
    qc = QuantumCircuit(qreg, creg)

    # 1) Feature map
    qc.compose(fm_circ, range(num_qubits_in), inplace=True)

    # 2) Strong cross-half entangler before any pooling (pair rows↔cols)
    # Interleaved CXs: (0↔4), (1↔5), (2↔6), (3↔7), then repeat reversed control
    for a, b in [(0,4), (1,5), (2,6), (3,7)]:
        qc.cx(a, b)
    for a, b in [(4,0), (5,1), (6,2), (7,3)]:
        qc.cx(a, b)

    # Optional: a light global Ry layer to kick-start mixing
    for q in range(num_qubits_in):
        qc.ry(0.11, q)

    # 3) QCNN reduction (8→1) that you already use (reduces onto q[7])
    qc.compose(ansatz, range(num_qubits_in), inplace=True)

    # 4) Readout bias on the final qubit (q[7])
    qc.ry(bias, 7)

    # Observable: Z on q[7]
    observable = SparsePauliOp.from_list([("Z" + "I"*7, 1.0)])

    if return_params:
        input_params = input_params_list
        weight_params = list(ansatz.parameters) + [bias]
        return qc, input_params, weight_params, observable

    return qc

