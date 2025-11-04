"""
Utilities for amplitude (amp) encoding feature maps.

This module provides helpers to:
- Convert 2D images to 1D vectors
- Normalize and (optionally) pad vectors to a power-of-two length
- Build a Qiskit circuit that prepares a quantum state with the desired amplitudes
- Simulate the resulting statevector for validation

Design notes:
- We use qiskit.circuit.library.StatePreparation to create a statepreparation circuit.
- If the input length is not a power of two, we zero-pad to the next power-of-two.
- By default, we keep real-valued amplitudes (typical for grayscale images), but complex inputs are also accepted.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Optional, Dict, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RYGate
try:
	# UCRY is available in newer Qiskit Terra versions. Fallback below if missing.
	from qiskit.circuit.library import UCRY  # type: ignore
	_HAS_UCRY = True
except Exception:  # pragma: no cover - environment-dependent
	UCRY = None  # type: ignore
	_HAS_UCRY = False


def _next_power_of_two(n: int) -> int:
	"""Return the smallest power of two that is >= n."""
	if n <= 1:
		return 1
	return 1 << (n - 1).bit_length()


def flatten_image_to_vector(img: np.ndarray) -> np.ndarray:
	"""Flatten a 2D image to a 1D numpy vector of shape (n*m,).

	Accepts any numeric dtype. The returned vector is a copy (safe to mutate).
	"""
	if img.ndim != 2:
		raise ValueError(f"Expected a 2D image array, got shape {img.shape}")
	return np.asarray(img, dtype=np.float64).reshape(-1).copy()


def normalize_vector(vec: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
	"""Normalize a 1D vector to unit L2 norm.

	Returns (normalized_vec, norm_before).
	If the norm is below eps, raises a ValueError to prevent division by ~0.
	"""
	v = np.asarray(vec, dtype=np.complex128)
	norm = np.linalg.norm(v)
	if norm < eps:
		raise ValueError("Cannot normalize: vector has ~zero norm.")
	return (v / norm, float(norm))


## NOTE: pad_to_power_of_two removed (unused). Use _next_power_of_two with explicit padding where needed.


def amplitude_feature_map(
	data: Iterable[float | complex],
	num_qubits: Optional[int] = None,
	*,
	normalize: bool = True,
	zero_pad: bool = True,
	name: str = "amp_encode",
) -> Tuple[QuantumCircuit, Dict[str, Any]]:
	"""Create a feature map circuit that prepares amplitudes equal to the input data.

	Parameters
	- data: Iterable of values whose relative magnitudes determine the target amplitudes.
	- num_qubits: Optionally force the number of qubits. If None, pick minimal s.t. 2^n >= len(data).
	- normalize: If True, normalize the input vector to unit length.
	- zero_pad: If True, zero-pad to the next power-of-two if needed (required by state prep).
	- name: Name prefix for the circuit.

	Returns
	- (circuit, info): A QuantumCircuit that prepares the desired state from |0...0>, and
	  an info dict with helpful metadata such as target_length, padded, norm_before.
	"""
	vec = np.asarray(list(data), dtype=np.complex128)

	norm_before: Optional[float] = None
	if normalize:
		vec, norm_before = normalize_vector(vec)

	# Resolve target length and optional padding
	original_len = len(vec)
	if num_qubits is None:
		target_len = _next_power_of_two(original_len)
	else:
		target_len = 1 << int(num_qubits)

	if len(vec) != target_len:
		if not zero_pad:
			raise ValueError(
				f"Input length {len(vec)} doesn't match 2^num_qubits={target_len} and zero_pad=False."
			)
		padded = True
		padded_vec = np.zeros(target_len, dtype=np.complex128)
		padded_vec[: len(vec)] = vec
		vec = padded_vec
	else:
		padded = False

	# Determine qubit count
	if num_qubits is None:
		n_qubits = int(np.log2(len(vec)))
	else:
		n_qubits = int(num_qubits)

	# Build the state preparation circuit
	qr = QuantumRegister(n_qubits, name="q")
	qc = QuantumCircuit(qr, name=name)
	# StatePreparation follows little-endian convention consistent with Qiskit
	sp = StatePreparation(vec)
	qc.compose(sp, qr, inplace=True)

	info: Dict[str, Any] = {
		"original_length": original_len,
		"target_length": len(vec),
		"num_qubits": n_qubits,
		"padded": padded,
		"norm_before": norm_before,
		"normalize": normalize,
		"zero_pad": zero_pad,
		"name": name,
	}
	return qc, info


def simulate_statevector(circuit: QuantumCircuit) -> np.ndarray:
	"""Return the statevector of the circuit using exact simulation.

	Uses qiskit.quantum_info.Statevector to avoid backend dependencies.
	"""
	sv = Statevector.from_instruction(circuit)
	return np.asarray(sv.data)


def verify_encoding(
	target: Iterable[float | complex],
	prepared: Iterable[float | complex],
) -> Tuple[float, float]:
	"""Compare target and prepared statevectors.

	Returns (fidelity, l2_error).
	- Fidelity is |<target|prepared>|^2 after normalizing both.
	- l2_error is the L2 norm of the difference between normalized vectors.
	"""
	t = np.asarray(list(target), dtype=np.complex128)
	p = np.asarray(list(prepared), dtype=np.complex128)

	tn, _ = normalize_vector(t)
	pn, _ = normalize_vector(p)

	fidelity = float(np.abs(np.vdot(tn, pn)) ** 2)
	l2_error = float(np.linalg.norm(tn - pn))
	return fidelity, l2_error


__all__ = [
	"flatten_image_to_vector",
	"normalize_vector",
	"amplitude_feature_map",
	"simulate_statevector",
	"verify_encoding",
]


# -----------------
# Parametric encoder
# -----------------

def angles_from_amplitudes_real(vec: np.ndarray) -> np.ndarray:
	"""Compute Möttönen-style real-amplitude angles for n-qubit encoder.

	Expects vec length to be a power of two (2^n). If not normalized, will
	normalize internally. Returns a flat array of length 2^n - 1 with the
	angles ordered by layers k=0..n-1 and blocks b=0..2^{n-1-k}-1 as:
		theta = [theta(k=0,b=0..), theta(k=1,b=0..), ..., theta(k=n-1,b=0)]

	Conventions:
	- Qubit indexing is little-endian: q[0] is LSB, q[n-1] is MSB.
	- At layer k, the target is q[k], controls are q[n-1]..q[k+1] (MSB->LSB),
	  and there are 2^{n-1-k} blocks/angles.
	- For each block, theta = 2*atan2(beta, alpha) with alpha/beta computed
	  from squared magnitudes of the half-blocks split by bit k.
	"""
	v = np.asarray(vec, dtype=np.complex128)
	n_total = v.size
	# Validate power-of-two length
	if n_total == 0 or (n_total & (n_total - 1)) != 0:
		raise ValueError("Input length must be a positive power of two for parametric encoder.")

	# Normalize
	v, _ = normalize_vector(v)

	n_qubits = int(np.log2(n_total))
	angles: list[float] = []
	# Layer k targets qubit k (from LSB upwards), controls are higher bits
	for k in range(n_qubits):
		block_size = 1 << (k + 1)  # 2^{k+1}
		half = 1 << k              # 2^k
		num_blocks = 1 << (n_qubits - 1 - k)
		for b in range(num_blocks):
			start = b * block_size
			left = v[start : start + half]
			right = v[start + half : start + block_size]
			a2 = float(np.sum(np.abs(left) ** 2))
			b2 = float(np.sum(np.abs(right) ** 2))
			if a2 <= 0 and b2 <= 0:
				theta = 0.0
			else:
				alpha = np.sqrt(a2)
				beta = np.sqrt(b2)
				theta = float(2.0 * np.arctan2(beta, alpha))
			angles.append(theta)
	return np.asarray(angles, dtype=np.float64)


def build_parametric_amp_encoder(n_qubits: int, name: str = "param_amp") -> tuple[QuantumCircuit, ParameterVector, dict]:
	"""Build a fixed parametric circuit that prepares real amplitudes via Ry trees.

	The circuit structure is fixed; data enters only via ParameterVector angles
	of length (2^n - 1). Gate layout follows the same ordering as
	angles_from_amplitudes_real:
		for k in 0..n-1 (target = q[k], controls = q[n-1]..q[k+1]):
			for b in 0..2^{n-1-k}-1:
				apply multi-controlled RY(theta[idx]) with ctrl_state = b

	Returns (qc, theta, info) where info contains per-layer index ranges.
	"""
	if n_qubits <= 0:
		raise ValueError("n_qubits must be positive")

	total_params = (1 << n_qubits) - 1
	theta = ParameterVector("theta", total_params)
	qc = QuantumCircuit(n_qubits, name=name)

	# First compute layer slices for k = 0..n-1 (LSB->MSB) to define parameter indexing
	layer_slices: list[tuple[int, int]] = []
	idx = 0
	for k in range(n_qubits):
		num_blocks = 1 << (n_qubits - 1 - k)
		layer_start = idx
		layer_end = idx + num_blocks
		layer_slices.append((layer_start, layer_end))
		idx = layer_end

	# Apply layers in MSB->LSB order using UCRY for robustness. For layer t,
	# controls are ascending [t+1, ..., n-1]. Angles within the layer are
	# ordered by integer b=0..2^{m}-1 where bit j (0-based) refers to control
	# qubit (t+1+j).
	for t in range(n_qubits - 1, -1, -1):
		controls = list(range(t + 1, n_qubits))  # ascending: [t+1, ..., n-1]
		target = t
		layer_start, layer_end = layer_slices[t]
		layer_params = [theta[i] for i in range(layer_start, layer_end)]

		if len(controls) == 0:
			# No controls: single RY on target
			qc.ry(layer_params[0], target)
		else:
			if _HAS_UCRY:
				ucry = UCRY(layer_params)
				qc.append(ucry, qargs=controls + [target])
			else:
				# Manual fallback: emulate UCRY by iterating control states in ascending order
				num_ctrls = len(controls)
				for b in range(1 << num_ctrls):
					ctrl_state = format(b, f"0{num_ctrls}b")
					gate = RYGate(layer_params[b]).control(num_ctrls, ctrl_state=ctrl_state)
					qc.append(gate, qargs=controls + [target])

	info = {
		"num_qubits": n_qubits,
		"total_params": total_params,
		"layer_slices": layer_slices,  # list of (start, end) for each layer k
		"param_order": "k ascending, block b ascending per layer",
		"qubit_convention": "q[0]=LSB (fastest), q[n-1]=MSB (slowest)",
	}
	return qc, theta, info


def bind_encoder_angles(qc: QuantumCircuit, theta: ParameterVector, angles: Iterable[float]) -> QuantumCircuit:
	"""Return a bound circuit by assigning angles into theta in order."""
	angles = list(angles)
	if len(theta) != len(angles):
		raise ValueError(f"Angle size mismatch: expected {len(theta)} got {len(angles)}")
	mapping = {theta[i]: float(angles[i]) for i in range(len(theta))}
	return qc.assign_parameters(mapping, inplace=False)


__all__ += [
	"angles_from_amplitudes_real",
	"build_parametric_amp_encoder",
	"bind_encoder_angles",
]


# -----------------------------
# Feature map API for EstimatorQNN
# -----------------------------

def build_amp_feature_map(
	num_pixels: int,
	*,
	allow_padding: bool = True,
	name: str = "amp_feature_map",
) -> tuple[QuantumCircuit, ParameterVector, dict]:
	"""Create a fixed, parameterized amplitude feature map for EstimatorQNN.

	Builds an n-qubit encoder circuit with (2^n - 1) input parameters where
	2^n >= num_pixels. If num_pixels is not a power of two and allow_padding is
	True, zeros are appended during preprocessing.

	Returns (feature_map_circuit, input_params, info_dict).
	info_dict contains:
	  - num_qubits
	  - num_pixels
	  - target_length (2^n)
	  - total_params (2^n - 1)
	  - layer_slices
	"""
	if num_pixels <= 0:
		raise ValueError("num_pixels must be positive")

	target_len = _next_power_of_two(num_pixels)
	n_qubits = int(np.log2(target_len))
	if target_len != num_pixels and not allow_padding:
		raise ValueError(
			f"num_pixels={num_pixels} is not a power of two and allow_padding=False"
		)

	fm_circ, theta, enc_info = build_parametric_amp_encoder(n_qubits, name=name)
	info = {
		"num_qubits": n_qubits,
		"num_pixels": num_pixels,
		"target_length": target_len,
		"total_params": enc_info["total_params"],
		"layer_slices": enc_info["layer_slices"],
	}
	return fm_circ, theta, info


def image_to_angles(
	img: np.ndarray,
	*,
	target_length: Optional[int] = None,
) -> np.ndarray:
	"""Convert a 2D image to encoder angles.

	- Flattens the image to 1D
	- Pads zeros up to target_length if provided and larger than current length
	- Computes real-amplitude angles for the parametric encoder
	"""
	v = flatten_image_to_vector(img)
	if target_length is not None and target_length > v.size:
		padded = np.zeros(target_length, dtype=np.float64)
		padded[: v.size] = v
		v = padded
	# angles_from_amplitudes_real normalizes internally
	return angles_from_amplitudes_real(v)


def images_to_angles_batch(
	images: Iterable[np.ndarray],
	*,
	target_length: Optional[int] = None,
) -> np.ndarray:
	"""Batch conversion of images to angle matrix of shape (batch, P).

	P is (2^n - 1) where n = log2(target_length). If target_length is None,
	it is inferred from the first image length by rounding up to power-of-two.
	"""
	images = list(images)
	if not images:
		return np.zeros((0, 0), dtype=np.float64)

	first_vec = flatten_image_to_vector(images[0])
	if target_length is None:
		target_length = _next_power_of_two(first_vec.size)

	angles_list = [image_to_angles(img, target_length=target_length) for img in images]
	return np.stack(angles_list, axis=0)


__all__ += [
	"build_amp_feature_map",
	"image_to_angles",
	"images_to_angles_batch",
]


# -----------------------------
# Dual (rows + columns) encoders on 8 qubits
# -----------------------------

def flatten_image_to_vector_col_major(img: np.ndarray) -> np.ndarray:
	"""Flatten a 2D image in column-major (Fortran) order to a 1D vector.

	Sequence is column by column: (0,0),(1,0),...,(n-1,0),(0,1),... .
	"""
	if img.ndim != 2:
		raise ValueError(f"Expected a 2D image array, got shape {img.shape}")
	return np.asarray(img, dtype=np.float64).flatten(order='F').copy()


def image_to_angles_col_major(
	img: np.ndarray,
	*,
	target_length: Optional[int] = None,
) -> np.ndarray:
	"""Column-major version of image_to_angles: flattens with order='F' then computes angles.

	Pads to target_length if provided and larger than current length.
	"""
	v = flatten_image_to_vector_col_major(img)
	if target_length is not None and target_length > v.size:
		padded = np.zeros(target_length, dtype=np.float64)
		padded[: v.size] = v
		v = padded
	return angles_from_amplitudes_real(v)


def images_to_angles_batch_dual(
	images: Iterable[np.ndarray],
	*,
	target_length: Optional[int] = None,
) -> np.ndarray:
	"""Batch conversion building concatenated angles: [row-major angles || col-major angles].

	For 4x4, each side yields 2^4-1 = 15 angles; concat gives 30 per sample.
	"""
	images = list(images)
	if not images:
		return np.zeros((0, 0), dtype=np.float64)

	first_vec = flatten_image_to_vector(images[0])
	if target_length is None:
		target_length = _next_power_of_two(first_vec.size)

	rows_list = [image_to_angles(img, target_length=target_length) for img in images]
	cols_list = [image_to_angles_col_major(img, target_length=target_length) for img in images]
	concat_list = [np.concatenate([r, c], axis=0) for r, c in zip(rows_list, cols_list)]
	return np.stack(concat_list, axis=0)


def build_dual_amp_feature_map(
	num_pixels: int,
	*,
	name_rows: str = "amp_rows",
	name_cols: str = "amp_cols",
) -> tuple[QuantumCircuit, list, dict]:
	"""Create an 8-qubit feature map: first 4 qubits encode rows, next 4 encode columns.

	Returns (circuit, input_params_list, info) where input_params_list is
	[theta_rows..., theta_cols...] length 30 for 4x4 (two sets of 15).
	"""
	if num_pixels <= 0:
		raise ValueError("num_pixels must be positive")

	target_len = _next_power_of_two(num_pixels)
	n_each = int(np.log2(target_len))
	if n_each != 4:
		# Still support arbitrary power-of-two sizes; total qubits = 2*n_each
		pass

	circ_rows, theta_rows, info_rows = build_parametric_amp_encoder(n_each, name=name_rows)
	circ_cols, theta_cols, info_cols = build_parametric_amp_encoder(n_each, name=name_cols)

	# Rename parameters of the second (columns) encoder to avoid name conflicts
	# when composing into a single circuit. The base encoder uses ParameterVector('theta', ...),
	# so we remap the second set to a distinct vector, e.g., 'phi'.
	from qiskit.circuit import ParameterVector as _ParameterVector  # local alias to avoid confusion
	phi_cols = _ParameterVector("phi", len(theta_cols))
	remap_cols = {theta_cols[i]: phi_cols[i] for i in range(len(theta_cols))}
	circ_cols = circ_cols.assign_parameters(remap_cols, inplace=False)

	total_qubits = 2 * n_each
	qc = QuantumCircuit(total_qubits, name=f"dual_{n_each+n_each}")

	# Place rows on qubits [0..n_each-1], cols on qubits [n_each..2*n_each-1]
	qc.compose(circ_rows, list(range(0, n_each)), inplace=True)
	qc.compose(circ_cols, list(range(n_each, 2 * n_each)), inplace=True)

	# Expose input parameter list in the composed-circuit order: rows then cols (renamed)
	params_list = list(theta_rows) + list(phi_cols)
	info = {
		"num_qubits_each": n_each,
		"total_qubits": total_qubits,
		"num_pixels": num_pixels,
		"target_length_each": target_len,
		"params_per_half": len(theta_rows),
		"total_input_params": len(params_list),
		"rows_layer_slices": info_rows["layer_slices"],
		"cols_layer_slices": info_cols["layer_slices"],
		"param_names": {
			"rows": "theta",
			"cols": "phi",
		},
	}
	return qc, params_list, info


__all__ += [
	"flatten_image_to_vector_col_major",
	"image_to_angles_col_major",
	"images_to_angles_batch_dual",
	"build_dual_amp_feature_map",
]

