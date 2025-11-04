# QCNN with Mid-Circuit Measurements & Qubit Reuse & Feedforward

This github repository contains the public code of my "Towards Quantum Convolutional Netural Networks" scientific students' association report.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebook QCNN:
   ```
   Executable_files/QCNN.ipynb
   ```

The notebook demonstrates training a QCNN on a simple stripe classification task (horizontal vs vertical patterns). By default, it loads a pretrained model for quick demonstration - set `do_train = True` if you wish to train from scratch.

3. (Optional) Run inference on real quantum hardware

   Note that this required and IBM api key saved as an environment variable.
   ```
   Executable_files/inference_on_real_backend.ipynb
   ```
   
   This notebook uses the SamplerQNN to perform inference with a pretrained model. It supports both simulators and real IBM Quantum backends

## Folder structure

- **`src/`** - Core implementations (circuit building, training callbacks, utilities)
- **`model_weights/`** - Pretrained model and training plots - for demonstration
- **`Executable_files/`** - Main notebook