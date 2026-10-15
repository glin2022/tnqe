# TNQE: Tensor-Network-based Quantum Data Encoding

Code for **"Structured Unitary Tensor Network Representations for Circuit-Efficient Quantum Data Encoding"** (NeurIPS 2026).

TNQE represents a classical image as a quantized tensor train (QTT) and compiles the tensor cores into a shallow encoding circuit. Three variants are provided:

| Variant | Idea | 32x32 MNIST circuit (qubits, depth) |
|---|---|---|
| **TNQE-full** | right-canonical MPS cores -> isometries -> local unitaries, sequential circuit | (12, 945) |
| **TNQE-core** | every core is amplitude-encoded on its own register, no inter-core entanglement | (26, 116) |
| **TNQE-unitary** | cores parameterised directly as learnable block unitaries (RY/RZ + CNOT), trained with a KL objective | (13, 81) |

Amplitude encoding of the same image needs (10, 2028).

## Requirements

Python 3.10, PyTorch, torchvision, NumPy, Qiskit, qiskit-aer, matplotlib, Pillow (qiskit-ibm-runtime only for the hardware runs). Tested with PyTorch 2.0, Qiskit 2.2.3 and qiskit-aer 0.17.2; MNIST / CIFAR-10 are downloaded automatically.

## Quick start

```python
from utils.data_utils import load_mnist32
from utils.qtt import train_qtt
from utils.unitary_qtt import train_unitary
from full_encoding import build_full_circuit, decode_full_statevector
from core_encoding import cores_to_chain_mats, build_core_circuit, decode_core_statevector
from unitary_encoding import build_unitary_circuit, decode_unitary_statevector
from utils.circuit_stats import circuit_stats

img = load_mnist32(1)[0][0]                       # 32x32 image in [0, 1]

# TNQE-full / TNQE-core share the classical QTT fit (rank 4)
model, recon = train_qtt(img, rank=4)
qc_full = build_full_circuit(model.cores_numpy())
qc_core = build_core_circuit(cores_to_chain_mats(model.cores_numpy()))[0]

# TNQE-unitary: learn block unitaries (rank 8, 4 layers) and emit the circuit directly
umodel, urecon = train_unitary(img, rank=8, n_layers=4)
qc_uni = build_unitary_circuit(umodel.theta_numpy(), L=5, rank=8, n_layers=4)

for name, qc in [("full", qc_full), ("core", qc_core), ("unitary", qc_uni)]:
    print(name, circuit_stats(qc))

img_full = decode_full_statevector(qc_full, L=5)
img_core = decode_core_statevector(cores_to_chain_mats(model.cores_numpy()))
img_uni = decode_unitary_statevector(qc_uni, L=5) * img.sum()
```

## Experiments

```bash
# encode the first 1000 MNIST test images
python scripts/run_amplitude.py --n 1000
python scripts/run_automatic.py --mnisq_root /path/to/base_test_mnist_784_f90 --n 1000
python scripts/run_tnqe_full.py --n 1000
python scripts/run_tnqe_core.py --n 1000
python scripts/run_tnqe_unitary.py --n 1000
```

| Script | Experiment |
|---|---|
| `run_tnqe_{full,core,unitary}.py` | main results, rank / layer ablations (`--rank`, `--layers`, `--tag`), CIFAR-10 (`--dataset cifar`) |
| `run_amplitude.py`, `run_automatic.py` | baselines (automatic encoding uses the pre-compiled circuits of the MNISQ dataset) |
| `eval_metrics.py`, `eval_classification.py` | quantitative evaluation (MSE / BCE / PSNR / SSIM, classifier accuracy) |
| `circuit_resources.py` | qubits / depth / gate counts, rank and layer sweeps, scalability curves |
| `run_hardware.py`, `run_gaussian_blobs.py` | IBM Quantum experiments (run `QiskitRuntimeService.save_account` once beforehand) |
| `run_pca_ae.py`, `timing.py` | PCA baseline, runtime |

## Citation

```bibtex
@article{lin2026structured,
  title={Structured Unitary Tensor Network Representations for Circuit-Efficient Quantum Data Encoding},
  author={Lin, Guang and Tanaka, Toshihisa and Zhao, Qibin},
  journal={arXiv preprint arXiv:2602.16266},
  year={2026}
}
```
