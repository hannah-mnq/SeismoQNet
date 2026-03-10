import torch
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler  # V2 Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from cae_model import ConvolutionalAutoencoder

# 1. Load Data & Encoder (Proof of concept ke liye 20 samples best hain)
data = np.load("data/spectrograms.npy")[:20] 
labels = np.load("data/labels.npy")[:20]

encoder = ConvolutionalAutoencoder().encoder
encoder.load_state_dict(torch.load("outputs/encoder_weights.pth", map_location='cpu'))
encoder.eval()

# 2. Extract 16 Features
input_tensors = torch.from_numpy(data).float().unsqueeze(1)
with torch.no_grad():
    features = encoder(input_tensors).numpy()

# 3. Quantum Setup (The Correct V2 Way)
# 16 Features -> 16 Qubits Feature Map
feature_map = ZZFeatureMap(feature_dimension=16, reps=1)

# Step A: Sampler banayein (V2 primitive)
sampler = StatevectorSampler()

# Step B: Fidelity object banayein jo sampler use karega
fidelity = ComputeUncompute(sampler=sampler)

# Step C: Kernel banayein (Ab yahan 'sampler' ki jagah 'fidelity' jayega)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

# Step D: QSVC Model
qsvc = QSVC(quantum_kernel=quantum_kernel)

print("⚛️ Quantum Training Shuru... (16 Qubits simulation is heavy, please wait)")
qsvc.fit(features, labels)

# 4. Results check
accuracy = qsvc.score(features, labels)
print(f"\n✨ FINAL QUANTUM RESULT ✨")
print(f"✅ Accuracy: {accuracy * 100:.2f}%")