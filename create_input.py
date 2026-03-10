import numpy as np
import os

os.makedirs("data", exist_ok=True)

num_samples = 1200
signal_length = 2000
t = np.linspace(0, 1, signal_length)

seismograms = []
labels = []

print("Generating signals with actual patterns...")
for i in range(num_samples):
    is_earthquake = np.random.choice([0, 1])
    # Base Noise
    signal = np.random.normal(0, 0.1, signal_length)
    
    if is_earthquake:
        # Add a 'P-wave' signature (localized high-freq burst)
        start = np.random.randint(400, 1200)
        # Ricker-like wavelet for structure
        burst = np.sin(2 * np.pi * 15 * t[start:start+200]) * np.exp(-np.linspace(-3, 3, 200)**2)
        signal[start:start+200] += burst * 2.0
        labels.append(1)
    else:
        labels.append(0)
    
    seismograms.append(signal)

np.save("data/seismograms.npy", np.array(seismograms).astype(np.float32))
np.save("data/labels.npy", np.array(labels))
np.save("data/p_arrivals.npy", np.zeros(num_samples)) # Dummy arrivals

print("✅ Success: Structured data created! Ab Phase 1 and 2 run karein.")