import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import SeismicDataset
from cae_model import ConvolutionalAutoencoder
import os

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Model & Weights
model = ConvolutionalAutoencoder().to(device)
weights_path = "outputs/encoder_weights.pth"

if not os.path.exists(weights_path):
    print(f"❌ Error: {weights_path} nahi mili. Pehle training puri karein.")
    exit()

# Load only encoder weights (since we saved only encoder in train script)
# We need to load them carefully into the full model
model.encoder.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# 3. Load Data
dataset = SeismicDataset("data/spectrograms.npy")
# Ek Earthquake sample (Label 1) aur ek Noise sample (Label 0) uthate hain
# Note: Humne labels save kiye the create_input.py mein
labels = np.load("data/labels.npy")
eq_idx = np.where(labels == 1)[0][0]

sample = dataset[eq_idx].unsqueeze(0).to(device)

# 4. Perform Reconstruction
with torch.no_grad():
    features, reconstructed = model(sample)

# 5. Plotting
original = sample.cpu().squeeze().numpy()
rebuilt = reconstructed.cpu().squeeze().numpy()

plt.figure(figsize=(12, 5), facecolor='#f0f0f0')
plt.subplot(1, 2, 1)
plt.imshow(original, aspect='auto', cmap='magma')
plt.title("Original Spectrogram (Phase 1)")
plt.xlabel("Time Bins")
plt.ylabel("Freq Bins")

plt.subplot(1, 2, 2)
plt.imshow(rebuilt, aspect='auto', cmap='magma')
plt.title("Reconstructed (Phase 2 Output)")
plt.xlabel("Time Bins")

plt.tight_layout()
plt.savefig("outputs/phase2_verification.png")
print("✅ Results saved to 'outputs/phase2_verification.png'. Ise open karke dekhiye!")
plt.show()

# 6. Feature Check
print(f"\n📊 16 Latent Features (Quantum Input) for this sample:")
print(features.cpu().numpy())