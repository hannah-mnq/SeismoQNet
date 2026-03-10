import torch
from torch.utils.data import DataLoader
from dataset import SeismicDataset
from cae_model import ConvolutionalAutoencoder
import matplotlib.pyplot as plt

# 1. Setup
dataset = SeismicDataset("data/spectrograms.npy") # Rasta check kar lena
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = ConvolutionalAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

# 2. Training Loop
print("🚀 Training Phase 2: Feature Extraction...")
for epoch in range(300):
    for batch in loader:
        optimizer.zero_grad()
        features, reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# 3. Save weights for Phase 3
torch.save(model.encoder.state_dict(), "outputs/encoder_weights.pth")
print("✅ Done! Encoder weights saved in 'outputs/'.")