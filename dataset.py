import torch
import numpy as np
from torch.utils.data import Dataset

class SeismicDataset(Dataset):
    def __init__(self, npy_path="data/spectrograms.npy"):
        # Hum Phase 1 ka output load kar rahe hain
        data = np.load(npy_path) 
        # Tensors mein badalna zaroori hai deep learning ke liye
        # unsqueeze(1) channel dimension add karta hai (Samples, 1, Freq, Time)
        self.data = torch.from_numpy(data).float().unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]