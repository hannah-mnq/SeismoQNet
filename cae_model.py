import torch.nn as nn

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        # ENCODER: Image -> 128 -> 16
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # (16, 32, 100)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # (32, 16, 50)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 50, 128), # Intermediate Layer
            nn.ReLU(),
            nn.Linear(128, 16)            # Final 16 Features
        )
        # DECODER: 16 -> 128 -> Image
        self.decoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * 16 * 50),
            nn.Unflatten(1, (32, 16, 50)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=(1, 1)),
            nn.Sigmoid() 
        )

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return features, reconstruction