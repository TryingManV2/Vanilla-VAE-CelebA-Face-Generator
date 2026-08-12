import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.Encoder import Encoder
from src.model.Decoder import Decoder

class VAE(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: list = [32, 64, 128, 256, 512],
        img_size: int = 128
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = Encoder(input_channels, latent_dim, hidden_dims, img_size)

        dummy_input = torch.zeros(1, input_channels, img_size, img_size)
        with torch.no_grad():

            conv_output = self.encoder.encoder(dummy_input)
            _, channels, height, width = conv_output.shape
            spatial_shape = (channels, height, width)
            print(f"Dynamically detected encoder output shape: {spatial_shape}")

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_channels=input_channels,
            spatial_shape=spatial_shape
        )

    @staticmethod
    def reparameterization_trick(
        mu,
        log_var
    ):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + (std * eps)

    def forward(
        self,
        x
    ):
        mu, log_var = self.encoder.encode(x)

        if self.training:
            z = self.reparameterization_trick(mu, log_var)
        else:
            z = mu

        x_recon = self.decoder(z)

        return x_recon, mu, log_var

    def loss(self, recon_x, x, mu, log_var, beta=1.0):
        """
        Args:
            recon_x: reconstructed images (output of decoder)
            x: original input images
            mu, log_var: latent statistics from encoder
            beta: weight for KL divergence (β-VAE).
                  For 128x128 images, start with beta=0.0005.

        Returns:
            total_loss: scalar tensor (recon + beta * kl)
            recon_loss: per-pixel MSE (mean over pixels and batch) – scale ~0.0–1.0
            kl_loss: KL divergence (summed over latent dims, averaged over batch) – scale ~0–1000
        """

        # This gives an average error per pixel in the [-1,1] range.
        # Typical values: start ~0.4, end ~0.02–0.05.
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        # Sum over latent dimensions, then average over batch.
        # Typical values: start ~1000, end ~50–200.
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kl_loss = kl_loss.mean()  # scalar

        # β controls the trade-off. For 128x128 images with per‑pixel MSE,
        # beta=0.0005 is a great starting point.
        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def generate(
        self,
        num_samples,
        device
    ):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            return self.decoder(z)