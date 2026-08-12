import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: list = [32, 64, 128, 256, 512],
        img_size: int = 128,
        cond_dim: int = 0
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        modules = []
        in_channels = input_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        dummy_input = torch.zeros(1, input_channels, img_size, img_size)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
            self.flatten_dim = dummy_output.view(1, -1).size(1)

        self.fc_mu = nn.Linear(self.flatten_dim + cond_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim + cond_dim, latent_dim)

        nn.init.constant_(self.fc_log_var.bias, 0.0)
        nn.init.normal_(self.fc_log_var.weight, mean=0.0, std=0.01)

        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.constant_(self.fc_mu.bias, 0.0)

    @staticmethod
    def reparameterization_trick(mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + (std * eps)

    def encode(self, x, c=None):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        if c is not None:
            h = torch.cat([h, c], dim=1)

        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def forward(self, x, c=None):
        mu, log_var = self.encode(x, c)
        z = self.reparameterization_trick(mu, log_var)
        return z, mu, log_var