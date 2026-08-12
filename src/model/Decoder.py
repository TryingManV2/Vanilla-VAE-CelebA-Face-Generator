import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list,
        output_channels: int,
        spatial_shape: tuple
    ):
        super().__init__()

        channels, height, width = spatial_shape
        self.hidden_dim = channels
        self.spatial_height = height
        self.spatial_width = width
        self.flatten_dim = channels * height * width

        self.fc = nn.Linear(latent_dim, self.flatten_dim)

        modules = []
        in_channels = self.hidden_dim
        for h_dim in reversed(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels, h_dim,
                        kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
            )
            in_channels = h_dim

        """
        We use Tanh() because images are normalized in range of [-1,+1]
        If we use Sigmoid(), we should normalize in range of [0,+1]
        """
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.hidden_dim, self.spatial_height, self.spatial_width)
        return self.decoder(h)