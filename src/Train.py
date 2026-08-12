import os
import yaml
import inspect
from typing import Optional

import torch
import torch.optim as optim
from torchvision.utils import save_image

def train(
    dataset:str = "flwrlabs/celeba",
    cache_dir: Optional[str] = "./dataset",
    n_epochs: int = 100,
    img_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    test_set: bool = False, # We do not need to test_loader in training
    checkpoint_dir: str = "./checkpoints",
    sample_dir: str = "./outputs/samples",
    input_channels: int = 3,
    latent_dim: int = 128,
    hidden_dims: list = [32, 64, 128, 256, 512],
    learning_rate: float = 1e-3,
    beta: float = 0.0005,
):

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.mps.is_available() else \
             "cpu"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    from src.model.VAE import VAE
    model = VAE(
        input_channels=input_channels,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        img_size=img_size
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    def get_beta(epoch, max_annealing_epochs=20, target_beta=beta):
        if epoch <= max_annealing_epochs:
            return target_beta * (epoch / max_annealing_epochs)
        return target_beta

    from src.preprocess.Preprocess import dataloaders
    train_loader, val_loader, test_loader = dataloaders(
        dataset=dataset,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        test_set=test_set,
        cache_dir=cache_dir
    )

    def train_epoch(epoch):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        current_beta = get_beta(epoch)  # compute once per epoch

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_x, mu, log_var = model(data)
            loss, recon_loss, kl_loss = model.loss(recon_x, data, mu, log_var, beta=current_beta)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch:03d} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}  Recon: {recon_loss.item():.4f}  KL: {kl_loss.item():.4f}")

        num_batches = len(train_loader)
        return total_loss / num_batches, total_recon / num_batches, total_kl / num_batches

    def validate_epoch(epoch):
        model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_loader):
                data = data.to(device)
                recon_x, mu, log_var = model(data)
                loss, recon_loss, kl_loss = model.loss(recon_x, data, mu, log_var, beta=beta)  # use fixed beta for eval

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

        num_batches = len(val_loader)
        return total_loss / num_batches, total_recon / num_batches, total_kl / num_batches

    def save_reconstructions(epoch):
        """Save a grid of original vs reconstructed images for visual inspection."""
        model.eval()
        with torch.no_grad():
            data, _ = next(iter(val_loader))
            data = data.to(device)
            recon, _, _ = model(data)

            n = min(8, data.size(0))
            comparison = torch.cat([data[:n], recon[:n]])
            save_image(comparison, f"{sample_dir}/recon_epoch_{epoch:03d}.png",
                    nrow=n, normalize=True)

    best_val_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        print(f"\n===== Epoch {epoch:03d}/{n_epochs} =====")

        train_loss, train_recon, train_kl = train_epoch(epoch)
        print(f"Train  -> Loss: {train_loss:.4f}  Recon: {train_recon:.4f}  KL: {train_kl:.4f}")

        val_loss, val_recon, val_kl = validate_epoch(epoch)
        print(f"Valid  -> Loss: {val_loss:.4f}  Recon: {val_recon:.4f}  KL: {val_kl:.4f}")

        save_reconstructions(epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/best_model.pt")
            print(f"✓ New best model saved with val_loss = {val_loss:.4f}")

    print("\n===== Generating new images from prior =====")
    model.eval()
    with torch.no_grad():
        generated = model.generate(num_samples=16, device=device)
        save_image(generated, f"{sample_dir}/generated_final.png", nrow=4, normalize=True)
        print(f"Generated images saved to {sample_dir}/generated_final.png")


def train_from_yaml(
    yaml_path: str = "./configs/standard_config.yaml"
    ) -> None:
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get the parameter names of the train() function
    sig = inspect.signature(train)
    valid_params = set(sig.parameters.keys())

    # Filter out keys that are not accepted by train()
    filtered_config = {k: v for k, v in config.items() if k in valid_params}

    train(**filtered_config)