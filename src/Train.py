import os
import yaml
import inspect
from typing import Optional
from collections import deque

import torch
import torch.optim as optim
from torchvision.utils import save_image


from src.model.VAE import VAE
from src.preprocess.Preprocess import dataloaders

def train(
    dataset: str = "flwrlabs/celeba",
    cache_dir: Optional[str] = "./dataset",
    n_epochs: int = 100,
    img_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    test_set: bool = False,
    checkpoint_dir: str = "./checkpoints",
    sample_dir: str = "./outputs/samples",
    input_channels: int = 3,
    latent_dim: int = 128,
    hidden_dims: list = [32, 64, 128, 256, 512],
    learning_rate: float = 2e-4,
    beta: float = 0.0005,
    clip_grad: bool = False,

    kl_plateau_check_start: int = 25,   # start checking a few epochs after annealing ends (epoch 20)
    kl_plateau_window: int = 10,        # compare KL now vs KL this many epochs ago
    kl_plateau_min_drop_frac: float = 0.05,  # require at least 5% relative drop over the window

    use_capacity_annealing: bool = False,
    capacity_target: float = 25.0,      # target nats of KL to allow through, tune per latent_dim
    capacity_anneal_epochs: int = 40,   # ramp epochs, usually longer than beta annealing
    capacity_gamma: float = 100.0,      # weight on |KL - C|, kept large & fixed
):

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.mps.is_available() else \
             "cpu"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    model = VAE(
        input_channels=input_channels,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        img_size=img_size
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    def get_beta(
        epoch: int,
        max_annealing_epochs: int = 20,
        target_beta: float = 0.0005,
        start_beta_from_zero: bool = False,
    ) -> float:
        if epoch >= max_annealing_epochs:
            return target_beta
        if start_beta_from_zero:
            return target_beta * (epoch / max_annealing_epochs)
        else:
            return target_beta * ((epoch + 1) / max_annealing_epochs)

    def get_capacity(epoch: int) -> float:
        """Linearly ramp allowed KL capacity from 0 to capacity_target."""
        if epoch >= capacity_anneal_epochs:
            return capacity_target
        return capacity_target * (epoch / capacity_anneal_epochs)

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

    def compute_loss(recon_x, data, mu, log_var, current_beta, current_capacity):
        """Wraps model.loss(); either standard beta-VAE or capacity-annealed KL."""
        _, recon_loss, kl_loss = model.loss(recon_x, data, mu, log_var, beta=0.0)
        # model.loss with beta=0 still gives us recon_loss and kl_loss separately;
        # we build total_loss ourselves depending on the mode.
        if use_capacity_annealing:
            total_loss = recon_loss + capacity_gamma * (kl_loss - current_capacity).abs()
        else:
            total_loss = recon_loss + current_beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def train_epoch(epoch, current_beta, current_capacity):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_x, mu, log_var = model(data)
            loss, recon_loss, kl_loss = compute_loss(recon_x, data, mu, log_var, current_beta, current_capacity)

            loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch:03d} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}  Recon: {recon_loss.item():.4f}  KL: {kl_loss.item():.4f}")

        num_batches = len(train_loader)
        return total_loss / num_batches, total_recon / num_batches, total_kl / num_batches

    def validate_epoch(target_beta, target_capacity):
        model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_loader):
                data = data.to(device)
                recon_x, mu, log_var = model(data)
                loss, recon_loss, kl_loss = compute_loss(recon_x, data, mu, log_var, target_beta, target_capacity)

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

    best_val_recon = float('inf')
    kl_history = deque(maxlen=kl_plateau_window + 1)
    kl_warning_issued = False

    for epoch in range(1, n_epochs + 1):
        current_beta = get_beta(epoch - 1, target_beta=beta)
        current_capacity = get_capacity(epoch - 1)

        before_lr = optimizer.param_groups[0]['lr']

        mode_str = f"Capacity: {current_capacity:.2f}" if use_capacity_annealing else f"Beta: {current_beta:.6f}"
        print(f"\n===== Epoch {epoch:03d}/{n_epochs} | LR: {before_lr:.2e} | {mode_str} =====")

        train_loss, train_recon, train_kl = train_epoch(epoch, current_beta, current_capacity)
        print(f"Train  -> Loss: {train_loss:.4f}  Recon: {train_recon:.4f}  KL: {train_kl:.4f}")

        val_loss, val_recon, val_kl = validate_epoch(target_beta=beta, target_capacity=capacity_target)
        print(f"Valid  -> Loss: {val_loss:.4f}  Recon: {val_recon:.4f}  KL: {val_kl:.4f}")

        save_reconstructions(epoch)

        scheduler.step(val_recon)

        after_lr = optimizer.param_groups[0]['lr']
        if after_lr < before_lr:
            print(f"LR reduced: {before_lr:.2e} → {after_lr:.2e}")

        kl_history.append(val_kl)
        if (epoch >= kl_plateau_check_start
                and len(kl_history) > kl_plateau_window
                and not kl_warning_issued):
            kl_then = kl_history[0]
            kl_now = kl_history[-1]
            relative_drop = (kl_then - kl_now) / max(kl_then, 1e-8)

            if relative_drop < kl_plateau_min_drop_frac:
                kl_warning_issued = True
                print(
                    f"\n WARNING: val_kl has not meaningfully decreased over the last "
                    f"{kl_plateau_window} epochs ({kl_then:.1f} → {kl_now:.1f}, "
                    f"{relative_drop*100:.1f}% change).\n"
                    f"    The KL term may be too weak to regularize a {latent_dim}-dim latent space "
                    f"at current beta={beta}.\n"
                    f"    Consider: (a) increasing `beta`, (b) enabling `use_capacity_annealing=True`, "
                    f"or (c) reducing `latent_dim`.\n"
                )

        if val_recon < best_val_recon:
            best_val_recon = val_recon
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_recon': best_val_recon,
                'val_kl': val_kl,
                'use_capacity_annealing': use_capacity_annealing,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/best_model.pt")
            print(f"New best model saved with val_recon = {val_recon:.4f}")

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

    sig = inspect.signature(train)
    valid_params = set(sig.parameters.keys())

    filtered_config = {k: v for k, v in config.items() if k in valid_params}

    train(**filtered_config)