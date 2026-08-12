import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from typing import Union, Tuple, Optional, List

def download_dataset(
    name: str = "flwrlabs/celeba",
    cache_dir: Optional[str] = "./dataset"
    ):

    from datasets import load_dataset
    dataset = load_dataset(name,cache_dir=cache_dir)

    print(dataset)

    print("Train:", len(dataset["train"]))
    print("Valid:", len(dataset["valid"]))
    print("Test :", len(dataset["test"]))

    total = sum(len(dataset[x]) for x in dataset)
    print("Total:", total)

    return dataset


class CelebADataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        img_size: Union[int, Tuple[int, int]],
        attributes: Optional[List[str]] = None
    ):
        self.dataset = hf_dataset

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if attributes is not None:
            self.attributes = attributes
        else:
            all_features = list(self.dataset.features.keys())
            not_feature = ["image", "celeb_id"]
            self.attributes = [f for f in all_features if f not in not_feature]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img.numpy(), mode="RGB")
        img = self.transform(img)

        attr_tensor = torch.tensor(
            [self.dataset[idx][attr] for attr in self.attributes],
            dtype=torch.float32
        )
        return img, attr_tensor

def dataloaders(
    dataset,
    img_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    test_set: bool = True,
    cache_dir: Optional[str] = "./dataset"
) -> tuple[DataLoader, DataLoader, DataLoader | None]:

    dataset = download_dataset(dataset,cache_dir)

    train_dataset = CelebADataset(dataset["train"], img_size)
    val_dataset = CelebADataset(dataset["valid"], img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_loader = None

    if test_set:
        test_dataset = CelebADataset(dataset["test"], img_size)

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

    return train_loader, val_loader, test_loader

def show_images(
    images,
    attributes=None,
    num_cols=4,
    figsize=(12, 12)
    ):
    """
    images: torch.Tensor of shape (B, C, H, W), values normalized [-1, 1]
    attributes: optional torch.Tensor of shape (B, num_attrs) – for titles
    """
    images = images * 0.5 + 0.5
    images = torch.clamp(images, 0, 1)

    batch_size = images.size(0)
    num_rows = (batch_size + num_cols - 1) // num_cols

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        if attributes is not None:
            attrs = attributes[i]
            # For brevity, just show count of positive attributes
            pos_count = (attrs > 0.5).sum().item()
            axes[i].set_title(f"Pos: {pos_count}", fontsize=8)

    for j in range(batch_size, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
