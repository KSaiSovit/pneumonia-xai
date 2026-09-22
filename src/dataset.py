import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=224, augment=True):
        self.root = os.path.join(root_dir, split)
        self.classes = ["NORMAL", "PNEUMONIA"]
        self.samples = []
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_class_weights(dataset):
    """Compute inverse-frequency weights for imbalanced data."""
    labels = [s[1] for s in dataset.samples]
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * 2  # normalise to mean 1
    return torch.tensor(weights, dtype=torch.float32)


def build_loaders(cfg):
    d = cfg["data"]
    train_ds = ChestXRayDataset(d["data_dir"], "train", d["img_size"], augment=True)
    val_ds   = ChestXRayDataset(d["data_dir"], "val",   d["img_size"], augment=False)
    test_ds  = ChestXRayDataset(d["data_dir"], "test",  d["img_size"], augment=False)

    # Weighted sampler for the training set to combat class imbalance
    labels = [s[1] for s in train_ds.samples]
    counts = np.bincount(labels, minlength=2)
    sample_weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=d["batch_size"],
                              sampler=sampler, num_workers=d["num_workers"])
    val_loader   = DataLoader(val_ds, batch_size=d["batch_size"],
                              shuffle=False, num_workers=d["num_workers"])
    test_loader  = DataLoader(test_ds, batch_size=d["batch_size"],
                              shuffle=False, num_workers=d["num_workers"])

    class_weights = get_class_weights(train_ds)
    return train_loader, val_loader, test_loader, class_weights