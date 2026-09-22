import os, time, copy, yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, recall_score
from tqdm import tqdm

from dataset import build_loaders
from model import build_model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, all_probs, all_labels = 0.0, [], []
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    return epoch_loss, auc


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss, all_probs, all_labels = 0.0, [], []
    for imgs, labels in tqdm(loader, desc="eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    preds = (np.array(all_probs) >= threshold).astype(int)
    sensitivity = recall_score(all_labels, preds, pos_label=1, zero_division=0)
    return epoch_loss, auc, sensitivity, all_probs, all_labels


def train(cfg):
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_weights = build_loaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_auc, best_state, patience_counter = 0.0, None, 0

    # ---- Phase 1: train classifier head only ----
    print("\n=== Phase 1: training classifier head ===")
    model.freeze_backbone()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=cfg["training"]["phase1_lr"],
                     weight_decay=cfg["training"]["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    for epoch in range(cfg["training"]["phase1_epochs"]):
        tr_loss, tr_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_auc, vl_sens, _, _ = evaluate(model, val_loader, criterion, device,
                                                   cfg["evaluation"]["threshold"])
        scheduler.step(vl_auc)
        print(f"[P1] Epoch {epoch+1}: train_loss={tr_loss:.4f} train_auc={tr_auc:.4f} | "
              f"val_loss={vl_loss:.4f} val_auc={vl_auc:.4f} val_sens={vl_sens:.4f}")
        if vl_auc > best_auc:
            best_auc, best_state = vl_auc, copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping in Phase 1.")
                break

    # ---- Phase 2: full fine-tuning ----
    print("\n=== Phase 2: full fine-tuning ===")
    model.load_state_dict(best_state)
    model.unfreeze_backbone()
    optimizer = Adam(model.parameters(),
                     lr=cfg["training"]["phase2_lr"],
                     weight_decay=cfg["training"]["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    patience_counter = 0

    for epoch in range(cfg["training"]["phase2_epochs"]):
        tr_loss, tr_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_auc, vl_sens, _, _ = evaluate(model, val_loader, criterion, device,
                                                   cfg["evaluation"]["threshold"])
        scheduler.step(vl_auc)
        print(f"[P2] Epoch {epoch+1}: train_loss={tr_loss:.4f} train_auc={tr_auc:.4f} | "
              f"val_loss={vl_loss:.4f} val_auc={vl_auc:.4f} val_sens={vl_sens:.4f}")
        if vl_auc > best_auc:
            best_auc, best_state = vl_auc, copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping in Phase 2.")
                break

    torch.save(best_state, cfg["paths"]["model_save"])
    print(f"\nBest validation AUC: {best_auc:.4f}. Model saved to {cfg['paths']['model_save']}")
    return model, test_loader


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    train(cfg)