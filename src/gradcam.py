import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import ChestXRayDataset
from model import build_model


class GradCAMExplainer:
    """Grad-CAM for DenseNet-121 targeting the final dense block's ReLU."""

    def __init__(self, model, device, target_layer=None):
        self.model = model
        self.device = device
        if target_layer is None:
            # DenseNet-121's final feature map lives in features.norm5
            target_layer = [model.backbone.features.norm5]
        self.cam = GradCAM(model=model, target_layers=target_layer)

    def explain(self, img_tensor, class_idx=1):
        """
        img_tensor: (1, 3, H, W) normalised tensor
        Returns: (H, W) heatmap in [0, 1]
        """
        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = self.cam(input_tensor=img_tensor.to(self.device),
                                 targets=targets)[0]
        return grayscale_cam

    def overlay(self, img_tensor, class_idx=1, alpha=0.45):
        """Return an RGB uint8 image with the Grad-CAM overlay."""
        cam = self.explain(img_tensor, class_idx)
        # Denormalise for display
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
        img = (img * std + mean).clip(0, 1).astype(np.float32)
        overlay = show_cam_on_image(img, cam, use_rgb=True)
        return overlay


def generate_gradcam_grid(cfg, num_normal=4, num_pneumonia=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(cfg["paths"]["model_save"], map_location=device))
    model.eval()

    explainer = GradCAMExplainer(model, device)
    test_ds = ChestXRayDataset(cfg["data"]["data_dir"], "test", cfg["data"]["img_size"],
                               augment=False)

    # Pick balanced samples
    normal_idx   = [i for i, s in enumerate(test_ds.samples) if s[1] == 0][:num_normal]
    pneumonia_idx = [i for i, s in enumerate(test_ds.samples) if s[1] == 1][:num_pneumonia]
    indices = normal_idx + pneumonia_idx

    os.makedirs(cfg["paths"]["gradcam_dir"], exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for ax, idx in zip(axes.ravel(), indices):
        img_t, label = test_ds[idx]
        img_batch = img_t.unsqueeze(0).to(device)
        overlay = explainer.overlay(img_batch, class_idx=1)

        with torch.no_grad():
            prob = torch.softmax(model(img_batch), dim=1)[0, 1].item()

        ax.imshow(overlay)
        ax.set_title(f"{'Pneumonia' if label else 'Normal'}\nP(pneu)={prob:.3f}",
                     fontsize=11)
        ax.axis("off")

    plt.suptitle("Grad-CAM Attribution — DenseNet-121", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(cfg["paths"]["gradcam_dir"], "gradcam_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM grid saved to {out_path}")


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    generate_gradcam_grid(cfg)