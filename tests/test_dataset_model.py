import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.dataset import ChestXRayDataset, get_class_weights
from src.model import PneumoniaDenseNet


class TestDatasetModel(unittest.TestCase):
    def test_dataset_and_class_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ["train", "val", "test"]:
                for cls in ["NORMAL", "PNEUMONIA"]:
                    d = root / split / cls
                    d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / "train" / "NORMAL" / "n1.png")
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / "train" / "PNEUMONIA" / "p1.png")
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / "train" / "PNEUMONIA" / "p2.png")

            ds = ChestXRayDataset(str(root), split="train", img_size=16, augment=False)
            self.assertEqual(len(ds), 3)
            weights = get_class_weights(ds).numpy()
            self.assertGreater(weights[0], weights[1])

    def test_freeze_and_unfreeze_backbone(self):
        model = PneumoniaDenseNet(pretrained=False)
        model.freeze_backbone()
        self.assertTrue(all(not p.requires_grad for p in model.backbone.features.parameters()))
        model.unfreeze_backbone()
        self.assertTrue(all(p.requires_grad for p in model.backbone.features.parameters()))


if __name__ == "__main__":
    unittest.main()
