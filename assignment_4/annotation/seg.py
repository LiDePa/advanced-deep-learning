from typing import Optional
from threading import Lock
from PIL import Image
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os


SEG_DIR = os.path.dirname(os.path.abspath(__file__))
SEGMENTATION_THRESHOLD = 0.5 # prediction threshold above which pixels are part of the mask created by sam2


class Segmentor:
    def __init__(self, device="cpu"):
        self._lock = Lock()
        self.device = device

        # initialize sam2 model
        checkpoint = os.path.join(SEG_DIR, "sam2.1_hiera_tiny.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device = self.device))

        # TODO: register forward hooks here
        # you'll want to create new methods for the Segmentor class and use them as callbacks for the hooks

    def prepare_embeddings(self, img_path: str):
        """
        Prepares the embeddings for the provided image. This function is called when the
        image is loaded for the first time.
        :param img_path: Path to the image
        """
        with self._lock:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image.convert("RGB"))
            self.predictor.set_image(image)

    def segment(self, img_path: str, clicks: np.ndarray, prev_masks: Optional[np.ndarray]):
        """
        Segments the given image based on the provided clicks and (optional) mask.
        :param img_path: Path to the image to segment.
        :param clicks: 2D NumPy array. Each row is one click. The first row is the first click. Columns are: `(x, y, isPositive)`.
        `x` and `y` coordinates are normalized to [0, 1]. Therefore, you must multiply by image height and width to get
        absolute coordinates.
        `isPositive` is 1 if it's a positive click and 0 if it's a negative click.
        :param prev_mask: If available, the mask output from the previous call to this function.
        If this is the first click, `prev_mask` is set to `None`.
        :returns: A tuple `(mask, logits)` where mask is a 2D NumPy array with `np.bool` data type and logits are the
        logits that can be used as mask input for the next turn.
        """
        image = Image.open(img_path)
        w = image.width
        h = image.height
        input_points = clicks[:, :2].copy()
        input_points[:, 0] *= w
        input_points[:, 1] *= h
        input_points = input_points.round().astype(int)
        input_labels = clicks[:, 2].astype(int)

        with self._lock:
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                if prev_masks is not None:
                    masks, logits, _ = self.predictor.predict(input_points, input_labels, prev_masks)
                else:
                    masks, logits, _ = self.predictor.predict(input_points, input_labels)

                binary_mask = masks[0] > SEGMENTATION_THRESHOLD
                return binary_mask, logits[0]

def compute_pca3_visualization(features: torch.Tensor):
    # """
    # Computes the PCA for the given features and return the visualization. This means that the number of channels
    # of the feature map is reduced to 3 using PCA and the result is visualized as an RGB image.
    # :param features: Feature map to visualize with shape [C, H, W] as torch float
    # :return: PCA map as NumPy array.
    # """

    oldh, oldw = features.shape[-2:]
    features = torch.reshape(features, (-1, oldh * oldw))
    features_centered = features - torch.mean(features, dim=1, keepdim=True)
    sigma = torch.cov(features_centered)
    eigenvals, eigenvecs = torch.linalg.eig(sigma)
    _, max_3_eigenvals = torch.topk(eigenvals.real, k=3)
    U = eigenvecs[:, max_3_eigenvals].real
    pca_features = U.T @ features_centered  # [3, oldh * oldw] | torch float
    pca_features = torch.reshape(pca_features, (3, oldh, oldw))  # [3, oldh, oldw] | torch float

    pca_features_np = pca_features.detach().cpu().numpy()
    pca_features_np = np.transpose(pca_features_np, (1, 2, 0))
    pca_features_np = (pca_features_np - pca_features_np.min()) / (pca_features_np.max() - pca_features_np.min())

    return pca_features_np

# TODO: don't forget to add your code for task 4.2
