# image2image_pl_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Tuple, Dict

# -----------------------
# Base Lightning Module
# -----------------------
class BaseSemanticToImageModule(pl.LightningModule):
    """
    Base LightningModule for image2image tasks.
    - Expects batches that are dicts with keys 'label' and 'image' both shaped BCHW (torch.Tensor).
      * 'label' is the input (e.g. one-hot semantic maps) with C_in channels.
      * 'image' is the target RGB image (C_out, H, W), typically C_out=3.
    - `backbone` must be an nn.Module that maps input label (B, C_in, H, W) -> (B, latent_dim).
      If your backbone returns a feature map, wrap it so the output is flattened to (B, latent_dim).
    - Subclasses should implement forward() and training/validation steps.
    """
    def __init__(
        self,
        backbone: nn.Module,
        lr: float = 2e-4,
        weight_decay: float = 0.0
    ):
        super().__init__()
        self.backbone = backbone
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        # subclasses can override to provide multiple optimizers
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.999))
        return opt

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")

    def _unpack_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # gets input and target; ensures they are float tensors on correct device
        inp = batch["label"].float()
        target = batch["image"].float()
        return inp, target
    
    def generation_step(self, x, cond) -> torch.Tensor:
        return NotImplementedError
