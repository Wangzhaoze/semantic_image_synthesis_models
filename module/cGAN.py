# models/conditional_gan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Tuple

from module.base_module import BaseSemanticToImageModule
from module.backbone.unet import UNet


# ---------------------------
# PatchGAN discriminator
# ---------------------------
class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator (70x70-ish). Takes concatenated (label, image) -> single-map logits.
    """
    def __init__(self, in_channels: int = 3, label_channels: int = 3, ndf: int = 64, n_layers: int = 4):
        super().__init__()
        input_nc = in_channels + label_channels
        layers = []
        # first layer (no norm)
        layers.append(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # final convs to produce 1-channel output
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))  # logits per patch

        self.model = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, label: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        # concatenate along channels (label is the condition)
        x = torch.cat([label, img], dim=1)
        return self.model(x)


# ---------------------------
# Conditional GAN LightningModule
# ---------------------------
class ConditionalGAN(BaseSemanticToImageModule):
    """
    Conditional GAN (cGAN) where:
      - Generator = UNet backbone, maps label -> generated image
      - Discriminator = PatchDiscriminator, judges (label, real/generated image)
    Training uses:
      - Adversarial loss (BCE with logits)
      - L1 reconstruction loss between generated and target image
    """

    def __init__(
        self,
        backbone: nn.Module,
        latent_dim: int = 0,  # not used by this generator but kept for BaseSemanticToImageModule signature
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        beta1: float = 0.5,
        lambda_l1: float = 100.0,
        adv_weight: float = 1.0,
    ):
        super().__init__(backbone=backbone, latent_dim=latent_dim, lr=lr_g)
        # override learning rates: store separately
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.beta1 = beta1
        self.lambda_l1 = lambda_l1
        self.adv_weight = adv_weight

        # generator is the backbone UNet (maps label -> image)
        self.generator = backbone

        # discriminator expects label channels and image channels; infer from generator and expected inputs
        # we don't have explicit label channels in Base class, so rely on generator.in_channels
        label_ch = getattr(self.generator, "in_channels", None)
        out_ch = getattr(self.generator, "out_channels", None)
        if label_ch is None or out_ch is None:
            raise ValueError("Backbone UNet must expose in_channels and out_channels attributes.")

        self.discriminator = PatchDiscriminator(in_channels=out_ch, label_channels=label_ch)

        # loss functions
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

        # save hyperparameters for checkpointing / logging
        self.save_hyperparameters(ignore=['backbone', 'generator', 'discriminator'])

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Accepts a dict batch with keys 'label' and 'image' (matching Base class docstring).
        Returns generated image tensor (B, C_out, H, W).
        """
        label = batch["label"].float()
        # generator returns images in generator's output range (we used Tanh by default in UNet, adjust if needed)
        fake = self.generator(label)
        return fake

    def _get_labels(self, shape: torch.Size, real: bool, device: torch.device):
        """
        PatchGAN outputs a map; labels should be same spatial dims as output logits.
        Create tensors filled with 1.0 (real) or 0.0 (fake).
        """
        return torch.full(shape, 1.0 if real else 0.0, device=device, dtype=torch.float)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        opt_g, opt_d = self.optimizers()

        labels = batch["label"].float()
        reals = batch["image"].float()
        device = reals.device

        # -------------------------
        # 1) Train Discriminator
        # -------------------------
        fakes = self.generator(labels).detach()
        real_logits = self.discriminator(labels, reals)
        fake_logits = self.discriminator(labels, fakes)

        real_targets = torch.ones_like(real_logits, device=device)
        fake_targets = torch.zeros_like(fake_logits, device=device)

        loss_real = self.adv_loss(real_logits, real_targets)
        loss_fake = self.adv_loss(fake_logits, fake_targets)
        d_loss = 0.5 * (loss_real + loss_fake)

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # -------------------------
        # 2) Train Generator
        # -------------------------
        fakes = self.generator(labels)
        fake_logits_for_g = self.discriminator(labels, fakes)
        real_targets_for_g = torch.ones_like(fake_logits_for_g, device=device)

        adv = self.adv_loss(fake_logits_for_g, real_targets_for_g)
        l1 = self.l1_loss(fakes, reals)
        g_loss = self.adv_weight * adv + self.lambda_l1 * l1

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # logging
        self.log("train/d_loss", d_loss, on_step=True, prog_bar=True)
        self.log("train/g_loss", g_loss, on_step=True, prog_bar=True)
        self.log("train/g_adv", adv, on_step=True)
        self.log("train/g_l1", l1, on_step=True)

    def configure_optimizers(self):
        self.automatic_optimization = False
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(self.beta1, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(self.beta1, 0.999))
        return [opt_g, opt_d]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # simple validation: compute L1 between generated and real, and adversarial score
        labels = batch["label"].float()
        reals = batch["image"].float()
        fakes = self.generator(labels)

        l1 = self.l1_loss(fakes, reals)
        # adversarial score (how "real" discriminator thinks generated samples are)
        with torch.no_grad():
            fake_logits = self.discriminator(labels, fakes)
            # convert logits to probabilities via sigmoid and take mean
            adv_score = torch.sigmoid(fake_logits).mean()

        self.log("val/l1", l1, prog_bar=True, logger=True)
        self.log("val/adv_score", adv_score, prog_bar=True, logger=True)

        # optionally return images for visual logging (trainer logger / callbacks can handle them)
        return {"val_l1": l1, "val_adv_score": adv_score, "generated": fakes.detach(), "target": reals.detach()}

