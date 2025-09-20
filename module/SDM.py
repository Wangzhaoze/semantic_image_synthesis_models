from module.backbone.gaussian_diffusion import GaussianDiffusion
from module.backbone.unet_attention import UNetModel
from module.backbone import gaussian_diffusion as gd
from module.backbone.respace import SpacedDiffusion, space_timesteps

# train_diffusion_lightning.py
import argparse
from typing import Dict, Any, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from torch.optim import AdamW


from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser


class SemanticDiffusionModule(pl.LightningModule):
    """
    LightningModule for diffusion training with segmentation map conditioning
    """
    def __init__(
        self,
        model: UNetModel,              # 直接传入实例化的UNet模型
        diffusion: SpacedDiffusion,    # 直接传入实例化的扩散过程
        lr: float = 1e-4,              # 学习率
        weight_decay: float = 0.0,     # 权重衰减
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'diffusion'])  # 忽略模型和扩散过程
        
        self.model = model
        self.diffusion = diffusion
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        """
        训练步骤
        batch: (x, cond) 其中 x: [B, 3, H, W], cond: dict {"labelmap": [B, 1, H, W]}
        """
        x, cond = batch
        x = x.to(dtype=torch.float32)
        
        # 采样时间步
        B = x.shape[0]
        t = torch.randint(
            low=0, 
            high=self.diffusion.num_timesteps, 
            size=(B,), 
            device=x.device, 
            dtype=torch.long
        )
        
        # 计算损失
        losses = self.diffusion.training_losses(
            self.model, x, t, model_kwargs=cond
        )
        
        # 提取主损失
        if isinstance(losses, dict) and "loss" in losses:
            loss = losses["loss"].mean()
        else:
            loss = losses.mean()
        
        # 记录损失
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        # 记录其他损失项（如果有）
        if isinstance(losses, dict):
            for k, v in losses.items():
                if k == "loss":
                    continue
                try:
                    val = v.mean()
                    self.log(f"train/{k}", val, on_step=True, on_epoch=True, sync_dist=True)
                except Exception:
                    pass
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        验证步骤
        batch: (x, cond)
        """
        x, cond = batch
        x = x.to(dtype=torch.float32)

        B = x.shape[0]
        t = torch.randint(
            low=0,
            high=self.diffusion.num_timesteps,
            size=(B,),
            device=x.device,
            dtype=torch.long
        )

        # 计算损失
        losses = self.diffusion.training_losses(
            self.model, x, t, model_kwargs=cond
        )

        # 主损失
        if isinstance(losses, dict) and "loss" in losses:
            loss = losses["loss"].mean()
        else:
            loss = losses.mean()

        # 记录损失
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # 可选记录其他损失项
        if isinstance(losses, dict):
            for k, v in losses.items():
                if k == "loss":
                    continue
                try:
                    val = v.mean()
                    self.log(f"val/{k}", val, on_step=False, on_epoch=True, sync_dist=True)
                except Exception:
                    pass

        return {"x": x, "cond": cond, "loss": loss}


    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer

