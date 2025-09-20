from module.backbone.unet import UNet
from module.backbone.respace import SpacedDiffusion, space_timesteps
import module.backbone.gaussian_diffusion as gd
from module.SDM import SemanticDiffusionModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from module.cGAN import ConditionalGAN
from datamodule.data_module import SemanticToImageDataModule
from datamodule.dataset.cmp_facade_dataset import CMPFacadeDataset
from callbacks.plot_generated_image import PlotGeneratedImagesCallback
from pytorch_lightning import Trainer, loggers as pl_loggers


if __name__ == "__main__":
    import torch
    torch.set_float32_matmul_precision("medium")  # (optional, to use Tensor Cores properly)

    module = ConditionalGAN(
        backbone=UNet(
            in_channels=12,
            out_channels=3
        )
    )

    data_module = SemanticToImageDataModule(
        dataset=CMPFacadeDataset(
        ),
        batch_size=4,
        num_workers=2
    )

    callbacks = [PlotGeneratedImagesCallback(num_samples=4, every_n_epochs=1)]

    tb_logger = pl_loggers.TensorBoardLogger("cgan/")

    # 或者在训练器配置中强制使用CPU
    trainer = pl.Trainer(
        accelerator="auto",  # 强制使用CPU
        devices='auto',
        max_epochs=50,
        logger=tb_logger,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # 开始训练
    trainer.fit(model=module, datamodule=data_module)

