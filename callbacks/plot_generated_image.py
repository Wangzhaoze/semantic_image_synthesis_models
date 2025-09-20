# callbacks/plot_generated_images.py
import torch
import matplotlib.pyplot as plt
from pytorch_lightning import Callback

class PlotGeneratedImagesCallback(Callback):
    def __init__(self, num_samples: int = 4, every_n_epochs: int = 1):
        """
        Args:
            num_samples: 每次可视化多少个样本
            every_n_epochs: 每隔多少个 epoch 执行一次
        """
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        # 只在指定的 epoch 执行
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        val_loader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))
        labels = batch["label"][: self.num_samples].to(pl_module.device)
        reals = batch["image"][: self.num_samples].to(pl_module.device)

        # 预测
        with torch.no_grad():
            preds = pl_module.generator(labels)

        # to cpu
        labels = labels.cpu()
        reals = reals.cpu()
        preds = preds.cpu()

        # matplotlib 可视化
        fig, axes = plt.subplots(self.num_samples, 3, figsize=(9, 3 * self.num_samples))

        if self.num_samples == 1:
            axes = [axes]  # 保持可迭代

        for i in range(self.num_samples):
            # label 可能是 one-hot, 转成 argmax (类别索引图)
            label_vis = labels[i].argmax(dim=0) if labels[i].dim() == 3 else labels[i]
            label_vis = label_vis.numpy()

            axes[i][0].imshow(label_vis, cmap="tab20")
            axes[i][0].set_title("Label")
            axes[i][0].axis("off")

            axes[i][1].imshow(self._to_numpy_img(reals[i]))
            axes[i][1].set_title("GT Image")
            axes[i][1].axis("off")

            axes[i][2].imshow(self._to_numpy_img(preds[i]))
            axes[i][2].set_title("Pred Image")
            axes[i][2].axis("off")

        plt.tight_layout()

        # log 到 TensorBoard
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.add_figure(
                "val/generated_images", fig, global_step=trainer.global_step
            )

        plt.close(fig)

    def _to_numpy_img(self, tensor):
        """
        把 (C,H,W) tensor 转成 (H,W,C)，并做反归一化到 [0,1]
        """
        img = tensor.detach().permute(1, 2, 0).numpy()
        img = (img + 1.0) / 2.0  # 如果用 Tanh 输出到 [-1,1]
        img = img.clip(0, 1)
        return img
