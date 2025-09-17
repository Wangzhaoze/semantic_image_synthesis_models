import matplotlib.pyplot as plt
import io
import torch
from pytorch_lightning.callbacks import Callback
import numpy as np
plt.switch_backend('agg')  # 使用非交互式后端

class PlotSyntheticRAmapsCallback(Callback):
    """
    在每个 epoch 结束时，在 TensorBoard 上可视化 n 对图像：
    原图 | 条件 label map | 模型生成 RA 图
    """
    def __init__(self, num_samples=4, every_n_epochs=1):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        # 取一个 batch
        val_loader = trainer.datamodule.val_dataloader() if hasattr(trainer.datamodule, "val_dataloader") else trainer.val_dataloaders[0]
        batch = next(iter(val_loader))
        x, cond = batch
        x = x[:self.num_samples].to(pl_module.device)
        cond = {k: v[:self.num_samples].to(pl_module.device) for k, v in cond.items()}

        # 生成去噪 RA 图
        with torch.no_grad():
            generated = pl_module.diffusion.p_sample_loop(
                pl_module.model,
                x.shape,
                device=pl_module.device,
                model_kwargs=cond
            )

        # 可视化
        fig, axes = plt.subplots(self.num_samples, 3, figsize=(9, 3 * self.num_samples))
        if self.num_samples == 1:
            axes = np.expand_dims(axes, 0)

        for batch_idx in range(self.num_samples):
            # 原图
            spectrum = np.power(10, x[batch_idx, 0].cpu().numpy())
            axes[batch_idx, 0].imshow(spectrum, cmap='jet')
            axes[batch_idx, 0].set_title("Original RA")
            axes[batch_idx, 0].axis('off')

            label_map = cond['labelmap'][batch_idx].cpu().numpy().transpose((1, 2, 0))
            label_map /= label_map.max()  # 归一化到 [0, 1]
            axes[batch_idx, 1].imshow(label_map)
            axes[batch_idx, 1].set_title("Condition Map")
            axes[batch_idx, 1].axis('off')

            # 生成 RA 图
            generated_spectrum = np.power(10, generated[batch_idx, 0].cpu().numpy())
            axes[batch_idx, 2].imshow(generated_spectrum, cmap='viridis')
            axes[batch_idx, 2].set_title("Generated RA")
            axes[batch_idx, 2].axis('off')

        plt.tight_layout()

        # 保存到 TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = plt.imread(buf)
        buf.close()
        plt.close(fig)

        # 转成 tensor
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        trainer.logger.experiment.add_image(
            "PlotGeneratedRAMap", image_tensor[0], global_step=epoch
        )
