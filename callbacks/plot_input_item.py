import matplotlib.pyplot as plt
import io
import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
plt.switch_backend('agg')  # 避免交互式绘图卡死

# TODO
# add RGB image
class PlotInputItemCallback(Callback):
    """
    在训练开始时（on_fit_start）可视化若干输入样本：
    Spectrum (RA map) | Label map
    """
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples

    def on_fit_start(self, trainer, pl_module):
        # 获取一个 batch
        train_loader = trainer.datamodule.train_dataloader() if hasattr(trainer.datamodule, "train_dataloader") else trainer.train_dataloader
        batch = next(iter(train_loader))
        x, cond = batch
        x = x[:self.num_samples]
        cond = {k: v[:self.num_samples] for k, v in cond.items()}

        # 可视化
        fig, axes = plt.subplots(self.num_samples, 2, figsize=(6, 3 * self.num_samples))
        if self.num_samples == 1:
            axes = np.expand_dims(axes, 0)

        for batch_idx in range(self.num_samples):
            # Spectrum
            spectrum = np.power(10, x[batch_idx, 0].numpy())
            axes[batch_idx, 0].imshow(spectrum, cmap='jet')
            axes[batch_idx, 0].set_title("Input Spectrum (RA map)")
            axes[batch_idx, 0].axis('off')

            # Label map (argmax if multi-channel)
            label_map = cond['labelmap'][batch_idx].numpy().transpose((1, 2, 0))
            label_map /= label_map.max()  # 归一化到 [0, 1]
            axes[batch_idx, 1].imshow(label_map)
            axes[batch_idx, 1].set_title("Label Map")
            axes[batch_idx, 1].axis('off')

        plt.tight_layout()

        # 保存到 TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = plt.imread(buf)
        buf.close()
        plt.close(fig)

        # 转 tensor
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        trainer.logger.experiment.add_image(
            "InputSamples", image_tensor[0], global_step=0
        )
