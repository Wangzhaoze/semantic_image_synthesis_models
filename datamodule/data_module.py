from .dataset.base_dataset import BaseSemanticToImageDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

class SemanticToImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: BaseSemanticToImageDataset,
        batch_size: int = 8,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True
    ):
        """
        Args:
            dataset: BaseSemanticToImageDataset 实例
            batch_size: 每个 batch 的大小
            num_workers: dataloader worker 数量
            val_split: 验证集占比
            test_split: 测试集占比
            shuffle: 是否对训练集打乱
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle

    def setup(self, stage=None):
        """按照 split 拆分数据集"""
        dataset_len = len(self.dataset)
        test_len = int(dataset_len * self.test_split)
        val_len = int(dataset_len * self.val_split)
        train_len = dataset_len - val_len - test_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_len, val_len, test_len]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
