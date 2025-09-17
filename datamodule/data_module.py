from torch.utils.data import Dataset, DataLoader, Subset
from datamodule.configs import RadarConfig, ConfMapConfig, CRUW_REC, RAMPCNN_REC
from typing import List, Dict
import pandas as pd
import torch
from utils.io import load_rec_data
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import KFold
from utils.io import load_adc
import os
import glob


class CRUWDataset(Dataset):
    def __init__(
            self, 
            data_dir: str = 'D:/Datasets/CRUW', 
            rec_date: str = '2019_04_30',
            radar_cfg: RadarConfig = RadarConfig(),
            confmap_cfg: ConfMapConfig = ConfMapConfig(),
            ):
        super().__init__()
        self.data_dir = data_dir
        self.recordings = CRUW_REC.get(rec_date, [])
        self.radar_cfg = radar_cfg
        self.confmap_cfg = confmap_cfg

        self.data_table = self._generate_data_table()

    def _generate_data_table(self) -> pd.DataFrame:

        rec_frame_table = pd.DataFrame()

        for rec in self.recordings:
            adc_paths = glob.glob(os.path.join(self.data_dir, 'TRAIN_RAD_H', rec, 'RADAR_RA_H', "*_0000.npy"))
            
            table = pd.read_csv(os.path.join(self.data_dir, 'TRAIN_RAD_H_ANNO', f'{rec}.txt'), 
                sep=r"\s+",   # split on any whitespace
                header=None,  # no header in your file
                names=["timestamp", "range", "azimuth", "class"])
            
            table = table.groupby('timestamp').agg({
                'range': list,
                'azimuth': list,
                'class': list
                }).reset_index()
            
            table['ra_path'] = adc_paths

            rec_frame_table = pd.concat([rec_frame_table, table], ignore_index=True)

        return rec_frame_table
    
    def __len__(self):
        return len(self.data_table)
    
    def __getitem__(self, index):
        frame_data = self.data_table.iloc[index]
        ra_path = frame_data['ra_path']
        ra_map = np.load(ra_path)  # Shape: [numRangeBins, numAngleBins, 2]
        ra_map = np.abs(ra_map[..., 0] + 1j * ra_map[..., 1])[np.newaxis, ...]  # Shape: [1, numRangeBins, numAngleBins]
        ra_map = np.log10(ra_map + 1e-8)  # Log scale
        
        # Generate Confidence Map
        label = self._generate_confmap(frame_data)  # Shape: [n_class, numRangeBins, numAngleBins]

        # ra_map: (1, 128, 128)
        # confmap: (num_classes, 128, 128)
        return ra_map, {'labelmap': label}
        

    def _generate_confmap(self, frame_label: pd.Series) -> np.ndarray:
        """
        Generate confidence map for a single frame.
        
        :param frame_label: DataFrame row containing frame label information with:
            - 'range': list of range values
            - 'azimuth': list of azimuth values  
            - 'class': list of class names
        :param radar_cfg: Radar configuration
        :param confmap_cfg: Confidence map configuration
        :return: Confidence map with shape [n_class, numRangeBins, numAngleBins]
        """
        # Initialize confidence map
        confmap = np.zeros((self.confmap_cfg.n_class, self.radar_cfg.numRangeBins, self.radar_cfg.numAngleBins), dtype=float)

        # Create range and angle grids
        range_grid = np.linspace(0, self.radar_cfg.maxRange, self.radar_cfg.numRangeBins)
        angle_grid = np.linspace(-self.radar_cfg.maxAngle, self.radar_cfg.maxAngle, self.radar_cfg.numAngleBins)

        # Process each object in the frame
        ranges = frame_label['range']
        azimuths = frame_label['azimuth']
        classes = frame_label['class']
        
        for range_val, azimuth_val, class_name in zip(ranges, azimuths, classes):
            if class_name not in self.confmap_cfg.classes:
                print(f"Warning: Unrecognized class: {class_name}")
                continue
            class_id = self.confmap_cfg.classes.index(class_name)
            # Convert physical coordinates to pixel indices
            # range_idx = int(np.clip(range_val / self.radar_cfg.maxRange * self.radar_cfg.numRangeBins, 0, self.radar_cfg.numRangeBins - 1))
            # angle_idx = int(np.clip((azimuth_val + self.radar_cfg.maxAngle) / (2 * self.radar_cfg.maxAngle) * self.radar_cfg.numAngleBins, 
            #                     0, self.radar_cfg.numAngleBins - 1))
            range_idx = (np.abs(range_grid - range_val)).argmin()
            angle_idx = (np.abs(angle_grid - azimuth_val * 180 / np.pi)).argmin()

            # Calculate adaptive sigma value based on range
            sigma = 2 * np.arctan(self.confmap_cfg.confmap_length[class_name] / (2 * range_val)) * self.confmap_cfg.confmap_sigmas[class_name]

            # Apply sigma range limits
            sigma_min, sigma_max = self.confmap_cfg.confmap_sigmas_interval[class_name]
            sigma = np.clip(sigma, sigma_min, sigma_max)
            
            # Generate Gaussian distribution
            for i in range(self.radar_cfg.numRangeBins):
                for j in range(self.radar_cfg.numAngleBins):
                    # Calculate distance (range dimension multiplied by 2 to balance scales)
                    distance = (((range_idx - i) * 2) ** 2 + (angle_idx - j) ** 2) / sigma ** 2

                    if distance < self.confmap_cfg.gaussian_thres:
                        value = np.exp(-distance / 2) / (2 * np.pi)
                        # Keep maximum value
                        if value > confmap[class_id, i, j]:
                            confmap[class_id, i, j] = value 
        return confmap


class RADataModule(pl.LightningDataModule):
    def __init__(
            self, 
            dataset: Dataset,
            batch_size: int,
            num_workers: int,
            prepare_data_flag: bool
            ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepare_data_flag = prepare_data_flag

    def prepare_data(self):
        if self.prepare_data_flag:
            # Download, tokenize, etc.
            pass
        else:
            pass

    def setup(self, stage=None):
        # 创建索引分割
        indices = np.arange(len(self.dataset))

        # 简单划分为训练集和验证集（80%训练，20%验证）
        np.random.seed(42)

        np.random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_idx, val_idx = indices[:split], indices[split:]
        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx)
        
        # if stage == 'fit' or stage is None:
        #     # K-fold交叉验证
        #     kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        #     all_splits = list(kf.split(indices))
        #     train_idx, val_idx = all_splits[self.current_fold]

        #     self.train_dataset = Subset(self.dataset, train_idx)
        #     self.val_dataset = Subset(self.dataset, val_idx)

        # if stage == 'test' or stage is None:
        #     # 测试集可以使用最后20%的数据
        #     test_size = int(0.2 * len(indices))
        #     test_idx = indices[-test_size:]
        #     self.test_dataset = Subset(self.dataset, test_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset, 
    #         batch_size=self.batch_size, 
    #         num_workers=self.num_workers
    #     )
    

