import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from typing import List




class BaseSemanticToImageDataset(Dataset):
    """
    Base class for semantic-to-image datasets.
    Converts single-channel semantic labels to one-hot encoding with n_classes channels.
    Returns a dictionary with 'image' and 'label' only.
    """
    def __init__(
            self, 
            root_dir, 
            n_classes, 
            transform=transforms.Compose(
                [transforms.ToTensor()]
                )
            ):
        """
        Args:
            root_dir (str): Directory with images and semantic maps.
            n_classes (int): Number of semantic classes.
            transform (callable, optional): Transform to apply to images.
            semantic_transform (callable, optional): Transform to apply to semantic maps.
        """
        self.root_dir = root_dir
        self.n_classes = n_classes
        self.transform = transform
        self.image_paths = self._get_image_paths()
        self.label_paths = self._get_label_paths()


    def _get_image_paths(self) -> List[str]:
        return NotImplementedError

    def _get_label_paths(self) -> List[str]:
        return NotImplementedError


    def __len__(self):
        return len(self.image_paths)

    def _one_hot_encode(self, label):
        """
        Convert single-channel label image to one-hot encoding using numpy.
        Args:
            label (PIL Image or np.array HxW): semantic label
        Returns:
            Tensor: one-hot encoded tensor of shape (n_classes, H, W)
        """
        if isinstance(label, Image.Image):
            label = np.array(label, dtype=np.int32)
        label_tensor = torch.from_numpy(label).long()  # H x W
        one_hot = torch.nn.functional.one_hot(label_tensor, num_classes=self.n_classes) 
        one_hot = one_hot.permute(2, 0, 1).float()
        return one_hot

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = transforms.ToTensor()(img)
        
        if self.transform:
            img = self.transform(img)  # shape: C x H x W
        
        # Load semantic map
        semantic_path = self.label_paths[idx]
        semantic = Image.open(semantic_path)
        semantic = self._one_hot_encode(semantic)  # shape: n_classes x H x W

        return {
            "image": img,
            "label": semantic
        }
    