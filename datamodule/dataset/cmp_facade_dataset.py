import os
import glob
from .base_dataset import BaseSemanticToImageDataset
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


class CMPFacadeDataset(BaseSemanticToImageDataset):
    """
    CMP Facade dataset for semantic-to-image generation.
    """
    def __init__(self, root_dir: str = './data/CMP_Facade_Dataset/base/', n_classes: int = 12, transform=None, img_size: int = 256):
        super().__init__(root_dir, n_classes, transform)
        self.resize = transforms.Resize((img_size, img_size)) 

    def _get_label_paths(self):
        return sorted(glob.glob(os.path.join(self.root_dir, "cmp_*.png")))
    
    def _get_image_paths(self):
        return sorted(glob.glob(os.path.join(self.root_dir, "cmp_*.jpg")))
    
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx])
        

        img = self.resize(img)
        label = self.resize(label)

        img = transforms.ToTensor()(img)
        label = torch.as_tensor(np.array(label)-1, dtype=torch.long)

        # one-hot encode
        label = torch.nn.functional.one_hot(label, num_classes=self.n_classes).permute(2,0,1).float()

        return {"image": img, "label": label}