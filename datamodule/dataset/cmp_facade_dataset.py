import os
import glob
from .base_dataset import BaseSemanticToImageDataset


class CMPFacadeDataset(BaseSemanticToImageDataset):
    """
    CMP Facade dataset for semantic-to-image generation.
    """
    def __init__(self, root_dir: str = './data/CMP_Facade_Dataset/base/', n_classes: int = 12, transform=None):
        super().__init__(root_dir, n_classes, transform)

    def _get_label_paths(self):
        return sorted(glob.glob(os.path.join(self.root_dir, "cmp_*.png")))
    
    def _get_image_paths(self):
        return sorted(glob.glob(os.path.join(self.root_dir, "cmp_*.jpg")))

