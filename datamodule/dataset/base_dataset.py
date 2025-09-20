from torch.utils.data import Dataset
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
            transform=None
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
        return NotImplementedError

    def __getitem__(self, idx):
        return NotImplementedError
