import torch
import os
from torch.utils.data import DataLoader, Dataset

from PIL import Image

class TrainDataset(Dataset):
    def __init__(self, gt_images_dir: str, lr_images_dir: str = None, upscale_factor: int = 4, transform = None):
        super().__init__()
        
        self.gt_images_dir = gt_images_dir
        self.lr_images_dir = lr_images_dir
        self.upscale_factor = upscale_factor
        self.transform = transform
        
        self.gt_image_filenames = os.listdir(gt_images_dir)
        if self.lr_images_dir is None:
            self.lr_images_dir = self.gt_images_dir
            self.lr_image_filenames = self.gt_image_filenames
        else:
            self.lr_image_filenames = os.listdir(lr_images_dir)
            
        assert len(self.lr_image_filenames) == len(self.gt_image_filenames)
        
    def __len__(self):
        
        return len(self.lr_image_filenames)
    
    def __getitem__(self, idx):
        gt_image_path = os.path.join(self.gt_images_dir, self.gt_image_filenames[idx])
        
        lr_image_path = os.path.join(self.lr_images_dir, self.lr_image_filenames[idx])

        gt_image = Image.open(gt_image_path)
        lr_image = Image.open(lr_image_path)

        gt_image = gt_image.resize((128, 128))
        lr_image = lr_image.resize((32, 32))                 
        
        if self.transform:
            gt_image = self.transform(gt_image)
            lr_image = self.transform(lr_image)
            
        return lr_image, gt_image