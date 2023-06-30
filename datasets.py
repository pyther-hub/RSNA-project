import os
from PIL import Image
import numpy as np
import torch
import torch.optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

TARGET = 'cancer'



def get_transforms(aug=False):
    def transforms(img):
        img = img.convert('RGB')
        
        if aug:
            tfm = [
                A.HorizontalFlip(p=0.5)
                A.VerticalFlip(p=0.5)
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8)
                A.OneOf([
                    A.RandomGamma(gamma_limit=(50, 150), p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5)
                ], p=0.5)
                A.CoarseDropout(max_height=8, max_width=8, p=0.5)
            ]
        else:
            tfm = [
                A.HorizontalFlip(p=0.5),
                A.Resize(height=1024, width=512)
            ]
        
        transform = A.Compose(tfm + [
            A.Normalize(mean=(0.2179, 0.2179, 0.2179), std=(0.0529, 0.0529, 0.0529)),
            ToTensorV2()
        ])
        
        augmented = transform(image=np.array(img))
        img = augmented['image']
        return img

    return lambda img: transforms(img)

class BreastCancerDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms
        self.each_row_weights=list(7*df.cancer+1)


    def __getitem__(self, i):

        path = f'{self.path}/{self.df.iloc[i].patient_id}/{self.df.iloc[i].image_id}.png'
        try:
            img = Image.open(path).convert('RGB')
        except Exception as ex:
            print(path, ex)
            return None

        if self.transforms is not None:
            img = self.transforms(img)


        if TARGET in self.df.columns:
            cancer_target = torch.as_tensor(self.df.iloc[i].cancer)
            return img, cancer_target

        return img

    def __len__(self):
        return len(self.df)