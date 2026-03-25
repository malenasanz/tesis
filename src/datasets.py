import os 
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CelebADataSet(Dataset):
    def __init__(self, img_dir, attr_path, transform=None, target_attr="Smiling"):

        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(attr_path, sep='\s+', header=1, index_col=0)
        df = (df + 1) // 2 

        self.df = df
        self.target_attr = target_attr 
        self.images = df.index.tolist()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = self.df.loc[img_name, self.target_attr]
        
        if self.transform:      
            image = self.transform(image)
        
        return image, label
