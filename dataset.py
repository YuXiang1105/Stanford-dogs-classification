from torch.utils.data import Dataset
from PIL import Image
import io

class DogsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        #the format of the bytes is a dict, we extract the bytes key
        img_bytes = self.df.iloc[idx]['pixel_values']['bytes']
        #We convert to the RGB format
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        #We apply the transformations
        if self.transform:
            img = self.transform(img)
            
        label = self.df.iloc[idx]['label']
        #We return the iamge and the label, this is iterable by the dataloader
        return img, label