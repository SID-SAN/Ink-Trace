import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import io
import torchvision.transforms as transforms

class InkTraceDataset(Dataset):
    def __init__(self, parquet_file, transform=None):
        self.data = pd.read_parquet(parquet_file)
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 256)), # Standardize input size
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
            # Extract the dictionary
            img_entry = self.data.iloc[idx]['image']
            text_label = self.data.iloc[idx]['text']
            
            # Access the binary data using the 'bytes' key
            img_binary = img_entry['bytes']
            
            # Convert binary to PIL Image
            image = Image.open(io.BytesIO(img_binary)).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, text_label