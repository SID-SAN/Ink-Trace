import pandas as pd
import torch
import json
import io
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torchvision.transforms as transforms

class InkTraceDataset(Dataset):
    def __init__(self, parquet_file, vocab_path, transform=None):
        self.data = pd.read_parquet(parquet_file)
        with open(vocab_path, 'r') as f:
            self.char2idx = json.load(f)
        
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Extract dictionary and binary data
        img_entry = self.data.iloc[idx]['image']
        text = self.data.iloc[idx]['text']
        
        # Convert binary to PIL Image
        image = Image.open(io.BytesIO(img_entry['bytes'])).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Convert string to indices
        label = [self.char2idx.get(char, 0) for char in text]
        
        return image, torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    """
    Handles variable length labels by padding them with 0 (<PAD>)
    """
    images, labels = zip(*batch)
    
    # Stack images (already fixed size from transform)
    images = torch.stack(images)
    
    # Pad sequences in the batch
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    
    return images, labels