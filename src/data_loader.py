import pandas as pd
import torch
import json
import io
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageOps
import torchvision.transforms as transforms

class InkTraceDataset(Dataset):
    def __init__(self, parquet_file, vocab_path):
        self.data = pd.read_parquet(parquet_file)
        with open(vocab_path, 'r') as f:
            self.char2idx = json.load(f)
            
        # Only keep the tensor math here; we will handle resizing manually
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_binary = self.data.iloc[idx]['image.bytes']
        text_label = self.data.iloc[idx]['text']
        
        # 1. Aspect-Ratio Preserving Resize for TRAINING
        img = Image.open(io.BytesIO(img_binary)).convert('L')
        w, h = img.size
        new_w = int(w * (64 / h))
        img = img.resize((new_w, 64), Image.Resampling.LANCZOS)
        
        # 2. Pad to fixed 256 width for batching
        target_w = 256
        if new_w < target_w:
            pad_width = target_w - new_w
            img = ImageOps.expand(img, border=(0, 0, pad_width, 0), fill=255)
        else:
            img = img.crop((0, 0, target_w, 64))
            
        image_tensor = self.to_tensor(img)
            
        # 3. Label Processing
        label = []
        for char in str(text_label):
            idx = self.char2idx.get(char, -1)
            if idx > 0:
                label.append(idx)
        
        return image_tensor, torch.tensor(label, dtype=torch.long)
    
    # At the bottom of src/data_loader.py

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