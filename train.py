import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_loader import InkTraceDataset, collate_fn
from src.model import InkTraceModel
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open('char_map.json', 'r') as f:
    vocab = json.load(f)

dataset = InkTraceDataset('data/your_file.parquet', 'char_map.json')
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
model = InkTraceModel(vocab_size=len(vocab) + 1).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CTCLoss(blank=0) # CTC is standard for OCR

images, labels = next(iter(loader))
images, labels = images.to(device), labels.to(device)

output = model(images)
print(f"Model output shape: {output.shape}") 