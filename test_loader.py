from src.data_loader import InkTraceDataset
from torch.utils.data import DataLoader

# Initialize dataset and loader
dataset = InkTraceDataset('data/train.parquet')
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Unpack the batch
images, labels = next(iter(loader))

print(f"Batch shape: {images.shape}")
print(f"Sample labels: {labels}")