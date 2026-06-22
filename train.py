import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_loader import InkTraceDataset, collate_fn
from src.model import InkTraceModel
import json
import os

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open('char_map.json', 'r') as f:
    vocab = json.load(f)

dataset = InkTraceDataset('data/train.parquet', 'char_map.json')
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
model = InkTraceModel(vocab_size=len(vocab)).to(device)

# 2. Training Config
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

epochs = 50
os.makedirs('checkpoints', exist_ok=True)

# 3. Production Training Loop
model.train()
print("Starting Production Training Loop...")

for epoch in range(epochs):
    total_loss = 0
    valid_batches = 0
    
    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        output = model(images)
        output = output.permute(1, 0, 2)
        
        time_steps = output.size(0)
        target_lengths = torch.sum(labels != 0, dim=1)
        valid_indices = (target_lengths > 0) & (target_lengths <= time_steps)
        
        if not valid_indices.any():
            continue 
            
        output = output[:, valid_indices, :]
        labels = labels[valid_indices]
        target_lengths = target_lengths[valid_indices]
        
        batch_size = labels.size(0)
        input_lengths = torch.full(size=(batch_size,), fill_value=time_steps, dtype=torch.long)
        
        loss = criterion(output, labels, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        valid_batches += 1
        
        if step % 50 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Current Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\n=== Epoch {epoch+1}/{epochs} Completed | Average Loss: {avg_loss:.4f} | LR: {current_lr:.6f} ===\n")
    scheduler.step(avg_loss)

    
    torch.save(model.state_dict(), f'checkpoints/inktrace_epoch_{epoch+1}.pth')

print("Training Complete!")