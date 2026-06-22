import torch
import json
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from src.model import InkTraceModel

# 1. Setup & Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('char_map.json', 'r') as f:
    vocab = json.load(f)
idx2char = {v: k for k, v in vocab.items()}

# 2. Load Model
CHECKPOINT_PATH = 'checkpoints/inktrace_epoch_50.pth' 

model = InkTraceModel(vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

# 3. Intelligent Image Processing
def process_image(image_path):
    img = Image.open(image_path).convert('L')

    w, h = img.size
    new_w = int(w * (64 / h))
    img = img.resize((new_w, 64), Image.Resampling.LANCZOS)
    
    target_w = 512
    if new_w < target_w:
        pad_width = target_w - new_w
        img = ImageOps.expand(img, border=(0, 0, pad_width, 0), fill=255)
    else:
        img = img.crop((0, 0, target_w, 64))
        
    tensor = transforms.ToTensor()(img)
    tensor = transforms.Normalize((0.5,), (0.5,))(tensor)
    
    return tensor.unsqueeze(0).to(device)

# 4. Prediction Logic
def predict(image_path):
    img_tensor = process_image(image_path)
    
    with torch.no_grad():
        output = model(img_tensor)

    pred_indices = torch.argmax(output, dim=2).squeeze(0)

    decoded = []
    last_idx = -1

    for idx in pred_indices:
        idx = idx.item()
        if idx != 0 and idx != last_idx:
            decoded.append(idx)
        last_idx = idx

    print("Decoded indices:", decoded)
    print("Decoded chars:", [idx2char.get(i, '?') for i in decoded])
        
    pred_indices = torch.argmax(output, dim=2).squeeze(0)
    
    decoded = []
    last_idx = -1
    for idx in pred_indices:
        idx = idx.item()
        if idx != 0 and idx != last_idx:
            decoded.append(idx)
        last_idx = idx
    
    result = "".join([idx2char.get(i, '?') for i in decoded])
    result = result.strip()
    result = result.rstrip('.,;:!?"\'')
    return result.upper()

print(f"\nFinal Prediction: {predict('test/test4.png')}\n")
