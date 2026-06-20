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
# Ensure this points to the latest epoch from your new training run!
CHECKPOINT_PATH = 'checkpoints/inktrace_epoch_25.pth' 

model = InkTraceModel(vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

# 3. Intelligent Image Processing
def process_image(image_path):
    # Load and convert to Grayscale
    img = Image.open(image_path).convert('L')
    
    # Binarization: Erase notebook lines (pixels lighter than 150 become pure white)
    # This leaves only the dark ink strokes
    img = img.point(lambda p: 255 if p > 100 else 0)
    
    # Aspect-Ratio Preserving Resize (Target Height: 64)
    w, h = img.size
    new_w = int(w * (64 / h))
    img = img.resize((new_w, 64), Image.Resampling.LANCZOS)
    
    # Pad to 256 width (or crop if it's too long)
    target_w = 256
    if new_w < target_w:
        # Pad right side with white (255) to prevent squashing
        pad_width = target_w - new_w
        img = ImageOps.expand(img, border=(0, 0, pad_width, 0), fill=255)
    else:
        # If the sequence is extremely long, crop it to fit the 256 window for now
        img = img.crop((0, 0, target_w, 64))
        
    # Standard Tensor Conversion & Normalization
    tensor = transforms.ToTensor()(img)
    tensor = transforms.Normalize((0.5,), (0.5,))(tensor)
    
    return tensor.unsqueeze(0).to(device)

# 4. Prediction Logic
def predict(image_path):
    img_tensor = process_image(image_path)
    
    with torch.no_grad():
        output = model(img_tensor) # Outputs log_softmax
        probs = torch.exp(output)  # Convert to standard probabilities
        
        # Take the highest probability index at each time step
        pred_indices = torch.argmax(probs, dim=2).squeeze(0)
        
    # X-Ray Diagnostic (See what the model thinks before collapsing)
    raw_frames = [idx2char.get(idx.item(), '?') if idx.item() != 0 else '-' for idx in pred_indices]
    print(f"\nModel X-Ray: {''.join(raw_frames)}")
        
    # Standard CTC Collapse
    decoded = []
    last_idx = -1
    for idx in pred_indices:
        idx = idx.item()
        # Append if it is NOT a blank token (0) AND NOT a consecutive duplicate
        if idx != 0 and idx != last_idx:
            decoded.append(idx)
        last_idx = idx
    
    return "".join([idx2char.get(i, '') for i in decoded])

# Run it!
print(f"\nFinal Prediction: {predict('test/test2.png')}\n")