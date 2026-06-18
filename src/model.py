import torch
import torch.nn as nn

class InkTraceModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(InkTraceModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        
        # Input to LSTM is (batch, sequence_length, features)
        self.lstm = nn.LSTM(input_size=128 * 16, hidden_size=hidden_size, 
                            num_layers=2, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        # x shape: [batch, 1, 64, 256]
        x = self.cnn(x)
        
        # Reshape for LSTM: [batch, sequence_length (64), features (128 * 16)]
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x