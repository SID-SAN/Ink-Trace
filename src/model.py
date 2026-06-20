import torch
import torch.nn as nn

class InkTraceModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(InkTraceModel, self).__init__()
        
        self.cnn = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), 
                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1))
                )
        
        self.lstm = nn.LSTM(input_size=2048 , hidden_size=hidden_size, 
                            num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
            x = self.cnn(x)
            b, c, h, w = x.size()
            x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
            x, _ = self.lstm(x)
            x = self.fc(x)
            return torch.nn.functional.log_softmax(x, dim=2)