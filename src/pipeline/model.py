import torch
import torch.nn as nn
import torch.nn.functional as F 


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv4 = nn.Conv2d(128, 256, 5, padding=2)

        self.pool = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTM(input_size=256 * 14, hidden_size=128, num_layers=2, batch_first=True)

        self.fc1 = nn.Linear(128, 1024) 
        self.fc2 = nn.Linear(1024, 62) 

    def forward(self, x):
        # Pass through conv and pooling layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)

      
        x = x.view(x.size(0), -1, 256 * 14)  
        x, (hn, cn) = self.lstm(x) 
        x = hn[-1]

      
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


