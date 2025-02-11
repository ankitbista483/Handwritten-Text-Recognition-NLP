import torch
import torch.nn as nn
import torch.nn.functional as F 


# class Model(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
#         self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
#         self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
#         self.conv4 = nn.Conv2d(128, 256, 5, padding=2)

#         self.pool = nn.MaxPool2d(2, 2)

#         self.lstm = nn.LSTM(input_size=256 * 14, hidden_size=128, num_layers=2, batch_first=True)

#         self.fc1 = nn.Linear(128, 1024) 
#         self.fc2 = nn.Linear(1024, 62) 

#     def forward(self, x):
#         # Pass through conv and pooling layers
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.pool(x)

#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.pool(x)

#         x = self.conv3(x)
#         x = F.relu(x)
#         x = self.pool(x)

#         x = self.conv4(x)
#         x = F.relu(x)
#         x = self.pool(x)

      
#         x = x.view(x.size(0), -1, 256 * 14)  
#         x, (hn, cn) = self.lstm(x) 
#         x = hn[-1]

      
#         x = self.fc1(x)
#         x = F.relu(x)

#         x = self.fc2(x)

#         return x



class Model(nn.Module):
    def __init__(self, use_bounding_box=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv4 = nn.Conv2d(128, 256, 5, padding=2)

        self.pool = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTM(input_size=256 * 14, hidden_size=128, num_layers=2, batch_first=True)

        self.fc1 = nn.Linear(128, 1024) 
        self.fc2 = nn.Linear(1024, 62)  # You can adjust the output size based on the number of possible characters or text tokens

        self.use_bounding_box = use_bounding_box

    def forward(self, image_tensor, word_tensor=None, bounding_box_tensor=None):
        # Pass through conv and pooling layers
        x = self.conv1(image_tensor)
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

        # Flatten the output for LSTM
        x = x.view(x.size(0), -1, 256 * 14)  
        x, (hn, cn) = self.lstm(x) 
        x = hn[-1]

        # Process the word tensor and bounding box tensor if provided
        if self.use_bounding_box:
            # Additional processing for bounding boxes, if you want to incorporate it into your LSTM (e.g., concatenate with word tensor or use it for a different task)
            # You can add a separate fully connected layer for bounding box prediction here

            # Example (dummy processing for bounding boxes)
            bbox_output = self.fc1(bounding_box_tensor)  # Adjust the layers as needed
            x = torch.cat((x, bbox_output), dim=1)  # Concatenate the LSTM output with bounding box output

        # Final fully connected layers for classification
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x
