
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


from src.pipeline.transform import TensorTransform
from src.pipeline.Label_tensor import LabelTensor
from src.pipeline.model import Model


# class HandwritingDataLoader():
#     def __init__(self):
#         self.image = TensorTransform()
#         self.label = LabelTensor()

#     def tensor_image(self):
#         return self.image.convert_to_tensor()   
    
#     def dict_label(self):
#         word_tensor = self.label.text_tensor()
#         Label_tensor = self.label.polygon_tensor()

#         return word_tensor, LabelTensor
        
    
#     def __len__(self):
#         return len(self.tensor_image())  
    
#     def __getitem__(self, idx):
#         image_tensor = self.tensor_image()[idx]
#         image_filename = list(self.dict_label().keys())[idx]  
#         label_dict = self.dict_label()[image_filename]  
        
#         return image_tensor, label_dict

import pandas as pd
import torch

class HandwritingDataLoader():
    def __init__(self, word_data_path, bbox_data_path):
        # Load the CSV files into DataFrames
        self.word_data = pd.read_csv(word_data_path)
        self.bbox_data = pd.read_csv(bbox_data_path)
        
        # Assuming word_data and bbox_data are in the correct format
        # Convert the DataFrames into dictionaries or other structures
        self.image_filenames = self.word_data['image_filename'].unique()
        
    def tensor_image(self):
        # Your existing method to convert image data into tensors
        # Assuming TensorTransform handles the image-to-tensor conversion
        return self.image.convert_to_tensor() 

    def dict_label(self):
        # Map image filenames to word and bounding box tensors
        labels = {}
        for image_filename in self.image_filenames:
            word_tensor = torch.tensor(self.word_data[self.word_data['image_filename'] == image_filename]['word'].values)
            bbox_tensor = torch.tensor(self.bbox_data[self.bbox_data['image_filename'] == image_filename].iloc[0, 1:].values)
            labels[image_filename] = (word_tensor, bbox_tensor)
        
        return labels

    def __len__(self):
        # Return the number of samples (images)
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        
        # Retrieve corresponding word tensor (words) and bounding box tensor
        word_tensor, bounding_box_tensor = self.dict_label()[image_filename]
        
        # You would need to load the actual image here (image_tensor) 
        # and return image_tensor, word_tensor, bounding_box_tensor
        # For now, I'm assuming you return the image tensor directly
        image_tensor = self.tensor_image()  # Make sure to load the image
        
        return image_tensor, word_tensor, bounding_box_tensor


word_data = '/Users/ankitbista/Desktop/practice/Handwritten Text Recoginition-NLP/tensor_data.csv'
bbox_data = '/Users/ankitbista/Desktop/practice/Handwritten Text Recoginition-NLP/bounding_boxes.csv'
dataset = HandwritingDataLoader(word_data, bbox_data)

# Create DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = Model(use_bounding_box=True)  # Set use_bounding_box to True when training with bounding box data

# Example for training with both image and bounding box tensors
# Assuming you have a model, optimizer, and dataloader ready
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer
text_loss_fn = nn.CrossEntropyLoss()  # For text classification
bbox_loss_fn = nn.MSELoss()  # For bounding box regression

# Example for training with both image and bounding box tensors
for image_tensor, word_tensor, bounding_box_tensor in dataloader:
    optimizer.zero_grad()  # Clear gradients from previous step
    
    # Forward pass
    output = model(image_tensor, word_tensor, bounding_box_tensor)
    
    # Separate the losses (for both text recognition and bounding box prediction)
    text_output = output[:, :-4]  # Assuming last 4 columns are for bounding boxes
    bbox_output = output[:, -4:]  # Last 4 columns for bounding box (x, y, w, h)
    
    # Compute text loss
    text_loss = text_loss_fn(text_output, word_tensor)
    
    # Compute bounding box loss
    bbox_loss = bbox_loss_fn(bbox_output, bounding_box_tensor)
    
    # Total loss
    total_loss = text_loss + bbox_loss  # You can scale these losses by a weight factor
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()  # Update the weights
    
    # Optionally print loss for monitoring
    print(f"Loss: {total_loss.item()} (Text Loss: {text_loss.item()}, Bounding Box Loss: {bbox_loss.item()})")
