from torchvision import transforms
from src.component.data_cleaning import DataCleaner
import numpy as np

class TensorTransform:
    def __init__(self):
        self.load_to_tensor = transforms.ToTensor()
        self.load_to_PIL = transforms.ToPILImage()

    def load_clean_data(self):
        images = DataCleaner.preprocess_images_parallel()
        return images

    def convert_to_tensor(self):
        img = self.load_clean_data()
        img_tensors = [self.load_to_tensor(self.load_to_PIL(image)) for image in img]
        
        # Print the size of each tensor
        # for i, tensor in enumerate(img_tensors):
        #     print(f"Tensor {i} size: {tensor.size()}")  # or tensor.shape()
        
        return img

  




j = TensorTransform()
j = j.convert_to_tensor()
print(j)