import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.pipeline.model import Model
from src.pipeline.transform import TensorTransform
from src.component.json_cleaner import DatasetProcessor
from src.component.data_augmentation import DataTransformer

class HandwrittenDataset(Dataset):
    def __init__(self, images, labels_dict, transform=None):
        self.images = images
        self.labels_dict = labels_dict
        self.transform = transform

        # Match images with labels
        self.image_filenames = sorted(labels_dict.keys())  # Sort keys to match image order
        self.labels = [labels_dict[filename] for filename in self.image_filenames]

        # Ensure labels are not lists (flatten or handle multi-class labels)
        self.labels = [label if isinstance(label, (str, int)) else label[0] for label in self.labels]

        # Convert labels to numeric values
        unique_labels = list(set(self.labels))  # Now it's safe to create a set
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.numeric_labels = [self.label_to_idx[label] for label in self.labels]

        if len(self.images) != len(self.labels):
            raise ValueError(f"Mismatch: {len(self.images)} images but {len(self.labels)} labels.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.numeric_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Load and process labels
# Load labels first
processor = DatasetProcessor()
processor.data_open()
processor.process_json_files()
labels_dict = processor.load_labels()

# Convert images to tensors
tensor_transform = TensorTransform()
images = tensor_transform.convert_to_tensor()

# Create dataset & dataloader
dataset = HandwrittenDataset(images, labels_dict)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
