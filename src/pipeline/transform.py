from torchvision import transforms
from src.component.data_augmentation import DataTransformer



class TorchTransform:
    def __init__(self):
        self.load_to_tensor = transforms.ToTensor()
        self.load_to_PIL = transforms.ToPILImage()

    def load_clean_data(self):
        images  = DataTransformer.transform_images()
        return images

    def convert_to_tensor(self):
        img = self.load_clean_data()
        img = [self.load_to_tensor(self.load_to_PIL(img)) for img in img]
        return img







