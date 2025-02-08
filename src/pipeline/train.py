
from src.pipeline.transform import TensorTransform
from src.pipeline.label_dict import LabelDictionary


class HandwritingDataLoader():
    def __init__(self):
        self.image = TensorTransform()
        self.label = LabelDictionary()
        self.images = self.image.convert_to_tensor()  
        self.labels = self.label.load_json_and_image()  

    def tensor_image(self):
        return self.images  
    
    def dict_label(self):
        return self.labels  
    
    def __len__(self):
        return len(self.images)  
    
    def __getitem__(self, idx):
        image_tensor = self.images[idx]
        image_filename = list(self.labels.keys())[idx]  
        label_dict = self.labels[image_filename]  
        
        return image_tensor, label_dict

dataset = HandwritingDataLoader()
image_tensor, label_dict = dataset[0]
print(image_tensor)  
print(label_dict) 

        



        