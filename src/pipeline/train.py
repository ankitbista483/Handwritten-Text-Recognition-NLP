
from src.pipeline.transform import TensorTransform
from src.pipeline.label_dict import LabelDictionary


class HandwritingDataLoader():
    def __init__(self):
        self.image = TensorTransform()
        self.label = LabelDictionary()

    def tensor_image(self):
        return self.image.convert_to_tensor()   
    
    def dict_label(self):
        return self.label.load_json_and_image() 
    
    def __len__(self):
        return len(self.tensor_image())  
    
    def __getitem__(self, idx):
        image_tensor = self.tensor_image()[idx]
        image_filename = list(self.dict_label().keys())[idx]  
        label_dict = self.dict_label()[image_filename]  
        
        return image_tensor, label_dict


dataset = HandwritingDataLoader()
tensor= dataset.__len__()
print(tensor)




max_value = 571


