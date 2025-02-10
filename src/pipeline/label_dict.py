import json
import os
from src.component.json_cleaner import DatasetProcessor
from src.pipeline.transform import TensorTransform

class LabelDictionary:
    def __init__(self):
        self.file = DatasetProcessor()
        self.image = TensorTransform()
    
    def load_processed_json(self):
        return self.file.process_and_clean_json_files()
    
    def load_augmented_image(self):
         return self.image.convert_to_tensor()
    
    def load_json_and_image(self):
        file_paths = self.load_processed_json()  
        # print("File paths:", file_paths) 
        result = {}

        for file in file_paths:
            with open(file, 'r') as json_file:
                data = json.load(json_file)

            base_name = os.path.splitext(os.path.basename(file))[0]
            image_path = base_name + ".jpg"  # Create image path
            
            result[os.path.basename(image_path)] = data

        #print("Loaded Labels Dictionary:")
        #print(result)

        return result

        


