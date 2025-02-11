import torch
import csv
from src.pipeline.label_dict import LabelDictionary

class LabelTensor:
    def __init__(self):
        self.vocab = {}
        self.label = LabelDictionary()
        self.label_dict = []

    def create_ascii(self):
   
        ascii_value_lower = [i for i in range(ord('a'), ord('z') + 1)]
        ascii_value_upper = [i for i in range(ord('A'), ord('Z') + 1)]
        ascii_value_digits = [i for i in range(ord('0'), ord('9') + 1)]

        lowercase = [chr(i) for i in ascii_value_lower]
        uppercase = [chr(i) for i in ascii_value_upper]
        digits = [chr(i) for i in ascii_value_digits]

    
        ascii_lower_dict = dict(zip(lowercase, ascii_value_lower))
        ascii_upper_dict = dict(zip(uppercase, ascii_value_upper))
        ascii_digits_dict = dict(zip(digits, ascii_value_digits))

       
        self.vocab = {**ascii_upper_dict, **ascii_lower_dict, **ascii_digits_dict}

        return self.vocab

    

    def load_label(self):
        return self.label.load_json_and_image()
    
    def __getitem__(self,idx):
        image_filename = list(self.load_label().keys())[idx]  
        self.label_dict= self.load_label()[image_filename]
        return self.label_dict
    

    def extract_image_labels(self):
        image_data_dict = self.load_label()  
        result = {}
        for image_name, data_entries in image_data_dict.items():
            cleaned_text_labels = [entry['cleaned_text_label'] for entry in data_entries]
            result[image_name] = cleaned_text_labels 
        return result

    def extract_polygon(self):
        image_data_dict = self.load_label()  
        result = {}
        for image_name, data_entries in image_data_dict.items():
            cleaned_text_labels = [entry['polygon'] for entry in data_entries]
            result[image_name] = cleaned_text_labels 
        return result

    
    def embedding_label(self):
        texts = self.extract_image_labels() 
        vocab = self.create_ascii()  
        encoded_data = {
            image_name: [[vocab.get(char, 0) for char in word] for word in words]
            for image_name, words in texts.items()
        }
     
        return encoded_data

    def polygon_tensor(self):
        data = self.extract_polygon()  
        
       
        tensor_data = {
            image_name: torch.tensor([box[key] for box in bounding_boxes for key in ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3']], dtype=torch.float32)
            for image_name, bounding_boxes in data.items()
        }

            # Prepare data for CSV writing
        csv_data = []
        for image_name, tensor in tensor_data.items():
            # Convert the tensor into a flat list of values
            flattened_values = tensor.tolist()
            # Append image_name to the beginning of the row
            row = [image_name] + flattened_values
            csv_data.append(row)

        # Define the header for CSV (image_name + the 8 coordinates of the bounding box)
        header = ['image_name', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3']

        # Write the data to a CSV file
        with open('bounding_boxes.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Write the header
            writer.writerows(csv_data)  # Write the rows

        return tensor_data

    def text_tensor(self):
        words = self.embedding_label()
        data ={
            image_name:[
                
                word + [0] * (39 - len(word)) if len(word) < 39 else word if word else [0] * 39
                for word in value
            ]
            for image_name, value in words.items()
            
        }

        tensor_data = {
            image_name: torch.tensor([word for word in word_list], dtype=torch.long)
            for image_name, word_list in data.items()
        }
        
            # Save the tensor data to a CSV file
        with open('tensor_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['image_name', 'tensor_data'])
            
            # Write the rows
            for image_name, tensor in tensor_data.items():
                # Convert tensor to a list and write to CSV
                writer.writerow([image_name, tensor.tolist()])

        return tensor_data

    def text_polygon_tensor(self):
        text = self.text_tensor()
        bbox = self.polygon_tensor()

        concatenated_dict = {
            key: torch.cat((text[key], bbox[key].unsqueeze(0)), dim=0)  
            for key in text.keys()  #
        }

        return concatenated_dict
    

j = LabelTensor()
j = j.text_tensor()