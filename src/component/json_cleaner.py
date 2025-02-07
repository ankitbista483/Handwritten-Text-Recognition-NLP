import json
import os
import re

class DatasetProcessor:
    def __init__(self):
        self.json_files = []

    
    def clean_label(self, label,  keep_punctuation=False):
        label = label.lower()  
        if not keep_punctuation:
            label = re.sub(r'[^\w\s]', '', label)  
        label = " ".join(label.split())  
        return label

    
    def clean_processed_label(self, processed_label):
        return processed_label.strip()  

    
    def preprocess_json_data(self, json_data, keep_punctuation=False):
        processed_data = []
        for entry in json_data:
            
            if 'processed_label' in entry:
                cleaned_label = self.clean_processed_label(entry['processed_label'])
            else:
                cleaned_label = ''  

          
            text_label = self.clean_label(entry['text'], keep_punctuation)

            
            entry['processed_label'] = cleaned_label

            
            entry['cleaned_text_label'] = text_label

            processed_data.append(entry)

        return processed_data

    
    def data_open(self):
        current_directory = 'dataset'  
        files = os.listdir(current_directory)
        for file in files:
            if file == '.DS_Store':
                continue  
            file_path = os.path.join(current_directory, file)
            if os.path.isdir(file_path):
                json_paths = os.listdir(file_path)
                for json_name in json_paths:
                    json_path = os.path.join(file_path, json_name)
                    if json_name.endswith('.json'):  
                        self.json_files.append(json_path)  

    
    def process_json_files(self):
        
        processed_files = []
        for json_file in self.json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            
            cleaned_data = self.preprocess_json_data(data)

           
            with open(json_file, 'w') as f:
                json.dump(cleaned_data, f, indent=4)

            processed_files.append(json_file)  
        
        return processed_files  


    def load_labels(self):
        json_paths = self.process_json_files() 

        labels_dict = {} 
        for json_path in json_paths:
            with open(json_path, "r") as f:
                labels_dict[json_path] = json.load(f)  
        return labels_dict
       



