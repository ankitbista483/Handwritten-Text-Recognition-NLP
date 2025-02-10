import json
import os
import re

class DatasetProcessor:
    def __init__(self):
        self.json_files = []
        self.preprocessed_json_files = []

    def clean_label(self, label, keep_punctuation=False):
        label = label.lower()
        if not keep_punctuation:
            label = re.sub(r'[^\w\s]', '', label)
        label = " ".join(label.split())
        return label

    def clean_processed_label(self, processed_label):
        return processed_label.strip() if processed_label else ""

    def preprocess_json_data(self, json_data, keep_punctuation=False):
        processed_data = []
        for entry in json_data:
            cleaned_label = self.clean_processed_label(entry.get('processed_label', ''))
            text_label = self.clean_label(entry.get('text', ''), keep_punctuation)
            entry['processed_label'] = cleaned_label
            entry['cleaned_text_label'] = text_label
            processed_data.append(entry)
        return processed_data

    def data_open(self, directory='dataset'):
        """Opens and processes all JSON files in the given directory."""
        self.json_files = []  # Reset the list before adding new files
        files = os.listdir(directory)
        for file in files:
            if file == '.DS_Store':
                continue
            file_path = os.path.join(directory, file)
            if os.path.isdir(file_path):
                json_paths = os.listdir(file_path)
                for json_name in json_paths:
                    if json_name.endswith('.json'):
                        self.json_files.append(os.path.join(file_path, json_name))
        return self.json_files
    
    def process_and_clean_json_files(self,  keep_punctuation=False):
        self.data_open()
        processed_files = []
        for json_file in self.json_files:
            
            with open(json_file, 'r') as f:
                file_content = f.read().strip()
                if not file_content:
                    #print(f"Skipping empty file: {json_file}")
                    continue
                data = json.loads(file_content)
        
            cleaned_data = self.preprocess_json_data(data, keep_punctuation)

            with open(json_file, 'w') as f:
                json.dump(cleaned_data, f, indent=4)
            processed_files.append(json_file)
        return processed_files
    
   



