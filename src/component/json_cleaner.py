import json
import os
import re

class DatasetProcessor:
    def __init__(self):
        self.json_files = []

    # Function to clean and normalize the label
    def clean_label(self, label,  keep_punctuation=False):
        label = label.lower()  # Normalize to lowercase (can adjust based on model requirements)
        if not keep_punctuation:
            label = re.sub(r'[^\w\s]', '', label)  # Remove punctuation if not needed
        label = " ".join(label.split())  # Normalize whitespace by removing extra spaces
        return label

    # Function to remove excessive padding from processed_label
    def clean_processed_label(self, processed_label):
        return processed_label.strip()  # Remove leading and trailing spaces

    # Function to preprocess the JSON data (cleaning labels, etc.)
    def preprocess_json_data(self, json_data, keep_punctuation=False):
        processed_data = []
        for entry in json_data:
            # Check if 'processed_label' exists in the entry, and clean it if it does
            if 'processed_label' in entry:
                cleaned_label = self.clean_processed_label(entry['processed_label'])
            else:
                cleaned_label = ''  # Handle case when 'processed_label' is missing

            # Clean the actual text label using your cleaning function
            text_label = self.clean_label(entry['text'], keep_punctuation)

            # Replace the 'processed_label' with the cleaned text label
            entry['processed_label'] = cleaned_label

            # Optional: Add the cleaned text label to the entry
            entry['cleaned_text_label'] = text_label

            processed_data.append(entry)

        return processed_data

    #Function to open and process all JSON files in the dataset directories
    def data_open(self):
        current_directory = 'dataset'  # Dataset directory
        files = os.listdir(current_directory)
        for file in files:
            if file == '.DS_Store':
                continue  # Skip macOS system files
            file_path = os.path.join(current_directory, file)
            if os.path.isdir(file_path):
                json_paths = os.listdir(file_path)
                for json_name in json_paths:
                    json_path = os.path.join(file_path, json_name)
                    if json_name.endswith('.json'):  # Check if the file is a JSON file
                        self.json_files.append(json_path)  # Append the path to the JSON file

    # Function to process and clean the JSON files
    def process_json_files(self):
        for json_file in self.json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Clean the data
            cleaned_data = self.preprocess_json_data(data)

            # Save the cleaned data back into the same JSON file
            with open(json_file, 'w') as f:
                json.dump(cleaned_data, f, indent=4)

            print(f"Processed and cleaned: {json_file}")



processor = DatasetProcessor()


processor.data_open()


processor.process_json_files()
