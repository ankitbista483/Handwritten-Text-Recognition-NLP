import torch
import torch.nn as nn
from src.pipeline.label_dict import LabelDictionary

class LabelTensor:
    def __init__(self):
        self.vocab = {}
        self.label = LabelDictionary()
        self.label_dict = []

    def create_ascii(self):
        ascii_value_lower = [i for i in range(ord('a'), ord('z') + 1)]
        ascii_value_upper = [i for i in range(ord('A'), ord('Z') + 1)]

        lowercase = [chr(i) for i in ascii_value_lower]
        uppercase = [chr(i) for i in ascii_value_upper]

        ascii_lower_dict = dict(zip(lowercase, ascii_value_lower))
        ascii_upper_dict = dict(zip(uppercase, ascii_value_upper))

        self.vocab= {**ascii_upper_dict, **ascii_lower_dict}

        return self.vocab
    

    def load_label(self):
        return self.label.load_json_and_image()
    
    def __getitem__(self,idx):
        image_filename = list(self.load_label().keys())[idx]  
        self.label_dict= self.load_label()[image_filename]
        return self.label_dict
    

    def extract_image_labels(self):
        image_data_dict = self.load_label()
        result = []
        
        for image_name, data_entries in image_data_dict.items():
            cleaned_text_labels = [entry['cleaned_text_label'] for entry in data_entries]
            polygon = [entry['polygon'] for entry in data_entries]
            combined_labels_and_polygons = [
            f"{label}: {polygon}" for label, polygon in zip(cleaned_text_labels, polygon)]
            labels_string = ', '.join(combined_labels_and_polygons)
            result.append(f"{image_name}[{labels_string}]")
        
        return result
    
    def embedding_label(self):
        texts = self.extract_image_labels()
        vocab = self.create_ascii()
        result = []

        for text in texts:
            label_string = text.split('[')[1].split(']')[0]
            label_string = label_string.split(', ')
            label_embed = [vocab.get(char,0) for char in text]
            image_name = text.split('[')[0]
            result.append({image_name: label_embed})
        return texts

    def embedding_label(self):
        texts = self.extract_image_labels()  # Extract the list of texts
        vocab = self.create_ascii()  # Create the vocabulary for embedding
        result = []

        for text in texts:
            # Split the text to extract the image name and bounding box labels
            parts = text.split('[:')
            if len(parts) < 2:
                continue  # Skip if the format is incorrect
            
            image_name = parts[0].strip()  # Image name (before [:)
            label_data = parts[1].strip().rstrip(']')  # Extract label data (between [: and ]])

            # Now split the label_data into individual word-label pairs
            label_entries = label_data.split(',')  # Split by commas to separate words and coordinates
            
            labels = []
            for entry in label_entries:
                # Extract the word and coordinates
                parts = entry.split(': {')
                if len(parts) < 2:
                    continue  # Skip if the entry format is incorrect
                
                word = parts[0].strip()  # Extract word
                coordinates = parts[1].strip().rstrip('}')  # Extract coordinates part, removing closing }

                # You can process coordinates further if needed
                coord_dict = {}
                for coord in coordinates.split(','):
                    key, value = coord.split(':')
                    coord_dict[key.strip()] = int(value.strip())
                
                # Embed the word using the vocabulary
                label_embed = [vocab.get(word, 0)]  # Using the vocabulary to embed the word

                # Append the processed data to the result
                labels.append({
                    'word': word,
                    'coordinates': coord_dict,
                    'embedding': label_embed
                })

            # Append the image name and its labels with embeddings
            result.append({
                'image_name': image_name,
                'labels': labels
            })

        return result


    
jpt = LabelTensor()
texts = jpt.embedding_label()
print(texts[0])