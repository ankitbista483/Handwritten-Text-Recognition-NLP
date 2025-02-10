from transformers import BertTokenizer, BertModel
from src.pipeline.label_dict import LabelDictionary
import torch

class WordEmbedding:
    def __init__(self):
        # Load the pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.label = LabelDictionary()

    def load_label(self):
        return self.label.load_json_and_image()
    
    def get_word_embeddings(self, words):
        
        inputs = self.tokenizer(words, return_tensors="pt", padding=True, truncation=True)


        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Get the average of token embeddings
        
        return embeddings

    def extract_and_embed(self):
        image_data_dict = self.load_label()  # Assuming this loads the data
        result = []
      
        embedder = WordEmbedding()

        for image_name, data_entries in image_data_dict.items():
        
            cleaned_text_labels = [entry['cleaned_text_label'] for entry in data_entries]
        
            word_embeddings = embedder.get_word_embeddings(cleaned_text_labels)
        
            result.append({
                "image_name": image_name,
                "words": cleaned_text_labels,
                "embeddings": word_embeddings
            })
        
        return result




    
        




















