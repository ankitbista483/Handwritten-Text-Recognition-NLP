import torch
import torch.nn as nn

# Sample JSON data
labels = [
    {
        "text": "Language",
        "polygon": {"x0": 438, "y0": 462, "x1": 1131, "y1": 473, "x2": 1200, "y2": 692, "x3": 404, "y3": 762},
        "line_idx": 0,
        "type": "H",
        "processed_label": "",
        "cleaned_text_label": "language"
    },
    {
        "text": "for",
        "polygon": {"x0": 1240, "y0": 375, "x1": 1483, "y1": 358, "x2": 1506, "y2": 542, "x3": 1240, "y3": 640},
        "line_idx": 0,
        "type": "H",
        "processed_label": "",
        "cleaned_text_label": "for"
    }
]

# Your ASCII dictionaries for word-to-index conversion (can be extended for larger vocab)
vocab = {'language': 0, 'for': 1}
vocab_size = len(vocab)
embedding_dim = 10  # Dimensionality of the word embeddings

# Create an embedding layer
embedding = nn.Embedding(vocab_size, embedding_dim)

# Function to extract bounding box and convert to tensor
def extract_data(item):
    word = item['cleaned_text_label']
    # Convert word to index
    word_idx = vocab.get(word, -1)  # -1 if word not found in vocab
    # Convert word to tensor (embedding vector)
    word_tensor = embedding(torch.tensor([word_idx])) if word_idx != -1 else torch.zeros(embedding_dim)
    
    # Extract bounding box coordinates (polygon)
    bbox_coords = item['polygon']
    bbox_tensor = torch.tensor([
        bbox_coords['x0'], bbox_coords['y0'],
        bbox_coords['x1'], bbox_coords['y1'],
        bbox_coords['x2'], bbox_coords['y2'],
        bbox_coords['x3'], bbox_coords['y3']
    ], dtype=torch.float32)
    
    return word_tensor, bbox_tensor

# Process all the items
word_tensors = []
bbox_tensors = []

for item in labels:
    word_tensor, bbox_tensor = extract_data(item)
    word_tensors.append(word_tensor)
    bbox_tensors.append(bbox_tensor)

# Convert the list of tensors to a single tensor if needed
word_tensors = torch.stack(word_tensors)
bbox_tensors = torch.stack(bbox_tensors)

print("Word Tensors:", word_tensors)
print("Bounding Box Tensors:", bbox_tensors)
