#!/usr/bin/env python3
# app.py

from flask import Flask, render_template, request
import torch
import torch.nn as nn
import math
import re
import pickle
import os

# Initialize the Flask application
app = Flask(__name__)

# ===========================
# Device Configuration
# ===========================

def get_device():
    """
    Determines the best available device (CUDA > MPS > CPU) for PyTorch operations.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

# ===========================
# Load Vocabulary
# ===========================

def load_vocab(vocab_path='word2idx.pkl'):
    """
    Loads the word-to-index mapping from a pickle file.

    Args:
        vocab_path (str): Path to the vocabulary pickle file.

    Returns:
        dict: A dictionary mapping words to unique indices.
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)
    return word2idx

word2idx = load_vocab()

# ===========================
# Model Definition
# ===========================

class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in the "Attention is All You Need" paper.
    Adds positional information to token embeddings.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the embeddings.
            dropout (float): Dropout rate.
            max_len (int): Maximum length of input sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough P matrix
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # Shape: [max_len, d_model]
        pe[:, 0::2] = torch.sin(position.float() * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position.float() * div_term)  # Apply cos to odd indices
        pe = pe.unsqueeze(1)  # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to input embeddings.

        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Positionally encoded embeddings.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for binary sentiment classification.
    """
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        """
        Initializes the TransformerEncoder module.

        Args:
            input_dim (int): Size of the vocabulary.
            model_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
        """
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # Ensures input shape is (batch_size, seq_len, model_dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(model_dim, 1)  # For binary classification

    def forward(self, src, src_mask=None):
        """
        Forward pass of the Transformer encoder.

        Args:
            src (Tensor): Input sequences of shape (batch_size, seq_len).
            src_mask (Tensor, optional): Mask for padded tokens.

        Returns:
            Tensor: Logits for each input in the batch.
        """
        embedded = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_encoder(embedded)
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=src_mask)
        pooled = transformer_out.mean(dim=1)  # Global average pooling over seq_len
        output = self.fc_out(self.dropout(pooled))
        return output.squeeze()

# ===========================
# Load Trained Model
# ===========================

def load_model(model_path='sentiment_model.pt', input_dim=10000, model_dim=256, num_heads=8, num_layers=4, dropout=0.2):
    """
    Loads the trained Transformer model from a file.

    Args:
        model_path (str): Path to the saved model state.
        input_dim (int): Size of the vocabulary.
        model_dim (int): Dimension of the embeddings.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of Transformer encoder layers.
        dropout (float): Dropout rate.

    Returns:
        nn.Module: The loaded Transformer model.
    """
    model = TransformerEncoder(
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ===========================
# Text Processing Utilities
# ===========================

def tokenize(text):
    """
    Tokenizes input text into lowercase words, removing non-alphanumeric characters.

    Args:
        text (str): The text to tokenize.

    Returns:
        list: List of tokenized words.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip().split()

def encode_text(text, word2idx, max_seq_len=256):
    """
    Encodes a text string into a list of integer indices based on the vocabulary.

    Args:
        text (str): The text to encode.
        word2idx (dict): Word to index mapping.
        max_seq_len (int): Maximum sequence length.

    Returns:
        list: List of encoded word indices.
    """
    tokens = tokenize(text)
    indices = [word2idx.get(token, word2idx.get('<UNK>', 1)) for token in tokens]
    if len(indices) > max_seq_len:
        indices = indices[:max_seq_len]
    else:
        indices += [word2idx.get('<PAD>', 0)] * (max_seq_len - len(indices))
    return indices

# ===========================
# Flask Routes
# ===========================

@app.route('/')
def index():
    """
    Renders the home page where users can submit their reviews.

    Returns:
        str: Rendered HTML template for the home page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction of sentiment based on user input.

    Returns:
        str: Rendered HTML template with the prediction result and confidence score.
    """
    review = request.form.get('review', '')
    if not review:
        return render_template('result.html', sentiment="No input provided.", confidence=None, review=review)
    
    # Encode the review text
    input_indices = encode_text(review, word2idx, max_seq_len=256)
    input_tensor = torch.LongTensor(input_indices).unsqueeze(0)  # Shape: [1, seq_len]
    
    with torch.no_grad():
        output = model(input_tensor.to(device))
        probability = torch.sigmoid(output).item()
        sentiment = 'Positive' if probability > 0.5 else 'Negative'
        confidence = probability if probability > 0.5 else 1 - probability
        confidence_percentage = round(confidence * 100, 2)
    
    return render_template('result.html', sentiment=sentiment, confidence=confidence_percentage, review=review)

# ===========================
# Main Execution
# ===========================

if __name__ == '__main__':
    # Ensure the app runs on the appropriate host and port
    # Set debug=False in production
    app.run(debug=True)
