#!/usr/bin/env python3
# sentiment_analysis.py
import os
import re
import math
import pickle
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ===========================
# Data Loading and Preparation
# ===========================

def load_short_reviews(file_path, label):
    """
    Load short reviews from a text file, assigning a specified label and length category.

    Args:
        file_path (str): Path to the text file containing reviews.
        label (int): Label to assign to the reviews (e.g., 1 for positive, 0 for negative).

    Returns:
        tuple: Lists of reviews, labels, and length categories.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().strip()
    reviews = content.split()
    labels = [label] * len(reviews)
    lengths = ['short'] * len(reviews)
    return reviews, labels, lengths

def load_imdb_data(data_dir):
    """
    Load IMDB dataset from the specified directory.

    Args:
        data_dir (str): Directory containing 'pos' and 'neg' subdirectories with review files.

    Returns:
        tuple: Lists of review texts and corresponding labels.
    """
    texts = []
    labels = []
    for label in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, label)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(1 if label == 'pos' else 0)
    return texts, labels

def prepare_dataset(short_pos_path, short_neg_path, imdb_train_dir, imdb_test_dir):
    """
    Combine short and long reviews into a single DataFrame and perform a stratified train-test split.

    Args:
        short_pos_path (str): Path to short positive reviews file.
        short_neg_path (str): Path to short negative reviews file.
        imdb_train_dir (str): Directory of IMDB training data.
        imdb_test_dir (str): Directory of IMDB test data.

    Returns:
        tuple: Training and testing DataFrames.
    """
    # Load short reviews
    short_positive_reviews, short_positive_labels, short_positive_lengths = load_short_reviews(short_pos_path, 1)
    short_negative_reviews, short_negative_labels, short_negative_lengths = load_short_reviews(short_neg_path, 0)
    
    short_reviews = short_positive_reviews + short_negative_reviews
    short_labels = short_positive_labels + short_negative_labels
    short_lengths = short_positive_lengths + short_negative_lengths
    
    # Load long reviews
    train_texts, train_labels = load_imdb_data(os.path.join(imdb_train_dir))
    test_texts, test_labels = load_imdb_data(os.path.join(imdb_test_dir))
    
    long_reviews = train_texts + test_texts
    long_labels = train_labels + test_labels
    long_lengths = ['long'] * len(long_reviews)
    
    # Create DataFrames
    long_df = pd.DataFrame({
        'review': long_reviews,
        'label': long_labels,
        'length': long_lengths
    })
    
    short_df = pd.DataFrame({
        'review': short_reviews,
        'label': short_labels,
        'length': short_lengths
    })
    
    # Combine and split the dataset
    all_reviews_df = pd.concat([long_df, short_df], ignore_index=True)
    train_df, test_df = train_test_split(
        all_reviews_df,
        test_size=0.2,
        stratify=all_reviews_df[['label', 'length']],
        random_state=42
    )
    
    return train_df, test_df

# ===========================
# Model Definition
# ===========================

class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in the "Attention is All You Need" paper.
    Adds positional information to token embeddings.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to input embeddings.

        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Positionally encoded embeddings.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for binary sentiment classification.
    """
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, model_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(model_dim, 1)  # Binary classification output

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
        pooled = transformer_out.mean(dim=1)  # Global average pooling
        output = self.fc_out(self.dropout(pooled))
        return output.squeeze()

# ===========================
# Data Processing Utilities
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

def build_vocab(texts, vocab_size=10000):
    """
    Builds a vocabulary dictionary mapping words to unique indices.

    Args:
        texts (list): List of text samples.
        vocab_size (int): Maximum size of the vocabulary.

    Returns:
        dict: Word to index mapping.
    """
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    
    most_common = counter.most_common(vocab_size - 2)  # Reserve indices for PAD and UNK
    word2idx = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    return word2idx

def encode_text(text, word2idx, max_seq_len):
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
    indices = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    if len(indices) > max_seq_len:
        indices = indices[:max_seq_len]
    else:
        indices += [word2idx['<PAD>']] * (max_seq_len - len(indices))
    return indices

# ===========================
# Custom Dataset Classes
# ===========================

class IndexedTensorDataset(Dataset):
    """
    Custom Dataset that returns inputs, labels, and their original indices.
    Useful for tracking additional information during evaluation.
    """
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index], index
    
    def __len__(self):
        return len(self.inputs)

# ===========================
# Training and Evaluation
# ===========================

def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=20):
    """
    Trains the Transformer model.

    Args:
        model (nn.Module): The Transformer model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        scheduler (lr_scheduler): Learning rate scheduler.
        device (torch.device): Device to train on.
        num_epochs (int): Number of training epochs.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        # scheduler.step() # Updates learning rate, commented out for now
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

def evaluate_model(model, test_loader, device, test_df):
    """
    Evaluates the trained model on the test dataset and prints classification metrics.

    Args:
        model (nn.Module): The trained Transformer model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to perform evaluation on.
        test_df (DataFrame): DataFrame containing test data and metadata.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_lengths = []
    
    with torch.no_grad():
        for inputs, labels, indices in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs))
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Retrieve length categories
            batch_lengths = test_df.iloc[indices.numpy()]['length'].tolist()
            all_lengths.extend(batch_lengths)
    
    # Convert to NumPy arrays for metric calculations
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_lengths = np.array(all_lengths)
    
    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    target_names = ['Negative', 'Positive']
    print("\nOverall Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

# ===========================
# Main Execution
# ===========================

def main():
    # Paths to data files and directories
    SHORT_POS_PATH = 'positive-words.txt'
    SHORT_NEG_PATH = 'negative-words.txt'
    IMDB_TRAIN_DIR = 'aclImdb/train'
    IMDB_TEST_DIR = 'aclImdb/test'
    
    # Prepare datasets
    train_df, test_df = prepare_dataset(SHORT_POS_PATH, SHORT_NEG_PATH, IMDB_TRAIN_DIR, IMDB_TEST_DIR)
    
    # Hyperparameters
    VOCAB_SIZE = 10000
    MODEL_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    DROPOUT = 0.2
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 256
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    
    # Build vocabulary
    word2idx = build_vocab(train_df['review'].tolist(), vocab_size=VOCAB_SIZE)
    
    # Encode text data
    train_sequences = [encode_text(text, word2idx, MAX_SEQ_LEN) for text in train_df['review']]
    test_sequences = [encode_text(text, word2idx, MAX_SEQ_LEN) for text in test_df['review']]
    
    # Create tensors
    train_inputs = torch.LongTensor(train_sequences)
    train_labels_tensor = torch.FloatTensor(train_df['label'].tolist())
    test_inputs = torch.LongTensor(test_sequences)
    test_labels_tensor = torch.FloatTensor(test_df['label'].tolist())
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_inputs, train_labels_tensor)
    test_dataset = IndexedTensorDataset(test_inputs, test_labels_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = TransformerEncoder(
        input_dim=VOCAB_SIZE,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    # Define loss, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Set device based on availability: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS)
    
    # Evaluate the model
    evaluate_model(model, test_loader, device, test_df)
    
    # Save the model and vocabulary
    torch.save(model.state_dict(), 'sentiment_model.pt')
    with open('word2idx.pkl', 'wb') as f:
        pickle.dump(word2idx, f)
    print("\nModel and vocabulary have been saved successfully.")

if __name__ == "__main__":
    main()