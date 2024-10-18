# Review Analyser with Self-Attention

This project implements a **Transformer-based sentiment analysis model** for movie reviews, drawing directly from the concepts presented in the seminal paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). The core focus of this project is to re-implement the transformer architecture and understand how **self-attention mechanisms** work in natural language processing tasks like sentiment classification.

To provide an intuitive interface, the model is deployed via a **Flask** web application where users can input movie reviews and receive sentiment predictions alongside a confidence score.

## Live Demo

You can access the live demo of the application here:

[Review Analyser with Self-Attention on PythonAnywhere](https://ankannn10.pythonanywhere.com)

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Understanding the Transformer](#understanding-the-transformer)
- [Data and Evaluation](#data-and-evaluation)
  - [Dataset](#dataset)
  - [Downloading the IMDB Dataset](#downloading-the-imdb-dataset)
  - [Evaluation Results](#evaluation-results)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

The main objective of this project is to build a **sentiment analysis system** using **transformers** and **self-attention mechanisms**, inspired by the "Attention is All You Need" paper. Transformers are a breakthrough in NLP and have proven to excel at tasks like text classification, language modeling, translation, and more.

This project implements a **binary sentiment classification** task (positive/negative) using a **custom Transformer model** trained on the **IMDB dataset** and some additional short movie reviews. The results are presented in a Flask-based web interface.

### Motivation

The project is designed to:
- **Showcase understanding of the transformer architecture**: Focusing on self-attention mechanisms for sentiment classification.
- **Provide a real-world use case**: Allowing users to analyze movie reviews via a web app interface.
- **Experiment with training and evaluating transformers** on varied data lengths (short and long reviews) to compare performance.

## Key Features

- **Transformer-Based Sentiment Classification**: Utilizes the self-attention mechanism to classify movie reviews as positive or negative.
- **Flask Web App**: A user-friendly interface for inputting reviews and viewing predictions.
- **Confidence Scores**: Displays how confident the model is in its prediction.
- **Custom Transformer Model**: Implements positional encoding and multi-head attention layers.
- **Detailed Evaluation Metrics**: Provides accuracy, precision, recall, and F1-scores based on review length (short vs. long reviews).

## Understanding the Transformer

### Self-Attention Mechanism

The **self-attention** mechanism is central to the transformer architecture, enabling the model to weigh the importance of different words in a sentence relative to each other, regardless of their position. This is in contrast to traditional RNNs or LSTMs, where relationships between words could weaken due to the sequential nature of processing.

The formula used for self-attention is:

$\[
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
\]$

Where:
- \( Q \), \( K \), and \( V \) are matrices representing the input queries, keys, and values.
- \( d_k \) is the dimension of the key vectors.

In this project, self-attention helps the model focus on important words within a review, enabling more accurate sentiment classification.

### Positional Encoding

Transformers do not inherently understand word order since they do not process inputs sequentially. To introduce positional information, we use **Positional Encoding**, which adds a unique representation for each position in the sequence to the word embeddings. This helps the model understand the order of the words in a review.

### Transformer Architecture

The **TransformerEncoder** class in this project mimics the encoder block from the original paper:
- **Multi-Head Attention** layers allow the model to attend to multiple aspects of the review simultaneously.
- **Feed-Forward Neural Networks** process the attention outputs.
- **Global Average Pooling** condenses the output, and a final linear layer produces the sentiment prediction.

## Data and Evaluation

### Dataset

The model is trained on a combination of:
1. **IMDB Movie Reviews Dataset** (long reviews): This consists of 25,000 movie reviews labeled as positive or negative.
2. **Short Reviews Dataset**: Custom-made short reviews were added to observe how the model handles shorter text inputs.

### Downloading the IMDB Dataset

You can download the IMDB Long Reviews dataset directly using the `wget` Python library. Use the following code to download and extract the dataset:

```python
import wget

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
wget.download(url, "aclImdb_v1.tar.gz")

# After downloading, you can extract the contents using tar or any archiving tool.
```

This will download the **IMDB dataset** to your working directory. You can then extract it using any archive tool (`tar` or similar) to use it in your training process.

### Evaluation Results

#### Overall Classification Performance

The model achieves an overall accuracy of **83.76%** on the test dataset, with the following performance across positive and negative reviews:

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.82      | 0.88   | 0.85     | 5957    |
| Positive  | 0.86      | 0.79   | 0.82     | 5401    |
| **Accuracy** | **0.84** | | | **11358** |

#### Performance by Review Length

##### Long Reviews (High Confidence):

- **Accuracy**: 86%
- **Confusion Matrix**:

  ```
  [[4401  599]
   [ 815 4185]]
  ```

- The model performs well on longer reviews, as it has more context to analyze and understand the sentiment.

##### Short Reviews (Lower Confidence):

- **Accuracy**: 68%
- **Confusion Matrix**:

  ```
  [[865  92]
   [338  63]]
  ```

- The model struggles with shorter reviews as they provide less information, leading to lower precision and recall for positive reviews.

The disparity between short and long reviews aligns with expectations. Transformers generally excel with larger context windows, and long reviews provide more useful information for the model to make accurate predictions.

## Technology Stack

- **Backend**:
  - [Flask](https://flask.palletsprojects.com/): For handling web requests and rendering templates.
  - [PyTorch](https://pytorch.org/): For building and training the transformer model.
- **Frontend**:
  - HTML/CSS with [Bootstrap](https://getbootstrap.com/): For building the UI.
- **Deployment**:
  - [PythonAnywhere](https://www.pythonanywhere.com/): The app is hosted on PythonAnywhere for public access.

## Project Structure

```
Review-Analyser-with-Self-Attention/
├── app.py                      # Flask app for serving the model
├── model.py                    # Model definition and training script
├── sentiment_model.pt          # Pre-trained PyTorch model (tracked with Git LFS)
├── word2idx.pkl                # Vocabulary mapping
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── templates/
│   ├── index.html              # Webpage for submitting reviews
│   └── result.html             # Webpage for displaying results
└── static/
    └── styles.css              # Custom CSS for styling
```

## Setup Instructions

### Prerequisites

- **Python 3.7 or higher**: Required for running the code.
- **Git**: For version control.
- **Virtual Environment (Optional)**: Recommended for managing dependencies.

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ankannn10/Review-Analyser-with-Self-Attention.git
   cd Review-Analyser-with-Self-Attention
   ```

2. **Set up the virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**:

   ```bash
   python app.py
   ```

5. **Access the app**:

   Open your browser and navigate to `http://127.0.0.1:5000/`.

### Running the App

- The app provides a simple interface where users can input a movie review and get a sentiment prediction.
- The app displays both the sentiment (positive/negative) and the **confidence score

** for each prediction.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- **"Attention is All You Need" Paper**: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) – The foundation for transformers and self-attention mechanisms.
- **Flask and PyTorch Communities**: For extensive documentation and community support.
- **PythonAnywhere**: For providing a platform to host this app.
- **IMDB Dataset**: For providing the long movie reviews dataset.

---
