# DAT494 Lab 4: Basic Text Sequence Analysis with Recurrent Nets

## Overview
This lab focuses on applying recurrent neural networks (RNNs) for text sequence analysis. The key objectives include:

1. **Word Embeddings with Word2Vec CBOW** - Creating a vocabulary and training an embedding matrix.
2. **Text Classification** - Distinguishing between texts from Shakespeare and Tolstoy using LSTMs.
3. **Character Generation Model** - Training an autoregressive LSTM model to generate Shakespeare-like text.

We utilize **Long Short-Term Memory (LSTM) networks** for recurrent layers instead of GRUs, which were used in previous coursework.

## Features
- **Custom Tokenization & Vocabulary Building**: Cleans text, builds a word index, and applies thresholding.
- **CBOW Model Training**: Implements continuous bag-of-words model for word embedding training.
- **Text Classification with LSTMs**: Classifies sentences as Shakespeare or Tolstoy.
- **Character-Level Text Generation**: Uses an LSTM-based autoregressive model for text synthesis.
- **Performance Metrics & Visualization**: Evaluates accuracy, similarity scores, and generates 2D word embeddings with t-SNE.

## Repository Structure
├── DAT494_Lab4.pdf                # Lab instructions

├── DAT494_Lab4_Template.ipynb     # Jupyter Notebook for implementation

├── war_and_peace.txt              # War and Peace text dataset

├── shakespeare.txt                # Shakespeare text dataset

├── README.md                      # Project documentation

## Dependencies
Ensure you have the required Python libraries installed:
```sh
pip install numpy pandas matplotlib torch torchvision seaborn scikit-learn
```
## How to Use

1. **Clone the repository:**
   ```sh
    git clone https://github.com/your_username/DAT494_Lab4.git
    cd DAT494_Lab4
   ```
2. **Open the Jupyter Notebook and follow the step-by-step implementation:**
   ```sh
   jupyter notebook DAT494_Lab4_Template.ipynb
   ```
3. **Run sections in order, ensuring:**
- Tokenization and vocabulary construction are correct.
- CBOW word embeddings are trained properly.
- Classification and text generation models are fully trained.

4. **Evaluate and visualize results:**
- Test the text classification accuracy.
- Generate Shakespeare-style text sequences.

## Results & Findings

- **CBOW Model**: Produces meaningful word embeddings, allowing retrieval of semantically similar words.

- **Text Classification**: Achieves over 95% accuracy in distinguishing Shakespeare from Tolstoy.

- **Character-Level Text Generation**: Generates coherent and structured text resembling Shakespearean writing.

## Future Improvements

- Fine-tuning model hyperparameters for better classification accuracy.
- Expanding datasets for more diverse text classification.
- Implementing additional RNN architectures like Transformers for comparison.

## Contributors

- **Logan Pugliese** - [loganpugliese23@gmail.com](mailto:loganpugliese23@gmail.com)
