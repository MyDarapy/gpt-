Welcome to the **GPT-1 from Scratch with PyTorch** repository! This project is an educational endeavor to recreate the original GPT-1 model from scratch using PyTorch, albeit with fewer parameters to accommodate training on a modest GPU setup (first pre-trained on a T4 GPU and then on an A10G). The primary goal was to deepen my understanding of key concepts such as multi-head attention, tokenization, checkpointing, learning rate scheduling, and the intricacies involved in model pre-training from the ground up.

## Features

- **Custom Implementation**: Recreated GPT-1 architecture from scratch using PyTorch.
- **Multi-Head Attention**: Implemented based on the [GPT-1 paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
- **Tokenization**: Trained a custom tokenizer with HF tokenizer library tailored for the dataset.
- **Training Utilities**: Includes checkpointing, learning rate scheduling, and more.
- **Educational Focus**: Designed to serve as a learning resource for those interested in transformer-based models.

## Architecture

The model architecture closely follows the original GPT-1 design, consisting of:

- **Embedding Layer**: Converts input tokens into dense vectors.
- **Positional Encoding**: Adds postion information about the position of tokens in the sequence.
- **Eight Transformer Blocks** each containing:
  - Multiple Heads of Self-Attention
  - Feed-Forward Neural Networks
  - Layer Normalization and Residual Connections
- **Output Layer**: Generates logits (probabilities) for the next token prediction.

Despite having fewer parameters, the simplified architecture maintains the core functionalities of GPT-1, making it suitable for educational purposes and experimentation on limited hardware.

## How It Works

### Tokenization

Tokenization is the process of converting raw text into a sequence of tokens that the model can understand. This implementation uses a custom tokenizer trained with Hugging Face tokenizer library that builds a vocabulary based on the dataset, assigning unique integer IDs to each token. The vocabulary consists of 15k unique tokens. 

### Multi-Head Attention

Multi-Head Attention (MHA) is a critical component of the transformer architecture, allowing the model to focus on different parts of the input sequence simultaneously. This implementation follows the approach outlined in the [GPT-1 paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), involving:

- **Scaled Dot-Product Attention**: Computes attention scores between queries and keys, scaled by the square root of their dimension.
- **12 Attention Heads per block**: Splits the queries, keys, and values into multiple heads to capture diverse contextual relationships between words.
For each word (or token) in the input, the model generates three vectors:
**Query (Q):** Represents the current word’s perspective. 
**Key (K):** Represents how each word can be referenced.
**Value (V):** Represents the actual information each word holds.
I like to think of the Q, K, V as: **Query =** What you're searching for. **Key =** Labels that help identify where to find what you need. **Value:=** The actual information you retrieve once you find the right labels.

**Calculating Attention Scores:**
The model computes how much focus to place on other words by taking the dot product of the Query with all Keys. This determines the relevance of each word to the current word.
**Weighted Sum of Values:**
The attention scores are normalized (usually with softmax) to create weights. These weights are then used to compute a weighted sum of the Values, which becomes the output for that word.
**Multiple Heads:**
This process is done multiple times in parallel (hence "multi-head") to capture different types of relationships and patterns in the data.
This is very important for **context understanding**. Queries, Keys, and Values allow the model to understand the context by focusing on relevant parts of the input when generating each word.

**Parallel Processing:**
By using multiple heads, the model can attend to different aspects of the data simultaneously, making the learning process more efficient and comprehensive.

- **Concatenation and Linear Transformation**: Merges the outputs from all heads and projects them back to the original dimension.

### Positional Encoding

Since transformers lack inherent positional awareness, positional encodings are added to the token embeddings to provide information about the token positions within the sequence. This implementation uses learned positional embeddings.

### Transformer Blocks

The entire architecture has **12 transformer blocks** in total. Each transformer block comprises of 

1. **8 Multi-Head Self-Attention layers**: This allows the model to attend to different parts of the input sequence. 
2. **Feed-Forward Neural Network**: Processes the attention outputs through two linear layers with a GELU activation.
3. **Layer Normalization**: Applied before each sub-layer to stabilize and accelerate training.
4. **Residual Connections**: Adds the input of each sub-layer to its output to facilitate gradient flow.

### Model Hyperparameters
context_length = 512
pad_index = 0
EOS_token = 1
embed_dim = 768  # also called model dimension d_model (512 in the GPT 1 paper)
num_of_heads = 12 # Attention heads
dropout = 0.1  #regularization
batch_size = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_layers = 12 #Transformer block
eval_iters = 200
max_iter = 7000
evaluation_intervals = 200
learning_rate = 2.5e-4
vocab_size = 20_000

### Training Process

The model is trained to predict the next token in a sequence using the following steps:

1. **Data Preparation**: Tokenize the input text and create batches.  I used a batch size of 6 
2. **Forward Pass**: Compute the logits for each token in the sequence.
3. **Loss Calculation**: Use Cross-Entropy Loss to measure the discrepancy between predicted and actual tokens.
4. **Backward Pass**: Compute gradients and update model parameters using an optimizer with learning rate scheduling.
5. **Checkpointing**: Save model states at intervals to prevent loss of progress and facilitate resuming training.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- **Python**: 3.7 or higher
- **PyTorch**: 1.7.1 or higher
- **CUDA**: For GPU acceleration (optional but recommended)
- **Other Dependencies**: Listed in `requirements.txt`

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/MyDarapy/gpt-1-from-scratch.git
   cd gpt-1-from-scratch
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the GPT-1 model from scratch:

1. **Prepare Your Dataset**

   Ensure your dataset is in a suitable text format. You can place your dataset in the `data/` directory.

2. **Run the Training Script**

   ```bash
   python train.py --config config/train_config.yaml
   ```

   *Optional arguments can be specified in the configuration file or via command-line parameters.*

### Generating Text

After training, you can generate text using the trained model:

```bash
python generate.py --model_path checkpoints/model_epoch_X.pth --prompt "Once upon a time"
```

Replace `model_epoch_X.pth` with the path to your trained model checkpoint and provide a suitable prompt.

## Data Loader

The data loader is responsible for:

- **Loading the Dataset**: Reads the text data from the specified directory.
- **Tokenization**: Converts raw text into sequences of token IDs.
- **Batching**: Organizes data into batches for efficient training.
- **Shuffling**: Randomizes data order to improve training robustness.

Implementation details can be found in `data_loader.py`. It leverages PyTorch's `Dataset` and `DataLoader` classes to handle large datasets efficiently.

## Hyperparameters

Key hyperparameters for the model and training process include:

- **Model Parameters**
  - `vocab_size`: Size of the tokenizer vocabulary.
  - `embedding_dim`: Dimension of token embeddings.
  - `num_heads`: Number of attention heads in MHA.
  - `num_layers`: Number of transformer blocks.
  - `hidden_dim`: Dimension of the feed-forward network.
  - `max_seq_length`: Maximum sequence length.

- **Training Parameters**
  - `batch_size`: Number of samples per batch.
  - `learning_rate`: Initial learning rate for the optimizer.
  - `num_epochs`: Total number of training epochs.
  - `weight_decay`: Weight decay (L2 regularization) factor.
  - `dropout`: Dropout rate for regularization.
  - `gradient_clip`: Maximum gradient norm for clipping.

All hyperparameters are configurable via the `config/train_config.yaml` file.

## Project Structure

```
gpt-1-from-scratch/
├── data/
│   ├── raw/
│   └── processed/
├── checkpoints/
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── data_loader.py
│   ├── train.py
│   ├── generate.py
│   └── utils.py
├── config/
│   └── train_config.yaml
├── requirements.txt
├── README.md
└── LICENSE
```

- **data/**: Contains raw and processed datasets.
- **checkpoints/**: Stores model checkpoints during training.
- **src/**: Source code for the model, data loading, training, and generation scripts.
- **config/**: Configuration files for training and model parameters.
- **requirements.txt**: Python dependencies.
- **README.md**: Project documentation.
- **LICENSE**: License information.

## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [OpenAI GPT-1 Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [PyTorch](https://pytorch.org/)
- Inspired by the original GPT-1 architecture and various educational resources on transformer models.

---

*This repository is solely for educational purposes. All rights to the original GPT-1 model and associated materials belong to OpenAI.*
