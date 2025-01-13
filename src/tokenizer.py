from tokenizers import ByteLevelBPETokenizer
import json
import os
import sys
import typing

# Initialize a tokenizer
def tokenization(corpus_path: str, vocab_size:int, min_frequency:int, tokenizer_path:str):
    tokenizer = ByteLevelBPETokenizer()

    # Train the tokenizer on your dataset
    tokenizer.train(files = corpus_path, vocab_size, min_frequency, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

    tokenizer.save_model(tokenizer_path)
    return tokenizer

def get_vocab(vocab_path):
    if not os.path.exists(vocab_path):
        print("Vocab path does not exist")
        sys.exit(1)
    
    with open(vocab_path, 'r') as file:
        vocab = json.load(file)
    return vocab 

def encoder(example, tokenizer, vocab):
    tokens = tokenizer.encode(example).tokens #convert the strings into tokens
    token_ids = [vocab.get(token) for token in tokens] #convert the tokens into ids
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    return token_ids
