from tokenizers import ByteLevelBPETokenizer
import json

# Initialize a tokenizer
def tokenizer(data_files, saving_filepath, vocab_size, min_frequency):
    tokenizer = ByteLevelBPETokenizer()

    # Train the tokenizer on your dataset
    tokenizer.train(files = data_files, vocab_size, min_frequency, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    ])

    # Save the trained tokenizer
    tokenizer.save_model('/content/drive/MyDrive/BPEtokenizer')
    print('tokenizer successfully trained')


#Get the vocabulary created by the tokenizer
def get_vocab(vocab_file_path):
    vocab_file_path = vocab_file_path
    with open(vocab_file_path, 'r') as file:
        vocab = json.load(file)
    return vocab 
