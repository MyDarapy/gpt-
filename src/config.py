import argparse

class ConfigBase: 
    def __init__(self):
        self.name = argparse.Namespace()
        self.parser =argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        self.parser.add_argument("corpus_path", type=str, help="Add corpus file path. Must be a txt file")
        self.parser.add_argument("--tokenizer_path", type=str, default="./BPEtokenizer", help="Directory path to save the trained tokenizer. Defaults to './BPEtokenizer'.")
        self.parser.add_argument("--min_frequency", type=int, default=2, help="Minimum frequency a token must have to be included in the vocabulary. Default is 2.")
        self.parser.add_argument("--vocab_size", type=int, default=20000, help="Vocabulary size. Vocab is created with BPE. 40,000 merges in the original GPT-1 paper")
        self.parser.add_argument("--vocab_path", type=str, default="./BPEtokenizer/vocab.json", help="Directory path to save the trained tokenizer. Defaults to './BPEtokenizer/vocab.json'.")
        self.parser.add_argument("--batch_size", type=int, default=6, help="Enter batch_size")
        self.parser.add_argument('--learning_rate', type=float, default= 2.5e-4, help="Maximum learning rate")
        self.parser.add_argument("--embedding_dim", type= int, default=768, help="Embedding dimension also called model dimension. This argument has to be a multiple of the attention head.")
        self.parser.add_argument("--context_length", type=int, default=1024, help='Size of the context window that goes in as input. 512 was used in the original GPT-1 paper')
        self.parser.add_argument("--num_of_heads", type=int, default=12, help="number of attention heads per transformer block. This has to be a divisor of the embedding dimension")
        self.parser.add_argument("--n_layers", type=int, default=12, help="number of transfomer blocks in the architecture")
        self.parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability. Float value needs to be between 0.0 and 1.0")
        self.parser.add_argument('--epochs', type=int, default=5, help="Number of epochs during training. 100 epochs was used in the original paper.")
        self.parser.add_argument("--eval_iters", type=int, default=400, help="Evaluation iterations")
        self.parser.add_argument("--wandb_project", type=str, default="GPT_EXPERIMENTS", help="Project name logs on weights and biases")


    def _parse(self):
        self.arg=self.parser.parse_args()
        return self.arg

class Config(ConfigBase):
    def __init__(self):
        super().__init__()
        self.initialize()

    def parse(self):
        arg = self._parse()

        return arg

