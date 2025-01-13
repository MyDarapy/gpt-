from datasets import load_dataset

dataset = load_dataset("bookcorpus", split="train",trust_remote_code=True)


def clean_data(sentence):
  sentence = sentence.translate(str.maketrans('', '', string.punctuation))
  sentence = ''.join(char for char in sentence if not char.isdigit())
  return sentence

def process_dataset(dataset):
    return [clean_data(sentence['text']) for sentence in dataset]

dataset = dataset.shuffle(seed=65).select(range(10_000_000))
cl_data = process_dataset(dataset)

file_path = '/content/drive/MyDrive/bookcorpus.txt'
def write_list_to_file(arg.file_path, cl_data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in cl_data:
            f.write(f"{item}\n")  # Write each item on a new line

text = write_list_to_file(file_path, cl_data)