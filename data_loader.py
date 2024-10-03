def split_data(data, train_data_size):
  n = int(0.8*len(data))
  train_dataset = data[:n]
  test_dataset = data[n:]
  return train_dataset, test_dataset

train_dataset, test_dataset = split_data(data, 0.8)

def load_dataset(split):
  data = train_dataset if split == 'train' else test_dataset
  idx = torch.randint((len(data)-context_length), (batch_size,)) #idx has size of {batch_size}
  x = torch.stack([data[i:i+context_length] for i in idx])
  y = torch.stack([data[i+1:i+context_length+1] for i in idx]) #The target this includes the next token we are trying to predict
  x,y = x.to(device), y.to(device)
  return x, y

xb,yb = load_dataset('train')
