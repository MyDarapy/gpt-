import torch 
import torch.nn as nn 
import config 
from data_loader import load_dataset
from decoder import GPTLanguageModel 


# Model evaluation code
@torch.no_grad()
def calculate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = load_dataset(split)
      logits, loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


# Function to save a checkpoint
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)




dataset = read_file_path(data)

train_data, test_data = split_data(dataset, train_data_size=0.8)
    
tokenizer = tokenizer(data_files, saving_filepath, vocab_size, min_frequency)

vocab = get_vocab(vocab_file_path)

train_data = tokenize_map_ids(train_data, tokenizer, vocab)
test_data = tokenize_map_ids(test_data, tokenizer, vocab)


train_dataloader = DataLoader(train_data, context_length, batch_size)
test_dataloader = DataLoader(test_data, context_length, batch_size)
train_iter = iter(train_dataset.load_dataset(split='train'))


model =SmolLM().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

max_iter = int(len(train_data) / batch_size)
lr_scheduler = GPTLearningRateScheduler(max_lr=3e-3, min_lr=0.0, warm_up_iters=2000, max_iters=max_iter)
learning_rate = lr_scheduler.get_lr(0)



optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
wandb.init(project='smol LLM Design Experiments')

for iter in range(max_iter):
  if iter % evaluation_intervals == 0 or iter == max_iter - 1:
    losses = calculate_loss()
    checkpoint = {
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss}
            
    save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{iter}.pth.tar")
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, Learning Rate: {lr_scheduler.get_lr(current_iter=iter)}")
    wandb.log({"train_loss": losses['train'], "val_loss": losses['val'], "learning_rate": lr_scheduler.get_lr(current_iter=iter)})

# Sample a batch of data
for x, y in train_iter:
  #Evaluate the loss, calculate gradient, update weight
  logits, loss = model(x, y)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  
  for params in optimizer.param_groups:
    params["lr"] = lr_scheduler.get_lr(current_iter=iter)