from transformer import GPTLanguageModel
from tokenizer import tokenization
import torch
import torch.nn as nn
from config import Config 
import wandb
from lr_scheduler import GPTLearningRateScheduler
import data_loader

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def main(arg):
    model = GPTLanguageModel(
        embed_dim=arg.embedding_dim, context_length=arg.context_length, num_of_heads=arg.num_of_heads, 
        vocab_size=arg.vocab_size, n_layers=arg.n_layers, dropout=arg.dropout).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    
    tokenizers = tokenization(corpus_path=arg.corpus_path, vocab_size=arg.vocab_size, min_frequency=arg.minimum_frequency, tokenizer_path=arg.tokenizer_path)
    vocab = tokenizer.get_vocab(arg.vocab_path)

    data = data_loader.read_file(arg.corpus_path)
    data = data_loader.encoder(data, tokenizers, vocab)

    train_dataset, test_dataset = data_loader.split_data(data, 0.8)
    
    X, y = data_loader.get_batch('train', arg.batch_size)

    max_iter = int(len(train_dataset) / arg.batch_size)

    lr_scheduler = GPTLearningRateScheduler(max_lr=arg.learning_rate, min_lr=0.0, warm_up_iters=2000, max_iters=max_iter)

    learning_rate = lr_scheduler.get_lr(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    wandb.init(project=arg.wandb_project)

    eval_iters = arg.eval_iters
    for iter in range(max_iter):
        if iter % arg.eval_iters == 0 or iter == max_iter - 1:
            losses = calculate_loss()
            checkpoint = {
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses}
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{iter}.pth.tar")
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, Learning Rate: {lr_scheduler.get_lr(current_iter=iter)}")
            wandb.log({"train_loss": losses['train'], "val_loss": losses['val'], "learning_rate": lr_scheduler.get_lr(current_iter=iter)})

        # Sample a batch of data
        X, y = data_loader.get_batch('train', arg.batch_size)

        #evaluate the loss, calculate gradient, update weight
        logits, loss = model(X, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


        for params in optimizer.param_groups:
            params["lr"] = lr_scheduler.get_lr(current_iter=iter)

    # Model evaluation code
    @torch.no_grad()
    def calculate_loss():
        out = {}
        model.eval()
        for split in ['train', 'test']:
            losses = torch.zeros(arg.eval_iters)
            for k in range(arg.eval_iters):
                X, Y = data_loader.get_batch(split, arg.batch_size)
                logits, loss = model(X,Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


    # Function to save a checkpoint
    def save_checkpoint(state, filename="checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)

if __name__ == "__main__":
    config = Config()
    arg = config.parse()
    main(arg)












