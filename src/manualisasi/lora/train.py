import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import SimpleLlamma
from functools import partial
from model import LinearWithLoRA

# Configuration class
class Config:
    vocab_size = 2000
    e_dim = 100
    n_heads = 1
    n_layers = 1
    max_len = 512
    dropout = 0.1
    lr = 1e-3
    batch_size = 1  # Set to 1 batch
    n_epochs = 1    # Set to 1 epoch
    lora_r = 4
    lora_alpha = 1.0
    

# Initialize configuration and logging
config = Config()
logging.basicConfig(level=logging.INFO)

# Initialize the model
model = SimpleLlamma(
    config.vocab_size,
    config.e_dim,
    config.n_heads,
    config.n_layers,
    config.max_len,
    config.dropout
)

# Freezing all parameters first, except LoRA parameters
for param in model.parameters():
    param.requires_grad = False

assign_lora = partial(LinearWithLoRA, rank=config.lora_r, alpha=config.lora_alpha)

for layer in model.blocks:
    for head in layer.attn.heads:
        head.q = assign_lora(head.q)
        head.k = assign_lora(head.k)
        head.v = assign_lora(head.v)

# # Function to print the shapes of the inputs and outputs
# def hook_fn(module, input, output):
#     print(f"{module.__class__.__name__} - Input shape: {input[0].shape}, Output shape: {output.shape}")

# # Attach hooks to layers you want to inspect
# for name, layer in model.named_modules():
#     layer.register_forward_hook(hook_fn)

# # Register hooks to print gradients during backward pass
# def backward_hook(module, grad_input, grad_output):
#     print(f"{module.__class__.__name__} - grad_input: {grad_input}, grad_output: {grad_output}")

# for name, layer in model.named_modules():
#     layer.register_backward_hook(backward_hook)

# Dummy tokenizer class
class DummyTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, text):
        tokens = [ord(c) % self.vocab_size for c in text]
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(text).squeeze(0)
        if tokens.size(0) < self.max_len:
            padding = torch.zeros(self.max_len - tokens.size(0), dtype=torch.long)
            tokens = torch.cat([tokens, padding], dim=0)
        else:
            tokens = tokens[:self.max_len]
        
        x = tokens[:-1]  # Input sequence
        y = tokens[1:]   # Target sequence
        
        return x, y

# Collate function for DataLoader
def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    x_batch = torch.stack(x_batch)
    y_batch = torch.stack(y_batch)
    return x_batch, y_batch

# Create DataLoader function
def create_dataloader(texts, tokenizer, max_len, batch_size, shuffle=True):
    dataset = TextDataset(texts, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# Training step function
def train_step(model, x, y, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat.view(-1, y_hat.size(-1)), y.view(-1))
    print(f"Loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    return loss.item()

# Training function
def train(model, dataloader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for x_batch, y_batch in dataloader:
        train_step(model, x_batch, y_batch, optimizer, criterion)

# Main function to execute training
if __name__ == '__main__':
    texts = ["hello world"]
    tokenizer = DummyTokenizer(config.vocab_size)
    dataloader = create_dataloader(texts, tokenizer, config.max_len, config.batch_size)
    train(model, dataloader, config)
    # print(tokenizer(texts[0]))