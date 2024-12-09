import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import SimpleLlamma

# Configuration class
class Config:
    vocab_size = 2000
    e_dim = 100
    n_heads = 10
    n_layers = 6
    max_len = 512
    dropout = 0.1
    lr = 1e-3
    batch_size = 32
    n_epochs = 10

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

# Dummy tokenizer class
class DummyTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, text):
        tokens = [ord(c) for c in text]
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
    loss.backward()
    optimizer.step()
    return loss.item()

# Training function
def train(model, dataloader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_history = []

    for epoch in range(config.n_epochs):
        epoch_start_time = time.time()
        total_loss = 0.0

        for x_batch, y_batch in dataloader:
            loss = train_step(model, x_batch, y_batch, optimizer, criterion)
            total_loss += loss

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        logging.info(f'Epoch {epoch+1}/{config.n_epochs} completed in {epoch_duration:.2f}s, Average Loss: {avg_loss:.4f}')

    return loss_history

def generate_text(model, tokenizer, prompt, max_len=100, temperature=1.0):
    model.eval()
    tokens = tokenizer(prompt).squeeze(0) 
    tokens = tokens.unsqueeze(0)
    
    generated_tokens = tokens
    for _ in range(max_len):
        with torch.no_grad():
            output = model(generated_tokens)
            logits = output[:, -1, :]  # Get logits for the last token
            logits = logits / temperature  
            probabilities = torch.softmax(logits, dim=-1) 
            
            next_token = torch.multinomial(probabilities, num_samples=1)
            
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            
            if next_token.item() == 0:
                break
    
    # Convert tokens to text
    generated_text = ''.join([chr(t) for t in generated_tokens.squeeze().tolist()])
    return generated_text


# Main function to execute training
if __name__ == '__main__':
    texts = ["hello world", "this is a test", "autoregressive model"]
    tokenizer = DummyTokenizer(config.vocab_size)
    dataloader = create_dataloader(texts, tokenizer, config.max_len, config.batch_size)
    loss_history = train(model, dataloader, config)

    # Generate text
    prompt = "this is a "
    generated_text = generate_text(model, tokenizer, prompt, max_len=50)
    print("Generated Text:", generated_text)