import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        x = x.float()
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return self.scale * x

class  Embeddings(nn.Module):
    def __init__(self, vocab_size, e_dim, max_len=512, dropout=0.1):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, e_dim)
        self.position_embedding = nn.Embedding(max_len, e_dim)
        self.rmsnorm = RMSNorm(e_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T = x.size() # B: batch size, T: sequence length
        positions = torch.arange(T, device=x.device).expand(B, T) # B, T
        x = self.token_embedding(x) + self.position_embedding(positions) # B, T, E + B, T, E -> B, T, E
        x = self.rmsnorm(x) # B, T, E
        x = self.dropout(x) # B, T, E
        return x

class SelfAttention(nn.Module):
    def __init__(self, e_dim, h_dim):
        super(SelfAttention, self).__init__()
        self.q = nn.Linear(e_dim, h_dim)  # query projection
        self.k = nn.Linear(e_dim, h_dim)  # key projection
        self.v = nn.Linear(e_dim, h_dim)  # value projection
    
    def forward(self, x):
        B, T, E = x.size()
        print(f"SelfAttention Input shape: {x.shape}")

        q = self.q(x)  # B, T, E -> B, T, H
        k = self.k(x)  # B, T, E -> B, T, H
        v = self.v(x)  # B, T, E -> B, T, H
        
        d_k = k.size(-1)
        
        scores = torch.bmm(q, k.transpose(-1, -2)) / d_k ** 0.5  # B, T, H @ B, H, T -> B, T, T
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, T, T) == 0
        scores = scores.masked_fill(mask, -float('inf'))
        weight = F.softmax(scores, dim=-1)  # B, T, T, softmax over T
        x = torch.bmm(weight, v)  # B, T, T @ B, T, H -> B, T, H
        
        print(f"SelfAttention Output shape: {x.shape}")
        print(f"Sample Attention Weights: {weight[0]}")
        
        return x


class FeedForwardNet(nn.Module):
    def __init__(self, e_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(e_dim, e_dim * 4)
        self.fc2 = nn.Linear(e_dim * 4, e_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        print(f"FeedForwardNet Input shape: {x.shape}")
        
        # B, T, E -> B, T, E * 4
        x = self.fc1(x)  
        x = self.gelu(x)
        
        # B, T, E * 4 -> B, T, E
        x = self.fc2(x)  
        x = self.dropout(x)
        
        print(f"FeedForwardNet Output shape: {x.shape}")
        print(f"Sample Output: {x[0]}")
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, e_dim, n_heads):
        super().__init__()
        self.h_dim = e_dim // n_heads
        self.heads = nn.ModuleList([
            SelfAttention(e_dim, self.h_dim) for _ in range(n_heads)
        ]) 
        self.out_fc = nn.Linear(e_dim, e_dim)
        
    def forward(self, x):
        print(f"MultiHeadAttention Input shape: {x.shape}")

        B, T, E = x.size()
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.out_fc(x)
        
        print(f"MultiHeadAttention Output shape: {x.shape}")
        print(f"Sample Output: {x[0]}")
        
        return x


class Block(nn.Module):
    def __init__(self, e_dim, n_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(e_dim, n_heads)
        self.ffn = FeedForwardNet(e_dim)
        self.rmsnorm1 = RMSNorm(e_dim)
        self.rmsnorm2 = RMSNorm(e_dim)
        
    def forward(self, x): # B, T, E
        print(f"Block Input shape: {x.shape}")
        x = x + self.attn(self.rmsnorm1(x)) # B, T, E
        x = x + self.ffn(self.rmsnorm2(x)) # B, T, E
        print(f"Block Output shape: {x.shape}")
        return x


class SimpleLlamma(nn.Module):
    def __init__(self, vocab_size, e_dim, n_heads, n_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, e_dim, max_len, dropout)
        self.blocks = nn.ModuleList([
            Block(e_dim, n_heads) for _ in range(n_layers)
        ])
        self.rmsnorm = RMSNorm(e_dim)
        self.fc = nn.Linear(e_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x): # B, T
        x = self.embeddings(x) # B, T, E
        for block in self.blocks: 
            x = block(x) # B, T, E
        x = self.rmsnorm(x)
        x = self.fc(x) # B, T, E -> B, T, V
        x = self.softmax(x)
        return x