import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Embeddings(nn.Module):
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
        self.q = nn.Linear(e_dim, h_dim) # query projection
        self.k = nn.Linear(e_dim, h_dim) # key projection
        self.v = nn.Linear(e_dim, h_dim) # value projection
    
    def forward(self, x):
        B, T, E = x.size()
        q = self.q(x) # B, T, E -> B, T, H
        k = self.k(x) # B, T, E -> B, T, H
        v = self.v(x) # B, T, E -> B, T, H
        d_k = k.size(-1)
        
        scores = torch.bmm(q, k.transpose(-1, -2)) / d_k ** 0.5 # B, T, H @ B, H, T -> B, T, T
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, T, T) == 0
        scores = scores.masked_fill(mask, -float('inf'))
        weight = F.softmax(scores, dim=-1) # B, T, T, softmax over T
        x = torch.bmm(weight, v) # B, T, T @ B, T, H -> B, T, H
        
        return x

class FeedForwardNet(nn.Module):
    def __init__(self, e_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(e_dim, e_dim * 4)
        self.fc2 = nn.Linear(e_dim * 4, e_dim)
        self.gelu = nn.GELU() # TODO: replace with SwiGLU
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x): # B, T, E
        x = self.fc1(x) # B, T, E -> B, T, E * 4
        x = self.gelu(x) 
        x = self.fc2(x) # B, T, E * 4 -> B, T, E
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, e_dim, n_heads):
        super().__init__()
        self.h_dim = e_dim // n_heads # agar h_dim (cat) h_dim .. (cat) h_dim = e_dim
        self.heads = nn.ModuleList([
            SelfAttention(e_dim, self.h_dim) for _ in range(n_heads)
        ]) 
        self.out_fc = nn.Linear(e_dim, e_dim) # e_dim = n_heads * h_dim
        
    def forward(self, x):
        B, T, E = x.size()
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.out_fc(x)
        return x

class Block(nn.Module):
    def __init__(self, e_dim, n_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(e_dim, n_heads)
        self.ffn = FeedForwardNet(e_dim)
        self.rmsnorm1 = RMSNorm(e_dim)
        self.rmsnorm2 = RMSNorm(e_dim)
        
    def forward(self, x):
        x = x + self.attn(self.rmsnorm1(x))
        x = x + self.ffn(self.rmsnorm2(x))
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