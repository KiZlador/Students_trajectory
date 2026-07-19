import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayerWithAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(
            src, src, src, 
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, attn_weights


class SASRec(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout=0.2):
        super().__init__()
        self.item_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithAttention(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, input_seq, mask=None):
        x = self.item_embedding(input_seq)
        x = self.dropout(x)
        x = self.pos_encoder(x)
        
        causal_mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device)).bool()
        causal_mask = ~causal_mask
        
        padding_mask = ~mask if mask is not None else None
        
        for layer in self.encoder_layers:
            x, _ = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask)
        
        x = self.ln_f(x)
        logits = F.linear(x, self.item_embedding.weight)
        return logits

    def get_session_embedding(self, input_seq, mask=None):
        x = self.item_embedding(input_seq)
        x = self.pos_encoder(x)
        
        causal_mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device)).bool()
        causal_mask = ~causal_mask
        padding_mask = ~mask if mask is not None else None
        
        for layer in self.encoder_layers:
            x, _ = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask)
        
        x = self.ln_f(x)
        last_token_embedding = x[:, -1, :]
        last_token_embedding = F.normalize(last_token_embedding, p=2, dim=1)
        
        return last_token_embedding

    def get_attention_weights(self, input_seq, mask=None):
        x = self.item_embedding(input_seq)
        x = self.pos_encoder(x)
        
        causal_mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device)).bool()
        causal_mask = ~causal_mask
        padding_mask = ~mask if mask is not None else None
        
        all_attention_weights = []
        
        for layer in self.encoder_layers:
            x, attn_weights = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask)
            all_attention_weights.append(attn_weights)
        
        x = self.ln_f(x)
        
        return torch.stack(all_attention_weights), x