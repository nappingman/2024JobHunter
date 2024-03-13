import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from einops import rearrange

def get_padding_mask(seq, padding_idx):
    # unsqueeze for multi head (maybe
    padding_mask = seq.ne(padding_idx)
    padding_mask = padding_mask.unsqueeze(1)
    return padding_mask * padding_mask.transpose(1,2)

def get_subsequent_mask(seq):
    batch_size, seq_len = seq.size()
    # need a mask to mask future-element,
    # which is the element on diagonal and below it to one
    seq_mask = (1 - torch.ones(seq_len, seq_len).triu(diagonal=1)).bool().unsqueeze(0)
    return seq_mask

class SinusoidalPE(nn.Module):
    def __init__(self, 
                 embed_dim=512,
                 theta=10000,
                 n_position=1024,
                 ):
        super().__init__()
        self.register_buffer('pe_table', self._get_sinusoidal_table(theta, n_position, embed_dim))
        
    def _get_sinusoidal_table(self, theta, n_position, embed_dim):
        def _get_theta_table(self, pos):
            return [pos / np.power(theta, 2*(i//2)/embed_dim) for i in range(embed_dim)]

        sinusoidal_table = np.array([_get_theta_table(pos, embed_dim) for pos in range(n_position)])
        sinusoidal_table[:, 0::2] = np.sin(sinusoidal_table[:, 0::2])
        sinusoidal_table[:, 1::2] = np.cos(sinusoidal_table[:, 1::2])
        
        # add a batch dimension
        return torch.FloatTensor(sinusoidal_table).unsqueeze(0)
    
    def forward(self, x):
        # return x add pe
        return x + self.pe_table[:, :x.size(1), :].clone().detach()
         
class FeedForward(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, in_dim)
        
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_dim, 1e-6)
        
        self.feedforward = lambda x : self.drop(self.fc2(self.act(self.fc1(x))))
        
    def forward(self, x):
        residual = x
        x = self.feedforward(x)
        x += residual
        x = self.layer_norm(x)
        
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 num_heads=8,
                 embed_dim=512,
                 dropout=0.1,
                 max_length=2048,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "Error"
        
        # self.c_attn = nn.Linear(embed_dim, embed_dim * 3)
        self.w_qs = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_ks = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_vs = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.c_proj = nn.Linear(embed_dim, embed_dim) 
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
        self.layer_norm = nn.LayerNorm(embed_dim, 1e-6)
                
    def forward(self, q, k, v, attn_mask=None):
        B, T, C = q.shape   # batch size, seq length, feature dim
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)
        residual = q
        
        # spilt to multi heads 
        q = rearrange(q, 'b l (n d) -> b l n d', n=self.num_heads).transpose(1, 2)
        k = rearrange(k, 'b l (n d) -> b l n d', n=self.num_heads).transpose(1, 2)
        v = rearrange(v, 'b l (n d) -> b l n d', n=self.num_heads).transpose(1, 2)
                
        # calculate attn score
        scale = self.embed_dim ** -0.5
        
        # [batch, num_head, seq_len, dim] @ [batch, num_head, dim, seq_len]
        # -> [batch, num_head, seq_len, seq_len]
        attn_matrix = (q @ k.transpose(-2, -1)) / scale
        
        
        if attn_mask is not None:
            attn_matrix = attn_matrix.masked_fill(attn_mask[:, None, :T, :T]==0, float('-inf'))

        attn_matrix = nn.functional.softmax(attn_matrix, dim=-1)
        attn_matrix = self.attn_drop(attn_matrix)
        
        # [batch, num_head, seq_len, seq_len] @ [batch, num_head, seq_len, dim]
        # -> [batch, num_head, seq_len, dim]
        output = attn_matrix @ v
        
        # combine all heads
        output = rearrange(output, 'b n l d -> b l (n d)')

        output = self.layer_norm(self.proj_drop(self.c_proj(output)) + residual)
        
        return output, attn_matrix

class EncoderLayer(nn.Module):
    def __init__(self,
                 num_heads=8,
                 embed_dim=512,
                 dropout=0.1,):
      super().__init__()
      self.ff = FeedForward(embed_dim, 4 * embed_dim, dropout)
      self.attn = MultiHeadSelfAttention(num_heads=num_heads,
                                         embed_dim=embed_dim,
                                         max_length=2048,
                                         dropout=dropout)
      
    def forward(self, x, attn_mask=None):
        y, attn = self.attn(q=x, k=x, v=x, attn_mask=attn_mask)
        y = self.ff(y)
        
        return y, attn
      
class DecoderLayer(nn.Module):
    def __init__(self,
                 num_heads=8,
                 embed_dim=512,
                 dropout=0.1,):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(num_heads=num_heads,
                                                embed_dim=embed_dim,
                                                max_length=2048,
                                                dropout=dropout)
        self.cross_attn = MultiHeadSelfAttention(num_heads=num_heads,
                                                embed_dim=embed_dim,
                                                max_length=2048,
                                                dropout=dropout)
        self.ff = FeedForward(embed_dim, embed_dim * 4, dropout)
        
        self.fc = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, tar, memo, self_attn_mask, cross_attn_mask=None):
        y, attn_self = self.self_attn(q=tar, k=tar, v=tar, attn_mask=self_attn_mask)
        y, attn_cross = self.cross_attn(q=memo, k=memo, v=y, attn_mask=self_attn_mask)
        
        y = self.fc(y)
        
        return y, attn_self, attn_cross

class Encoder(nn.Module):
    def __init__(self,
                 depth=4,
                 num_heads=8,
                 embed_dim=512,
                 dropout=0.1,
                 max_length=2048,
                 padding_idx=0,
                 num_words=10000):
        super().__init__()
        self.word_embed = nn.Embedding(num_words, embed_dim, padding_idx)
        self.position_embed = SinusoidalPE(embed_dim=embed_dim, 
                                           theta=10000,
                                           n_position=max_length)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(num_heads, embed_dim, dropout) for _ in range(depth)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim, 1e-6)
    
    def forward(self, x, attn_mask):
        B, T = x.shape
        x_emebdding = self.word_embed(x)
        x = self.drop(self.position_embed(x_emebdding))
        x = self.layer_norm(x)
        
        for layer in self.layers:
            x, _ = layer(x, attn_mask)
        return x
        
class Decoder(nn.Module):
    def __init__(self,
                 depth=8,
                 embed_dim=512,
                 num_heads=8,
                 dropout=0.1,
                 max_length=2048,
                 num_words=10000,
                 padding_idx=0,
                 ):
        super().__init__()
        self.word_embed = nn.Embedding(num_words, embed_dim, padding_idx)
        self.position_embed = SinusoidalPE(embed_dim=embed_dim,
                                           theta=10000,
                                           n_position=max_length)
        self.layers = nn.ModuleList([
            DecoderLayer(num_heads, embed_dim, dropout) for _ in range(depth)
        ])
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim, 1e-6)
        
    def forward(self, tar_seq, memo, tar_mask, attn_mask=None):
        B, T = tar_seq.shape
        x_embedding = self.word_embed(tar_seq)
        x = self.drop(self.position_embed(x_embedding))
        x = self.layer_norm(x)
        for layer in self.layers:
            x, _, _ = layer(x, memo, 
                            self_attn_mask=tar_mask, 
                            cross_attn_mask=attn_mask)
        
        return x

class Model(nn.Module):
    def __init__(self,
                 encoder_depth=4,
                 decoder_depth=8,
                 num_heads=8,
                 embed_dim=512,
                 dropout=0.1,
                 max_length=2048,
                 num_words=1000,
                 padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.encoder = Encoder(depth=encoder_depth,
                               num_heads=num_heads,
                               embed_dim=embed_dim,
                               dropout=dropout,
                               max_length=max_length,
                               num_words=num_words,
                               padding_idx=padding_idx,
                               )
        self.decoder = Decoder(depth=decoder_depth,
                               num_heads=num_heads,
                               embed_dim=embed_dim,
                               dropout=dropout,
                               max_length=max_length,
                               num_words=num_words,
                               padding_idx=padding_idx
                               )
        self.head = nn.Linear(embed_dim, num_words)
        
        # xavier init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
        
        if True:
            self.head.weight = self.decoder.word_embed.weight

        if True:
            self.encoder.word_embed.weight = self.decoder.word_embed.weight
        
    def forward(self, src_seq, tar_seq):
        src_mask = get_padding_mask(src_seq, self.padding_idx)
        tar_mask = get_padding_mask(tar_seq, self.padding_idx) & get_subsequent_mask(tar_seq)
                
        encode_vec = self.encoder(src_seq, src_mask)
        decode_vec = self.decoder(tar_seq, encode_vec, tar_mask, src_mask)
        
        seq_logit = self.head(decode_vec)
        return seq_logit.view(-1, seq_logit.size(2))

    
if __name__ == '__main__':
    batch = 2
    seq_len = 100
    num_words = 200
    max_length = 100
    padding_idx = -1
    
    x = torch.randint(0, num_words, (batch, seq_len)).long()
    y = torch.randint(0, num_words, (batch, seq_len)).long()
    print(f"x.shape = {x.shape}, x.max = {x.max()}, x.min = {x.min()}, x.dtype = {x.dtype}")
    
    
    encoder_depth = 4
    decoder_depth = 8
    num_heads = 8
    embed_dim = 512
    dropout = 0.1    
    model = Model(encoder_depth,
                  decoder_depth,
                  num_heads,
                  embed_dim,
                  dropout,
                  max_length,
                  num_words,
                  padding_idx)
    
    logit = model(x, y)
    p = nn.functional.softmax(logit, dim=1)
    print(f'logit shape = {logit.shape}, p shape = {p.shape}, p[0,0,:].sum() = {p[0,:].sum()}')