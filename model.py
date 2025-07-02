import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
    

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
    


class FilterLayer(nn.Module):
    """
    The implementation of efficient filtering layer.
    """
    def __init__(self, d_model, dropout, seq_len=16, init_ratio=1):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, seq_len//2 + 1, d_model, 2, dtype=torch.float32) * init_ratio)
        self.out_dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)


    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        weight = torch.view_as_complex(self.complex_weight)

        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states

    

class DeepFilter(nn.Module):
    
    def __init__(self, args, featureNum, outputNum):

        super(DeepFilter, self).__init__()
        
        self.seq_len = args.window
        self.dimS = featureNum
        self.hidden = args.d_k
        self.hidR = args.hidR
        self.layerR = args.layerR
        self.outputNum = outputNum
        self.round = args.round
        self.init_ratio = args.init_ratio

        self.token_mixer = nn.ModuleList([FilterLayer(self.hidden, dropout=args.dropout, seq_len=self.seq_len, init_ratio=self.init_ratio) for i in range(self.round)])
        self.ffn = nn.ModuleList([PositionwiseFeedForward(self.hidden, self.hidden, dropout=args.dropout) for i in range(self.round)])
        self.mlp1 = nn.Linear(self.dimS, self.hidden)

        self.GRU = nn.GRU(self.hidden, self.hidR, num_layers=args.layerR)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear_out=nn.Linear(self.hidR, outputNum)

    def forward(self, input_tensor):

        output = self.mlp1(input_tensor)
        for round in range(self.round):
            output = self.token_mixer[round](output)
            output = self.ffn[round](output)
        
        output = output.transpose(0, 1).contiguous() # seq, batch, n
        _, output = self.GRU(output)
        output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
        out = self.linear_out(output)

        return out
    
