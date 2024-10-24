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


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_hidden, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_hidden // n_head
        self.w_qs = nn.Linear(d_model, d_hidden)
        self.w_ks = nn.Linear(d_model, d_hidden)
        self.w_vs = nn.Linear(d_model, d_hidden)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_k, 0.2))
        self.layer_norm = nn.LayerNorm(d_hidden)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):

        d_k, n_head = self.d_k, self.n_head

        bs, seq_len, _ = x.size()
        # residual = self.res(q)
        q = self.w_qs(x).view(bs, seq_len, n_head, d_k)
        k = self.w_ks(x).view(bs, seq_len, n_head, d_k)
        v = self.w_vs(x).view(bs, seq_len, n_head, d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k) # (n*b) x lv x dv

        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, bs, seq_len, d_k)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, seq_len, -1) # b x lq x (n*dv)
        output = self.dropout(output)
        output = self.layer_norm(output + x)

        return output, attn


class ProbAttention(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = 5
        self.scale = None
        self.mask_flag = True
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.output_attention = True
        self.dropout = nn.Dropout(dropout)
        self.w_qs = nn.Linear(d_model, d_hidden)
        self.w_ks = nn.Linear(d_model, d_hidden)
        self.w_vs = nn.Linear(d_model, d_hidden)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_hidden)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_hidden)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_hidden)))

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        # if self.mask_flag:
        #     attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, x, mask=None):
        d_k, n_head = self.d_k, self.n_head

        bs, seq_len, _ = x.size()
        # residual = self.res(q)

        q = self.w_qs(x).view(bs, seq_len, n_head, d_k)
        k = self.w_ks(x).view(bs, seq_len, n_head, d_k)
        v = self.w_vs(x).view(bs, seq_len, n_head, d_k)

        # B, L_Q, H, D = q.shape
        # _, L_K, _, _ = k.shape

        q = q.transpose(2,1)
        k = k.transpose(2,1)
        v = v.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(seq_len)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(seq_len)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<seq_len else seq_len
        u = u if u<seq_len else seq_len
        
        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./np.sqrt(d_k)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(v, seq_len)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, v, scores_top, index, seq_len, mask)
        
        return context.reshape(bs, seq_len, -1).contiguous(), attn


class SymMultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_hidden, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_hidden // n_head
        self.w_qs = nn.Linear(d_model, d_hidden)
        self.w_vs = nn.Linear(d_model, d_hidden)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_k, 0.2))
        self.layer_norm = nn.LayerNorm(d_hidden)
        # self.fc = nn.Linear(n_head * d_v, d_k)
        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):

        d_k, n_head = self.d_k, self.n_head

        bs, seq_len, _ = x.size()
        # residual = self.res(q)

        q = self.w_qs(x).view(bs, seq_len, n_head, d_k)
        k = self.w_qs(x).view(bs, seq_len, n_head, d_k)
        v = self.w_vs(x).view(bs, seq_len, n_head, d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k) # (n*b) x lv x dv

        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, bs, seq_len, d_k)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, seq_len, -1) # b x lq x (n*dv)
        output = self.dropout(output)
        output = self.layer_norm(output + x)

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


class PositionwiseDistilling(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 3, padding=2) 
        self.pool = nn.MaxPool1d(stride=2, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(d_in)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x.transpose(1, 2))
        x = self.norm(x)
        x = self.act(x)
        output = self.pool(x)
        output = output.transpose(1, 2)
        return output


class Informer(nn.Module):
    
    def __init__(self, args, featureNum, outputNum):

        super().__init__()
        
        self.seq_len = args.window
        self.dimS = featureNum
        self.hidden = args.d_k
        self.hidR = args.hidR
        self.layerR = args.layerR
        self.outputNum = outputNum
        self.hidden_t = self.seq_len
        self.round = args.round

        self.TA = nn.ModuleList([ProbAttention(args.n_head, self.hidden, self.hidden, dropout=args.dropout) for i in range(self.round)])
        self.ffn = nn.ModuleList([PositionwiseDistilling(self.hidden, self.hidden, dropout=args.dropout) for i in range(self.round)])
        self.mlp1 = nn.Linear(self.dimS, self.hidden)

        self.GRU = nn.GRU(self.hidden, self.hidR, num_layers=args.layerR)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear_out=nn.Linear(self.hidR, outputNum)

    def forward(self, input):

        output = self.mlp1(input)
        graph = []
        for round in range(self.round):
            output, _graph = self.TA[round](output, mask=None)
            graph.append(_graph)
            output = self.ffn[round](output)
            graph.append(_graph)
        
        output = output.transpose(0, 1).contiguous() # seq, batch, n
        _, output = self.GRU(output)
        output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
        out = self.linear_out(output)

        return out, graph[0], graph[1], graph[2], graph[3]
    

class TransFormer(nn.Module):
    def __init__(self, args, featureNum, outputNum):

        super().__init__()
        
        self.dimS = featureNum
        self.decayT = args.decayT
        self.hidden = args.d_k
        self.hidR = args.hidR
        self.layerR = args.layerR
        self.outputNum = outputNum
        self.round = args.round

        self.TA = nn.ModuleList([MultiHeadAttention(args.n_head, self.hidden, self.hidden, dropout=args.dropout) for i in range(self.round)])
        self.ffn = nn.ModuleList([PositionwiseFeedForward(self.hidden, self.hidden, dropout=args.dropout) for i in range(self.round)])
        self.mlp1 = nn.Linear(self.dimS, self.hidden)

        self.GRU = nn.GRU(self.hidden, self.hidR, num_layers=self.layerR)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear_out=nn.Linear(self.hidR, self.outputNum)

    def forward(self, input):

        output = self.mlp1(input)
        graph = []
        for round in range(self.round):
            output, _graph = self.TA[round](output, mask=None)
            graph.append(_graph)
            graph.append(_graph)
            output = self.ffn[round](output)
        
        output = output.transpose(0, 1).contiguous() # seq, batch, n
        _, output = self.GRU(output)
        output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
        out = self.linear_out(output)

        return out, graph[0], graph[1], graph[2], graph[3]
    


def get_frequency_modes(seq_len, modes=4, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index
    

class FourierBlock(nn.Module):

    def __init__(self, n_head, d_model, d_hidden, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_hidden // n_head
        seq_len = 16
        self.w_qs = nn.Linear(d_model, d_hidden)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))

        self.index = get_frequency_modes(seq_len, modes=4, mode_select_method='random')
        self.scale = (1 / (d_model * d_hidden))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.n_head, self.d_k, self.d_k, len(self.index), dtype=torch.cfloat))

        self.layer_norm = nn.LayerNorm(d_hidden)
        self.dropout = nn.Dropout(dropout)

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)
    
    def forward(self, x, mask=None):

        d_k, n_head = self.d_k, self.n_head

        bs, seq_len, _ = x.size()
        # residual = self.res(q)

        q = self.w_qs(x).view(bs, seq_len, n_head, d_k)
        q = q.permute(0, 2, 3, 1) # bs, n, dk, seq
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(q, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(bs, n_head, d_k, seq_len // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        x_if = torch.fft.irfft(out_ft, n=q.shape[-1]) # bs, h, e, s
        output = x_if.permute(0, 3, 2, 1).contiguous().view(bs, seq_len, -1) # b x lq x (n*dv)
        output = self.dropout(output)
        output = self.layer_norm(output + x)

        return output


class FilterLayer(nn.Module):
    """
    Canonical FMLP-Rec
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


class FrequencyMLP(nn.Module):
    """
    Canonical FMLP-Rec
    """
    def __init__(self, d_model, dropout, seq_len=16, init_ratio=1):
        super().__init__()
        self.linear = nn.Linear(seq_len//2+1, seq_len//2+1)
        self.out_dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)


    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        # weight = torch.view_as_complex(self.complex_weight)
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        x = x.transpose(2,1)
        x = self.linear(x)
        x = x.transpose(2,1)
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

        self.SA = nn.ModuleList([FilterLayer(self.hidden, dropout=args.dropout, seq_len=self.seq_len, init_ratio=self.init_ratio) for i in range(self.round)])
        self.ffn = nn.ModuleList([PositionwiseFeedForward(self.hidden, self.hidden, dropout=args.dropout) for i in range(self.round)])
        self.mlp1 = nn.Linear(self.dimS, self.hidden)

        self.GRU = nn.GRU(self.hidden, self.hidR, num_layers=args.layerR)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear_out=nn.Linear(self.hidR, outputNum)

    def forward(self, input_tensor):

        output = self.mlp1(input_tensor)
        for round in range(self.round):
            output = self.SA[round](output)
            output = self.ffn[round](output)
        
        output = output.transpose(0, 1).contiguous() # seq, batch, n
        _, output = self.GRU(output)
        output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
        out = self.linear_out(output)

        return out
    

class FNN(nn.Module):
    
    def __init__(self, args, featureNum, outputNum):

        super().__init__()
        
        self.seq_len = args.window
        self.dimS = featureNum
        self.hidden = args.d_k
        self.hidR = args.hidR
        self.layerR = args.layerR
        self.outputNum = outputNum
        self.round = args.round
        self.init_ratio = args.init_ratio

        self.SA = nn.ModuleList([FrequencyMLP(self.hidden, dropout=args.dropout, seq_len=self.seq_len, init_ratio=self.init_ratio) for i in range(self.round)])
        self.ffn = nn.ModuleList([PositionwiseFeedForward(self.hidden, self.hidden, dropout=args.dropout) for i in range(self.round)])
        self.mlp1 = nn.Linear(self.dimS, self.hidden)

        self.GRU = nn.GRU(self.hidden, self.hidR, num_layers=args.layerR)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear_out=nn.Linear(self.hidR, outputNum)

    def forward(self, input_tensor):

        output = self.mlp1(input_tensor)
        for round in range(self.round):
            output = self.SA[round](output)
            output = self.ffn[round](output)
        
        output = output.transpose(0, 1).contiguous() # seq, batch, n
        _, output = self.GRU(output)
        output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
        out = self.linear_out(output)

        return out

class Fedformer(nn.Module):
    
    def __init__(self, args, featureNum, outputNum):

        super().__init__()
        
        self.dimS = featureNum
        self.decayT = args.decayT
        self.hidden = args.d_k
        self.hidR = args.hidR
        self.layerR = args.layerR
        self.outputNum = outputNum
        self.round = args.round

        self.TA = nn.ModuleList([FourierBlock(args.n_head, self.hidden, self.hidden, dropout=args.dropout) for i in range(self.round)])
        self.ffn = nn.ModuleList([PositionwiseFeedForward(self.hidden, self.hidden, dropout=args.dropout) for i in range(self.round)])
        self.mlp1 = nn.Linear(self.dimS, self.hidden)

        self.GRU = nn.GRU(self.hidden, self.hidR, num_layers=self.layerR)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear_out=nn.Linear(self.hidR, self.outputNum)

    def forward(self, input):

        output = self.mlp1(input)
        for round in range(self.round):
            output = self.TA[round](output, mask=None)
            output = self.ffn[round](output)
        
        output = output.transpose(0, 1).contiguous() # seq, batch, n
        _, output = self.GRU(output)
        output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
        out = self.linear_out(output)

        return out

    


class STANet(nn.Module):
    
    def __init__(self, args, featureNum, outputNum):

        super(STANet, self).__init__()
        
        self.seq_len = args.window
        self.dimS = featureNum
        self.decayT = args.decayT
        self.hidden = args.d_k
        self.decayS = args.decayS
        self.hidR = args.hidR
        self.layerR = args.layerR
        self.outputNum = outputNum
        self.hidden_t = self.seq_len
        self.round = args.round

        self.SA = nn.ModuleList([MultiHeadAttention(args.n_head, self.hidden_t, self.hidden_t, dropout=args.dropout) for i in range(self.round)])
        self.TA = nn.ModuleList([MultiHeadAttention(args.n_head, self.hidden, self.hidden, dropout=args.dropout) for i in range(self.round)])
        self.mlp1 = nn.Linear(self.dimS, self.hidden)

        self.GRU = nn.GRU(self.hidden, self.hidR, num_layers=args.layerR)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear_out=nn.Linear(self.hidR, outputNum)

    def forward(self, input):

        output = input.transpose(2, 1)
        output, graphS1 = self.SA[0](output, mask=None)
        output = output.transpose(2, 1)
        output = self.mlp1(output)
        output, graphT1 = self.TA[0](output, mask=None)
        if self.round == 1:
            output = output.transpose(0, 1).contiguous() # seq, batch, n
            _, output = self.GRU(output)
            output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
            out = self.linear_out(output)
            return out, graphS1, graphT1, graphS1, graphT1

        output = output.transpose(2, 1)
        output, graphS2 = self.SA[1](output, mask=None)
        output = output.transpose(2, 1)
        output, graphT2 = self.TA[1](output, mask=None)
        if self.round == 2:
            output = output.transpose(0, 1).contiguous() # seq, batch, n
            _, output = self.GRU(output)
            output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
            out = self.linear_out(output)
            return out, graphS1, graphT1, graphS2, graphT2
        
        for round in range(2, self.round):
            output = output.transpose(2, 1)
            output, graph = self.SA[round](output, mask=None)
            output = output.transpose(2, 1)
            output, graph = self.TA[round](output, mask=None)
        
        output = output.transpose(0, 1).contiguous() # seq, batch, n
        _, output = self.GRU(output)
        output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
        out = self.linear_out(output)

        return out, graphS1, graphT1, graphS2, graphT2
    
class STANetSym(STANet):
    
    def __init__(self, args, featureNum, outputNum):
        super().__init__(args, featureNum, outputNum) 
        self.SA = nn.ModuleList([SymMultiHeadAttention(args.n_head, self.hidden_t, self.hidden_t, dropout=args.dropout) for i in range(self.round)])
        self.TA = nn.ModuleList([SymMultiHeadAttention(args.n_head, self.hidden, self.hidden, dropout=args.dropout) for i in range(self.round)])


class SANet(STANet):
    
    def __init__(self, args, featureNum, outputNum):
        super().__init__(args, featureNum, outputNum)     
        del self.TA

    def forward(self, input):

        output = self.mlp1(input)
        output = output.transpose(2, 1)
        output, graphS1 = self.SA[0](output, mask=None)
        output, graphS2 = self.SA[1](output, mask=None)
        output = output.transpose(2, 1)
        output = output.transpose(0, 1).contiguous() # seq, batch, n
        _, output = self.GRU(output)
        output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
        out = self.linear_out(output)

        return out, graphS1, graphS1, graphS2, graphS2
    

class TANet(STANet):
    
    def __init__(self, args, featureNum, outputNum):
        super().__init__(args, featureNum, outputNum)     
        del self.SA
        del self.mlp1
        # self.TA = nn.ModuleList([MultiHeadAttention(args.n_head, self.hidden, self.hidden, dropout=args.dropout) for i in range(self.round)])
        # self.TA[0] = MultiHeadAttention(args.n_head, featureNum, featureNum, dropout=args.dropout)
        self.TA = nn.ModuleList([MultiHeadAttention(args.n_head, featureNum, featureNum, dropout=args.dropout) for i in range(self.round)])
        self.GRU = nn.GRU(featureNum, self.hidR, num_layers=args.layerR)

    def forward(self, input):

        output, graphT1 = self.TA[0](input, mask=None)
        output, graphT2 = self.TA[1](output, mask=None)
        output = output.transpose(0, 1).contiguous() # seq, batch, n
        _, output = self.GRU(output)
        output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
        out = self.linear_out(output)

        return out, graphT1, graphT1, graphT2, graphT2
    


class iTransformer(nn.Module):
    
    def __init__(self, args, featureNum, outputNum):

        super().__init__()
        
        self.seq_len = args.window
        self.dimS = featureNum
        self.hidden = args.d_k
        self.decayS = args.decayS
        self.hidR = args.hidR
        self.layerR = args.layerR
        self.outputNum = outputNum
        self.hidden_t = self.hidden
        self.round = args.round

        self.SA = nn.ModuleList([MultiHeadAttention(args.n_head, self.seq_len, self.seq_len, dropout=args.dropout) for i in range(self.round)])
        self.TA = nn.ModuleList([PositionwiseFeedForward(self.seq_len, self.seq_len, dropout=args.dropout) for i in range(self.round)])
        self.mlp1 = nn.Linear(self.dimS, self.hidden)

        self.GRU = nn.GRU(self.hidden, self.hidR, num_layers=args.layerR, batch_first=True)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear_out=nn.Linear(self.hidR, outputNum)

    def forward(self, input):
        output = self.mlp1(input)
        output = output.transpose(2, 1)
        
        output, _ = self.SA[0](output, mask=None)
        # output = output.transpose(2, 1)
        output = self.TA[0](output)
        if self.round == 1:
            output = output.transpose(2, 1).contiguous() # seq, batch, n
            _, output = self.GRU(output)
            output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
            out = self.linear_out(output)
            return out

        # output = output.transpose(2, 1)
        # output = self.SA[1](output, mask=None)
        # output = output.transpose(2, 1)
        # output = self.TA[1](output, mask=None)
        # if self.round == 2:
        #     output = output.transpose(2, 1).transpose(0, 1).contiguous() # seq, batch, n
        #     _, output = self.GRU(output)
        #     output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
        #     out = self.linear_out(output)
        #     return out
        
        for round in range(1, self.round):
            # output = output.transpose(2, 1)
            output, _ = self.SA[round](output, mask=None)
            # output = output.transpose(2, 1)
            output = self.TA[round](output)
        
        output = output.transpose(2, 1).contiguous() # seq, batch, n
        _, output = self.GRU(output)
        output = self.dropout(torch.squeeze(output[-1:, :, :], 0))
        out = self.linear_out(output)

        return out