'''
    TCFD: Task-Conditioned Feature Decomposition
'''

from collections import OrderedDict
import pdb
import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F

class TCFDLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, ffn_dim=128, dropout=0., ffn_token=None, side_models=None):
        super(TCFDLayer, self).__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.self_attention = torch.nn.MultiheadAttention(embed_dim = dim, num_heads = heads, dropout = dropout, batch_first=True)

        self.resampler_list = side_models

        self.FFN = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            # nn.BatchNorm1d(ffn_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            # nn.BatchNorm1d(ffn_token),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, prompts, n_side, n_prompt, anchor_tokens, side_tokens):
        # self attention
        x = torch.cat((prompts, anchor_tokens), dim=1)
        x_q = self.to_q(x)
        x_k = self.to_k(x)
        x_v = self.to_v(x) # [ , , embed_dim]

        x_att, _ = self.self_attention(x_q, x_k, x_v) 
        q_int = x_att[:, :prompts.shape[1]]
        x_l = x_att[:, prompts.shape[1]:] # [ , , embed_dim]

        # Resampler
        q_int_ = []
        
        for i, model_i in enumerate(self.resampler_list):
            q_int_i, _ = model_i(q_int[:,i*n_prompt:n_prompt+i*n_prompt,:], side_tokens[i])
            q_int_.append(q_int_i)
        if n_prompt+i*n_prompt<q_int.shape[1]:
            q_int_.append(q_int[:,n_prompt+i*n_prompt:,:])
        q_int_ = torch.cat(q_int_, dim=1) # [ , , embed_dim]

        # FFN & BN
        x_ao = x + torch.cat((q_int_, x_l), dim=1)
        x_ = self.norm(x_ao + self.FFN(x_ao))
        q_ = x_[:, :prompts.shape[1]]
        x_l_ = x_[:, prompts.shape[1]:]

        return q_, x_l_ # [ , , dim]
    

class TCFD(nn.Module):
    def __init__(self, dim, depth, att_heads, n_side_models, n_prompt, total_prompt=None, ffn_dim=128, dropout = 0.):
        '''
        parameters:self_att_
            n_side_models: how many modality expect anchor modality
            n_prompt: how many prompt per modality
        note:
            embed_dim = att_heads*att_dim_head
        '''
        super(TCFD, self).__init__()
        self.n_side = n_side_models
        self.n_prompt = n_prompt
        assert dim%att_heads==0, '(qap embed_dim) could not  devided by (attention heads)'
        att_dim_head = dim//att_heads

        # project_in = not (embed_dim == dim)
        # self.to_in_prompts = nn.Sequential(
        #     nn.Linear(dim, embed_dim),
        #     nn.Dropout(dropout)
        # ) if project_in else nn.Identity()
        # self.to_in_anchor_tokens = nn.Sequential(
        #     nn.Linear(dim, embed_dim),
        #     nn.Dropout(dropout)
        # ) if project_in else nn.Identity()

        self.layers = nn.ModuleList([])
        for i in range(depth):
            side_models = nn.ModuleList([])
            for _ in range(n_side_models):
                side_models.append(
                    UnimodalDecomposition(dim=dim)
                )
            self.layers.append(
                TCFDLayer(dim, heads=att_heads, ffn_dim=ffn_dim, ffn_token=total_prompt, dim_head=att_dim_head, dropout=dropout, side_models=side_models)
            )

        # self.to_out_prompts = nn.Sequential(
        #     nn.Linear(embed_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_in else nn.Identity()
        # self.to_out_anchor_tokens = nn.Sequential(
        #     nn.Linear(embed_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_in else nn.Identity()

    def forward(self, prompts, anchor_tokens, side_tokens):
        '''
        parameters:
            promts:(batch_size, modalitity_num_of_prompt*token_num_per_modality_of_promt, dim)
            anchor_tokens: (batch_size, token_num, dim) anchor input
            side_tokens:tuple(( batch_size, token_num_per_modality_of_feature, dim), ,)
        '''
        # q, x = self.to_in_prompts(prompts), self.to_in_anchor_tokens(anchor_tokens)
        q, x = prompts, anchor_tokens
        for i, qapLayer in enumerate(self.layers):
            q, x= qapLayer(q, self.n_side, self.n_prompt, x, side_tokens)

        # q, x = self.to_out_prompts(q), self.to_out_anchor_tokens(x)
        return q, x


class UnimodalDecomposition(nn.Module):
    def __init__(self, dim):
        super(UnimodalDecomposition, self).__init__()
        self.linear_kv = nn.Linear(dim, dim)
        self.linear_q = nn.Linear(dim, dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=1, dropout=0., batch_first=True)
        # self.attention = Attention(dim=dim, heads=1, dim_head=dim, dropout=0.)
        # self.gate = nn.Parameter(torch.ones(1, dim, dim)*resampler_gate)
        self.linear_add = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            )
        
    def forward(self, prompt, x):
        '''
        parameters:
            prompt:(batch_size, token_num_per_modality_of_promt, dim)
            x:(batch_size, token_num_per_modality_of_feature, dim)
        return:
            r_p:(batch_size, 1, dim)
        '''

        f = self.linear_kv(x)
        f_att, _ = self.attention(prompt, f, f)
        # gate = repeat(self.gate, '1 n d -> b n d', b = x.shape[0])
        # r_p = prompt + f_att@gate
        # r_p = prompt + self.linear_add(f_att)
        r_p = self.linear_add(prompt + f_att)

        return r_p, _