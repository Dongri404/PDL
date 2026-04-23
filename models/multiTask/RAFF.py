import pdb

import torch.nn as nn
import torch
import torch.nn.functional as F

class RAFF(nn.Module):
    def __init__(self, dim, fusion_dim, n_prompt):
        super(RAFF, self).__init__()
        self.dim = dim
        self.fusion_dim = fusion_dim

        self.squence_length = n_prompt

        self.self_att = nn.MultiheadAttention(self.dim, num_heads=1, batch_first=True)
        self.classifiers_list = nn.ModuleList([])
        for i in range(self.squence_length*2):
            self.classifiers_list.append(
                nn.Sequential(
                    nn.Linear(self.dim, int(self.dim/2)),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(int(self.dim/2), 1),
                    nn.ReLU()
                )
            )
        self.fusion_proj = nn.Sequential(
        nn.Linear(self.dim, self.fusion_dim),
        nn.LayerNorm(self.fusion_dim),
        nn.ReLU(),
        nn.Dropout(0.1)
        )
        self.projector_a = nn.Sequential(
            nn.Linear(self.dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.projector_v = nn.Sequential(
            nn.Linear(self.dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, q):
        '''
        q: [b, n_prompt*2, d]
        '''
        
        q_att, _ = self.self_att(q, q, q) #  q_att [b, n_prompt*2, d]

        d = []
        for i in range(q.shape[1]):
            d_i = self.classifiers_list[i](q[:,i]) # [b, 1]
            d.append(d_i)
        d = torch.cat(d, dim=1) # [b, n_prompt*2]
        q_att_a = q_att[:,:self.squence_length,:]
        q_att_v = q_att[:,self.squence_length:,:]
        d_a = F.softmax(d[:,:self.squence_length], dim=-1)
        d_v = F.softmax(d[:,self.squence_length:], dim=-1)
        
        audio = (q_att_a*d_a.unsqueeze(2)).sum(1)
        video = (q_att_v*d_v.unsqueeze(2)).sum(1)
        fusion = (audio+video)/2
        
        fusion_h = self.fusion_proj(fusion) # [b, dim]
        fusion_a = self.projector_a(audio)
        fusion_v = self.projector_v(video)
        return fusion_h, fusion_a, fusion_v, d_a, d_v