# self supervised multimodal multi-task learning network
import os
import pdb
import sys
import collections

from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import RobertaTextEncoder
from models.multiTask.TCFD import TCFD
from models.multiTask.RAFF import RAFF

__all__ = ['PDL']

class PDL(nn.Module):
    def __init__(self, args):
        super(PDL, self).__init__()
        self.args = args
        
        self.text_model = RobertaTextEncoder(language=args.language, raw_text=True, use_finetune=args.use_finetune)
        self.audio_model = AuViSubNet(args, modal='audio')
        self.video_model = AuViSubNet(args, modal='vision')

        self.proj_t0 = nn.Sequential(
            nn.Linear(args.text_out, args.qap_embed_dim)
        )
        self.proj_a0 = nn.Sequential(
            nn.Linear(args.audio_out, args.qap_embed_dim)
        )
        self.proj_v0 = nn.Sequential(
            nn.Linear(args.video_out, args.qap_embed_dim)
        )
        self.use_anchor_prompt = args.use_t_prompt
        n_prompt, n_side_models = args.n_prompt, 2
        n_use_p_modality = n_side_models
        if self.use_anchor_prompt == True:
            n_use_p_modality = n_use_p_modality + 1
        # self.promt_t = nn.Parameter(torch.ones(n_prompt, args.post_fusion_dim))
        self.promt_a = nn.Parameter(torch.ones(n_prompt, args.qap_embed_dim))
        self.promt_v = nn.Parameter(torch.ones(n_prompt, args.qap_embed_dim))
        
        total_prompt = n_prompt*n_use_p_modality
        anchor_token_num = args.seq_lens[0] if args.as_anchor=='t' else (args.seq_lens[1] if args.as_anchor=='a' else args.seq_lens[2] )
        self.decomposition = TCFD(dim=args.qap_embed_dim, depth=args.qap_depth, att_heads=8, ffn_dim=args.ffn_dim, total_prompt=total_prompt+anchor_token_num, n_side_models=n_side_models, n_prompt=n_prompt, dropout=args.qap_dropout).to(args.device)

        self.fusion = RAFF(dim=args.qap_embed_dim, fusion_dim=args.post_fusion_dim, n_prompt=n_prompt).to(args.device)
        
        self.regression = nn.Sequential(
            nn.Linear(args.post_fusion_dim, 1)
        )
        self.regression_a = nn.Sequential(
            nn.Linear(args.post_fusion_dim, 1)
        )
        self.regression_v = nn.Sequential(
            nn.Linear(args.post_fusion_dim, 1)
        )

    def forward(self, text0, audio0, video0):
        audio0, audio_lengths = audio0
        video0, video_lengths = video0

        x_text = self.text_model(text0)

        x_audio = self.audio_model(audio0)
        x_video = self.video_model(video0)

        h_t = self.proj_t0(x_text)
        h_a = self.proj_a0(x_audio)
        h_v = self.proj_v0(x_video)

        if self.use_anchor_prompt == True:
            promts_i = torch.cat(( self.promt_a.unsqueeze(0), self.promt_v.unsqueeze(0), self.promt_t.unsqueeze(0)), dim=0) # [3, 1, dim]
        else:
            promts_i = torch.cat((self.promt_a.unsqueeze(0), self.promt_v.unsqueeze(0)), dim=0) # [2, 1, dim]
        n_modal_p, n_prompt, dim = promts_i.shape
        promts = repeat(promts_i.view(1, n_modal_p*n_prompt, dim), '1 n d -> b n d', b = h_t.shape[0]) # [b, n_side*n_prompts, dim]

        if self.args.as_anchor=='t':
            side_tokens = (h_a, h_v)
            anchor_tokens = h_t
        elif self.args.as_anchor=='a':
            side_tokens = (h_a, h_v)
            anchor_tokens = h_a
        elif self.args.as_anchor == 'v':
            side_tokens = (h_a, h_v)
            anchor_tokens = h_v

        q, _ = self.decomposition(promts, anchor_tokens, side_tokens)
        
        fusion_h, fusion_a, fusion_v, d_a, d_v = self.fusion(q)
        
        y = self.regression(fusion_h)
        output_audio = self.regression_a(fusion_a)
        output_video = self.regression_v(fusion_v)
        
        res = {
            'M': y,
            'A': output_audio,
            'V': output_video,
            'Feature_a': fusion_a,
            'Feature_v': fusion_v,
            'Feature_f':fusion_h,
            'distribution_a': d_a,
            'distribution_v': d_v

        }
        return res

class AuViSubNet(nn.Module):
    def __init__(self, args, modal):
        super(AuViSubNet, self).__init__()
        self.encoder_mode = args.encoder_mode
        if modal == 'audio':
            in_features = args.feature_dims[1]
            out_features = args.audio_out
        elif modal =='vision':
            in_features = args.feature_dims[2]
            out_features = args.video_out

        self.linear_in = nn.Linear(in_features,512)
        encoder_layers = nn.TransformerEncoderLayer(512, 8, batch_first= True)
        self.tf = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=args.tf_layer_num)
        self.linear_out = nn.Linear(512, out_features)

        
    def forward(self, x):
        '''
         x: (batch_size, sequence_len, in_size)
        '''
        out_linear_in = self.linear_in(x)
        out_rnn = self.tf(out_linear_in)
        out = self.linear_out(out_rnn)
        return out