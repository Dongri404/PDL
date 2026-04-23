import json
import os
import pdb
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from trains.multiTask.emd import SinkhornDistance

logger = logging.getLogger('MSA')

class PDL():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "M" 
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
        }

        self.center_map = {
            'fusion': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            },
            'text': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            },
            'audio': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            },
            'vision': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            }
        }


        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

    def do_train(self, model, dataloader):
        # load pretrain model
        if self.args.load_pretrain_model:
            logger.info('-Loading pretrained model...')
            pretrain_dict = {}
            for modality in ['audio', 'vision']:
                uni_model_save_path = os.path.join(self.args.model_save_dir,'pretrain-model', f'{self.args.datasetName}-{modality}-{self.args.pretrain_set}-encoder.pth')
                uni_dict = torch.load(uni_model_save_path, map_location=self.args.device)['model_state_dict']
                pretrain_dict.update(uni_dict)
                logger.info(f'-Loading {modality} pretrained model from {uni_model_save_path}...')
            model_dict = model.Model.state_dict()
            same_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
            model_dict.update(same_dict)
            model.Model.load_state_dict(model_dict)
            logger.info('-Pretrained model loaded')
        
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                                'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
        
        # warm-up+Cosine Anneal
        warm_up_iter = self.args.warmup_epoch*len(dataloader['train'])//4
        T_max = 100*len(dataloader['train'])//4
        lr_max = 1*self.args.max_lr_times
        lr_min = 1e-6

        # Warm up + Cosine Anneal
        lambda0 = lambda cur_iter: (cur_iter / warm_up_iter)*lr_max if  cur_iter < warm_up_iter else \
                (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))
        # lambda0 = lambda cur_iter: (cur_iter / warm_up_iter)*lr_max if  cur_iter < warm_up_iter else 1.*lr_max

        lambda1 = lambda cur_iter: 1

        # LambdaLR
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1,lambda0,lambda0,lambda0,lambda0])
        lr = []

        saved_labels = {}
        
        # init labels
        logger.info("Init labels...")
        with torch.no_grad():
            model.eval()
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    raw_text = batch_data['raw_text']
                    labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                    if self.args.datasetName == 'sims':
                        labels_a = batch_data['labels']['A'].view(-1).to(self.args.device)
                        labels_v = batch_data['labels']['V'].view(-1).to(self.args.device)
                    else:
                        labels_a, labels_v = None, None
                    indexes = batch_data['index'].view(-1)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0
                    
                    outputs = model(raw_text, (audio,  audio_lengths), (vision, vision_lengths))

                    self.init_labels(indexes, labels_m, a_labels=labels_a, v_labels=labels_v)
                    # self.update_features(f_fusion, indexes)
            if self.args.use_simi_matrix:
                dataloader['train'].dataset.update_features(self.feature_map['fusion'])
            model.train()

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss', 'MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        bestResult = self.args.metrics
        
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
                
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    raw_text = batch_data['raw_text']
                    labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward
                    outputs = model(raw_text, (audio, audio_lengths), (vision, vision_lengths),)
                      
                    # store results
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(self.label_map[self.name_map[m]][indexes].cpu())
                    # compute loss
                    loss = self.get_loss(outputs, self.label_map, batch_data, mode='TRAIN')
                    # backward
                    loss.backward()
                    train_loss += loss.item()
                    # update features
                    f_fusion = outputs['Feature_f'].detach()
                    # f_text = outputs['Feature_t'].detach()
                    f_audio = outputs['Feature_a'].detach()
                    f_vision = outputs['Feature_v'].detach()
                    if epochs > 1 and self.args.datasetName != 'sims': # update the pseudo label
                        self.update_labels(f_fusion, f_audio, f_vision, epochs, indexes, outputs)

                        self.update_features(f_fusion, f_audio, f_vision, indexes)
                        self.update_centers()
                    
                    # update parameters
                    if not left_epochs:
                        # update
                        lr.append(optimizer.param_groups[-1]['lr'] )
                        optimizer.step()
                        scheduler.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    lr.append(optimizer.param_groups[-1]['lr'] )
                    optimizer.step()
                    scheduler.step()
            train_loss = train_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss))
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(train_results))

            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]

            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                bestResult.update(val_results)
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # save labels
            if self.args.save_labels:
                tmp_save = {k: v.cpu().numpy() for k, v in self.label_map.items()}
                tmp_save['ids'] = ids
                saved_labels[epochs] = tmp_save
            # early stop
            if (epochs - best_epoch >= self.args.early_stop and epochs>self.args.train_epoch and self.args.tune_normals == "normals") or \
                (epochs - best_epoch >= self.args.early_stop and epochs>self.args.tune_epoch and self.args.tune_normals == "tunes"):
                if self.args.save_labels:
                    with open(os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                        plk.dump(saved_labels, df, protocol=4)
                if not os.path.exists(os.path.dirname(self.args.save_path)):
                    os.makedirs(os.path.dirname(self.args.save_path))
                logger.info(bestResult)
                return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
      
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    raw_text = batch_data['raw_text']
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels_m = batch_data['labels']['M'].to(self.args.device).view(-1)
                    
                    outputs = model(raw_text, (audio, audio_lengths), (vision, vision_lengths))
                    
                    loss = self.get_loss(outputs, labels_m, batch_data, mode=mode)
                    eval_loss += loss.item()
                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels_m.cpu())
                    
        eval_loss = eval_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        eval_results = self.metrics(pred, true)
        logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = eval_loss
        
        return eval_results
    
    def MAELoss(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        # MAE
        loss = torch.abs(y_pred - y_true)
        loss = torch.mean(loss)
                
        return loss
    
    def CDLLoss(self, distr, targets, preds, w_l, w_h, weights=1, e=1.):
        preds = preds.view(-1, 1)
        targets = targets.view(-1, 1)

        l_k = targets.flatten()[None, :] # [1,32]
        l_q = targets # [32, 1]

        p_k = preds.flatten()[None, :] # [1,32]
        p_q = preds # [32,1]

        l_dist = torch.abs(l_q - l_k) # [32, 32]
        p_dist = torch.abs(p_q - p_k) #[32, 32]

        pos_i = l_dist.le(w_l) # [32, 32]
        
        neg_i = (~(l_dist.le(w_l)) * (p_dist.le(w_l)))

        # mask the diagonal elements.
        for i in range(pos_i.shape[0]):
            pos_i[i][i] = 0
            
        # prod = torch.einsum("nc,kc->nk", [q, k]) / t # prod = q@k.T/t
        emd_dis = SinkhornDistance(eps = 0.1, max_iter=3)
        prod = -emd_dis(distr, distr)[-1]
        pos = prod * pos_i
        neg = prod * neg_i

        pushing_w = weights * torch.exp(l_dist * e) # !!!
        neg_exp_dot = (pushing_w * (torch.exp(neg)) * neg_i).sum(1)

        # For each query sample, if there is no negative pair, zero-out the loss.
        no_neg_flag = (neg_i).sum(1).bool()

        # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
        denom = pos_i.sum(1) + 1e-8

        loss = ((-torch.log(
            torch.div(torch.exp(pos), (torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1))) * (
                    pos_i)).sum(1) / denom)

        loss = (weights * (loss * no_neg_flag).unsqueeze(-1)).mean() # weights!!!
       
        return loss
    
    
    def update_features(self, f_fusion,f_audio, f_vision, indexes):
        self.feature_map['fusion'][indexes] = f_fusion
        self.feature_map['audio'][indexes] = f_audio
        self.feature_map['vision'][indexes] = f_vision

    def update_centers(self):
        def update_single_center(mode):
            neg_indexes = self.label_map[mode] < 0
            if self.args.excludeZero:
                pos_indexes = self.label_map[mode] > 0
            else:
                pos_indexes = self.label_map[mode] >= 0
            self.center_map[mode]['pos'] = torch.mean(self.feature_map[mode][pos_indexes], dim=0)
            self.center_map[mode]['neg'] = torch.mean(self.feature_map[mode][neg_indexes], dim=0)

        update_single_center(mode='fusion')
        # update_single_center(mode='text')
        update_single_center(mode='audio')
        update_single_center(mode='vision')
    
    def init_labels(self, indexes, m_labels, t_labels=None, a_labels=None, v_labels=None):
        self.label_map['fusion'][indexes] = m_labels
        if a_labels!=None and v_labels!=None:
            # self.label_map['text'][indexes] = t_labels
            self.label_map['audio'][indexes] = a_labels
            self.label_map['vision'][indexes] = v_labels
        else:
            # self.label_map['text'][indexes] = m_labels
            self.label_map['audio'][indexes] = m_labels
            self.label_map['vision'][indexes] = m_labels
    
    def update_labels(self, f_fusion, f_audio, f_vision, cur_epoches, indexes, outputs):
        MIN = 1e-8
        def update_single_label(f_single, mode):
            d_sp = torch.norm(f_single - self.center_map[mode]['pos'], dim=-1) 
            d_sn = torch.norm(f_single - self.center_map[mode]['neg'], dim=-1) 
            delta_s = (d_sn - d_sp) / (d_sp + MIN)
            # d_s_pn = torch.norm(self.center_map[mode]['pos'] - self.center_map[mode]['neg'], dim=-1)
            # delta_s = (d_sn - d_sp) / (d_s_pn + MIN)
            alpha = delta_s / (delta_f + MIN)

            new_labels = 0.5 * alpha * self.label_map['fusion'][indexes] + \
                        0.5 * (self.label_map['fusion'][indexes] + delta_s - delta_f)
            new_labels = torch.clamp(new_labels, min=-self.args.H, max=self.args.H)
            # new_labels = torch.tanh(new_labels) * self.args.H

            n = cur_epoches
            self.label_map[mode][indexes] = (n - 1) / (n + 1) * self.label_map[mode][indexes] + 2 / (n + 1) * new_labels

        d_fp = torch.norm(f_fusion - self.center_map['fusion']['pos'], dim=-1) 
        d_fn = torch.norm(f_fusion - self.center_map['fusion']['neg'], dim=-1) 
        # d_f_pn = torch.norm(self.center_map['fusion']['pos'] - self.center_map['fusion']['neg'], dim=-1)
        # delta_f = (d_fn - d_fp) / (d_f_pn + MIN)
        delta_f = (d_fn - d_fp) / (d_fp + MIN)
        
        # update_single_label(f_text, mode='text')
        update_single_label(f_audio, mode='audio')
        update_single_label(f_vision, mode='vision')

    def get_loss(self, outputs, label_truth, batch_data, mode="TRAIN"):
        indexes = batch_data['index'].view(-1)
        loss = 0.0

        if mode == 'TRAIN':
            labels_m = label_truth['fusion'][indexes]
            labels_a = label_truth['audio'][indexes]
            labels_v = label_truth['vision'][indexes]
        else:
            labels_m = label_truth
            labels_a = label_truth
            labels_v = label_truth
        
        mae_loss_m= self.MAELoss(outputs['M'], labels_m)
        mae_loss_v = self.MAELoss(outputs['V'], labels_v)
        mae_loss_a = self.MAELoss(outputs['A'], labels_a)
        conr_d = self.args.lossLambda*self.CDLLoss(distr=torch.cat((outputs['distribution_a'], outputs['distribution_v']), dim=0),\
             targets=torch.cat((labels_a, labels_v), dim=0), preds=torch.cat((outputs['A'],outputs['V'])).detach(),\
                 w_l=self.args.conr_w, w_h=self.args.H-self.args.conr_w)
        
        loss = mae_loss_m + mae_loss_v + mae_loss_a + conr_d
        
        return loss