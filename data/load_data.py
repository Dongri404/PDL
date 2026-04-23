import copy
import math
import os
import logging
import pdb
import pickle
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
        }
        DATA_MAP[args.datasetName]()
        self.sim_matrix = None
        if args.use_simi_matrix:
            self.scaled_fusion = None

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)
        # if self.args.datasetName == 'mosi' or self.args.datasetName == 'mosei':
        #     with open(f"../Self-MM-Origin/{self.args.datasetName}-labels.pkl", 'rb') as f:
        #         uni_labels = pickle.load(f)
        if self.args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        # proj = {
        #     'fusion':'M',
        #     'text':'T',
        #     'audio':'A',
        #     'vision':'V',
        #     'mosi':17,
        #     'mosei':5
        # }
        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }
        if self.args.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]
        # if self.args.datasetName == 'mosi' or self.args.datasetName == 'mosei':
        #     for m in ["text","audio","vision"]:
        #         pdb.set_trace()
        #         self.labels[proj[m]] = uni_labels[proj[self.args.datasetName]][self.mode][self.args.train_mode+'_labels_'+m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if  self.args.need_normalized:
            self.__normalize()

        # scale
        self.__scale()

    def __scale(self):
        # scale[-1,1]
        vision = self.vision.copy()
        audio = self.audio.copy()
        for i in range(self.vision.shape[-1]):
            max_num = np.max(vision[:,:,i])
            min_num = np.min(vision[:,:,i])
            self.vision[:,:,i]=(vision[:,:,i] - min_num)/((max_num - min_num)*2 + 0.0001) - 1
        for i in range(self.audio.shape[-1]):
            max_num = np.max(audio[:,:,i])
            min_num = np.min(audio[:,:,i])
            self.audio[:,:,i] = (audio[:,:,i]-min_num)/((max_num-min_num)*2 + 0.0001) -1
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __truncated(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
                       
        text_length, audio_length, video_length = self.args.seq_lens
        self.vision = Truncated(self.vision, video_length)
        self.text = Truncated(self.text, text_length)
        self.audio = Truncated(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    
    def normal_sampling(self, mean, label_k, std=3):
        return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

    def __getitem__(self, index):
        sample = {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        if not self.args.need_data_aligned:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]

        # gen_label_distribution
        # if True:
        #     label_M = sample['labels']['M']
        #     label_d = [self.normal_sampling(mean=label_M, label_k=i) for i in self.args.labels]
        #     label_d = [i if i>1e-15 else 1e-15 for i in label_d]
        #     label_d = torch.tensor(label_d)
        #     sample['labels_d_M'] = label_d
        return sample
    
    def update_features(self, f_fusion):
        self.scaled_fusion = f_fusion
        self.__gen_cos_matrix()
    
    def __gen_cos_matrix(self, model=None):
        feature = self.scaled_fusion # [1284, dim=256]
        n, _ = feature.shape

        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        batch_size = 1284
        print(f"Calculating Sim Matrix in Batch {batch_size}")
        cut = math.floor(n / batch_size)
        A = feature.unsqueeze(1)
        B = feature.unsqueeze(0)
        self.cos_matrix_M = torch.zeros((n, n))
        star_row = None
        end_row = None
        star_column = None
        end_column = None
        for i in range(cut+1):
            star_row = i*batch_size
            end_row = star_row + batch_size
            if end_row>n:
                end_row=n
            for j in range(cut+1):
                star_column = j*batch_size
                end_column = star_column+batch_size
                if end_column>n:
                    end_column = n
                self.cos_matrix_M[star_row:end_row,star_column:end_column] = cos(
                    A[star_row:end_row,:,:].to(torch.device(self.args.device)),
                    B[:,star_column:end_column,:].to(torch.device(self.args.device))).to(torch.device('cpu'))

        # # verify the sim_matrix
        # epsilon = 0.0001
        # for i in range(n):
        #     for j in range(n):
        #         value = cos(A[i,:,:],B[:,j,:])[0]
        #         if abs(value-self.cos_matrix_M[i,j])>epsilon:
        #             print(f"不相等！！！！{i},{j},value={value}") 
        del A,B
        
        print("\t----------Finished")
        
        self.rank_M = torch.zeros(self.cos_matrix_M.shape)
        
        for i in range(len(self.cos_matrix_M)):
            _, self.rank_M[i, :] = torch.sort(self.cos_matrix_M[i, :], descending=True) # 降序
        self.cos_matrix_M = None
        self.M_retrieve = self.__pre_sample(self.rank_M, np.round(self.labels['M']))
    
    def __pre_sample(self, _rank, _label,  matrix_batch_size=None,index1=None, index2=None):
        n, _ = _rank.shape
        retrieve = {'ss': [], 
                    'sd': [],
                    'ds': [],
                    'dd': [],
                    }
        for i in range(n):
            _ss = []
            _sd = []
            _ds = []
            _dd = []
            for j in range(int(400)):
                if i == j: continue
                if _label[i] == _label[int(_rank[i][j])]:
                    _ss.append(j) # 相似度高，标签相同
                else:
                    _sd.append(j) # 相似度高，标签不同
            for j in range(-1, -int(400), -1):
                if i == j: continue
                if _label[i] == _label[int(_rank[i][j])]:
                    _ds.append(j) # 相似度低，标签相同
                else:
                    _dd.append(j) # 相似度低，标签不同

            if len(_ss) < 2 or len(_sd) < 2 or len(_ds) < 2 or len(_dd) < 2:
                print('Unique sample detected, may cause error!')

            retrieve['ss'].append(_ss[:10]) # append相似度高的前10个
            retrieve['sd'].append(_sd[:10])
            retrieve['ds'].append(_ds[:10]) # append相似度低的前10个
            retrieve['dd'].append(_dd[:10])
        return retrieve
    
    def sample(self, _sample_idx):
        samples2 = {}
        idx2 = []
        for i in _sample_idx:   
            idx2 += random.sample(self.M_retrieve['ss'][i], 1)
            idx2 += random.sample(self.M_retrieve['dd'][i], 1)
            idx2 += random.sample(self.M_retrieve['sd'][i], 1)
        # text
        samples2['text'] = torch.Tensor(self.text[idx2])
        # audio
        samples2['audio'] = torch.Tensor(self.audio[idx2])
        # vision
        samples2['vision'] = torch.Tensor(self.vision[idx2])
        # index
        samples2['index'] = torch.Tensor(idx2)
        # id
        samples2['id'] = np.array(self.ids)[idx2]
        # labels
        samples2['labels'] = {k: torch.Tensor(v[idx2].reshape(-1)) for k, v in self.labels.items()}
        # raw_text
        samples2['raw_text'] = self.rawText[idx2]
        # length
        samples2['audio_lengths'] = np.array(self.audio_lengths)[idx2]
        samples2['vision_length'] = np.array(self.vision_lengths)[idx2]
        return samples2

def MMDataLoader(args):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader