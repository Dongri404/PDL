import os
import random
import argparse

from utils.functions import Storage

class ConfigTune():
    def __init__(self, args):
        # global parameters for running
        self.globalArgs = args
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'pdl': self.__PDL
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()
        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['debugParas'],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = '/mnt/disk1/zwj/Multimodality/data'
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (1024, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (1024, 5, 20),
                    'text_out': 1024,
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE',
                    'H': 3.0 ,
                    'label_arange': (-3.0, 3.0),
                    'bin_width':0.5,
                    'label_markers':[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                    # warmup
                    'metrics':{
                        "Has0_acc_2":0.,
                        "Has0_F1_score": 0.,
                        "Non0_acc_2":  0.,
                        "Non0_F1_score": 0.,
                        "Mult_acc_5": 0.,
                        "Mult_acc_7": 0.,
                        "MAE": 1e8,
                        "Corr": -100,
                        'Loss': -1
                    },
                    # loss
                    'conr_w':0.1,
                    'conr_tau':1.,
                    # tune_epoch
                    'tune_epoch':80
                }
            },
            'mosei':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (1024, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (1024, 74, 35),
                    'text_out': 1024,
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE',
                    'H': 3.0,
                    'label_arange': (-3.0, 3.0),
                    'bin_width':0.5,
                    'label_markers':[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                    'metrics':{
                        "Has0_acc_2":0.,
                        "Has0_F1_score": 0.,
                        "Non0_acc_2":  0.,
                        "Non0_F1_score": 0.,
                        "Mult_acc_5": 0.,
                        "Mult_acc_7": 0.,
                        "MAE": 1e8,
                        "Corr": -100,
                        'Loss':-1.
                    },
                    # loss
                    'conr_w':0.1,
                    'conr_tau':1.,
                    # tune_epoch
                    'tune_epoch':20
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'text_out': 768,
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'MAE',
                    'H': 1.0,
                    'label_arange': (-1.0, 1.0),
                    'bin_width':0.2,
                    'label_markers':[-1.0, 0.0, 1.0],
                    # loss
                    'conr_w':0.1,
                    'conr_tau': 1.,
                    # warmup
                    'metrics' : {
                        "Mult_acc_2": 0.,
                        "Mult_acc_3": 0.,
                        "Mult_acc_5": 0.,
                        "F1_score": 0.,
                        "MAE": 1e8,
                        "Corr": -100, # Correlation Coefficient
                        'Loss':-1.
                    },
                    # tune_epoch
                    'tune_epoch':80
                }
            }
        }
        return tmp

    def __PDL(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
                'early_stop': 8,
                'pretrain_epoch':300,
                'update_epochs': 4,
                'warmup_epoch':5,
                'train_epoch':100
            },
            'debugParas':{
                'd_paras': ['batch_size', 'learning_rate_bert','learning_rate_audio', 'learning_rate_video', \
                            'learning_rate_other', 'weight_decay_bert','weight_decay_audio', 'weight_decay_video',\
                             'weight_decay_other', 'tf_layer_num','audio_out', 'video_out',\
                            'post_fusion_dim', 'post_fusion_dropout',\
                            'qap_embed_dim','qap_dropout', 'qap_depth','n_prompt','ffn_dim', 'max_lr_times'],
                'batch_size': random.choice([16,32,64]),
                'learning_rate_bert': random.choice([5e-5]),
                'learning_rate_audio': random.choice([1e-4]),
                'learning_rate_video': random.choice([1e-4]),
                'learning_rate_other': random.choice([1e-4]),
                'weight_decay_bert': random.choice([0.001]),
                'weight_decay_audio': random.choice([0.001]),
                'weight_decay_video': random.choice([0.001]),
                'weight_decay_other': random.choice([0.001]),
                # feature subNets
                'tf_layer_num':random.choice([2]),
                'audio_out': random.choice([768]),
                'video_out': random.choice([768]), 
                # post feature
                'post_fusion_dim': random.choice([128, 256]),
                'post_fusion_dropout': random.choice([0, 0.01, 0.1]),
                # qap
                'qap_embed_dim':random.choice([256,512]),
                'qap_dropout':random.choice([0, 0.01, 0.1]),
                'qap_depth':random.choice([2,3,4]),
                'n_prompt':random.choice([2, 4, 6, 8]),
                # 'resampler_gates':-1,
                # 'adapter_gate':-1,
                'ffn_dim':random.choice([32, 64, 128, 256]),
                # fusion
                # 'n_prompt_dic':{
                #     'a':2,
                #     'v':2
                # },
                # warmup
                'max_lr_times':random.choice([1.0, 0.1, 0.5])
            }
        }
        return tmp

    def get_config(self):
        return self.args