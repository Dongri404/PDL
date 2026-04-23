import os
import argparse

from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
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
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name]
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = '/mnt/disk1/zwj/Multimodality/data'
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE' 
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (1024, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE' ,
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
                        'Loss': -1
                    }
                }
            },
            'mosei':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
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
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE',
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
                    }
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'MAE',
                    'label_arange': (-1.0, 1.0),
                    'bin_width':0.2,
                    'label_markers':[-1.0, 0.0, 1.0],
                    'metrics' : {
                        "Mult_acc_2": 0.,
                        "Mult_acc_3": 0.,
                        "Mult_acc_5": 0.,
                        "F1_score": 0.,
                        "MAE": 1e8,
                        "Corr": -100, # Correlation Coefficient
                        'Loss':-1.
                    }
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
                'update_epochs': 4,
                'warmup_epoch':5
                # 'train_epoch':80

            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # result 1/2/3:
                    'train_epoch':80,
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-4,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-4,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.001,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'tf_layer_num':2,
                    'text_out': 1024, 
                    'audio_out': 768,
                    'video_out': 768, 
                    # post feature
                    'post_fusion_dim': 128,
                    'post_fusion_dropout': 0.01,
                    # qap
                    'qap_embed_dim':512,
                    'qap_dropout':0.01,
                    'qap_depth':2,
                    'n_prompt':8,
                    # 'resampler_gates':[1.0,1.0],
                    # 'adapter_gate':0.9,
                    'ffn_dim':32,
                    # fusion
                    # 'fusion_proj_dropout':0.1,
                    # 'dis_q_fuison_loss_weight': 0.1,
                    # 'labels':[-3, -2.0, -1.0, 0, 1.0, 2.0, 3.],
                    # loss
                    'conr_w':0.1,
                    'conr_tau':1.,
                    # warmup
                    'max_lr_times':1.0,
                    # res
                    'H': 3.0
                },
                'mosei':{
                    # result
                    'train_epoch':30,
                    'batch_size': 64,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-4,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-4,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.001,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'tf_layer_num':2,
                    'text_out': 1024, 
                    'audio_out': 768,
                    'video_out': 768, 
                    # post feature
                    'post_fusion_dim': 128,
                    'post_fusion_dropout': 0.0,
                    # qap
                    'qap_embed_dim':256,
                    'qap_dropout':0.1,
                    'qap_depth':2,
                    'n_prompt':6,
                    'ffn_dim':256,
                    # fusion
                    # 'dis_q_fuison_loss_weight': 0.1,
                    # loss
                    'conr_w':0.1,
                    'conr_tau':1.,
                    # warmup
                    'max_lr_times':1.,
                    # res
                    'H': 3.0
                },
                'sims':{
                    # result:
                    'train_epoch':80,
                    'batch_size': 64,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-4,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-4,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.001,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'tf_layer_num':2,
                    'text_out': 768, 
                    'audio_out': 768,
                    'video_out': 768,
                    # post feature
                    'post_fusion_dim': 256,
                    'post_fusion_dropout': 0.01,
                    # qap
                    'qap_embed_dim':512,
                    'qap_dropout':0.01,
                    'qap_depth':2,
                    'n_prompt':8,
                    'ffn_dim':64,
                    # fusion
                    # 'dis_q_fuison_loss_weight': 0.01,
                    # loss
                    'conr_w':0.1,
                    'conr_tau': 1.,
                    # warmup
                    'max_lr_times':1.,
                    # res
                    'H': 1.0
                },
            }
        }
        return tmp

    def get_config(self):
        return self.args