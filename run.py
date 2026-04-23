'''
   The code structure is based on Self-MM: https://github.com/thuiar/Self-MM
'''
import os
import gc
import pdb
import sys
import time
import random
import torch
import pynvml
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_tune import ConfigTune
from config.config_regression import ConfigRegression
import yaml

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)

def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir,\
                                        f'{args.time}-{args.modelName}-{args.datasetName}-{args.train_mode}.pth')
    
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        # load free-most gpu
        pynvml.nvmlInit()
        dst_gpu_id, min_mem_used = 0, 1e16
        for g_id in [0, 1]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        print(f'Find gpu: {dst_gpu_id}, use memory: {min_mem_used}!')
        logger.info(f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        args.gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # data
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer
    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    atio = ATIO().getTrain(args)

    # do train
    if 'train' in args.mode:
        atio.do_train(model, dataloader)
    # load pretrained model
    pre_model_save_path = args.model_save_path
    if args.mode == ['test']:
        pre_model_save_path = args.model_load_path
    assert os.path.exists(pre_model_save_path)
    logger.info(f"load saved model from {pre_model_save_path}")
    model.load_state_dict(torch.load(pre_model_save_path))
    model.to(device)

    # do test
    if args.tune_mode:
        # using valid dataset to debug hyper parameters
        results = atio.do_test(model, dataloader['valid'], mode="VALID")
    else:
        results = atio.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results

def run_tune(args, tune_times=50):

    # args.res_save_dir = os.path.join(args.res_save_dir, 'tunes')
    init_args = args
    has_debuged = [] # save used paras
    save_path = os.path.join(args.res_save_dir, args.tune_normals, \
                                f'{args.datasetName}-{args.modelName}-{args.train_mode}-tune.csv')
    args.save_path = save_path
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    for i in range(tune_times):
        # cancel random seed
        setup_seed(int(time.time()))
        pynvml.nvmlInit()
        args = init_args
        config = ConfigTune(args)
        args = config.get_config()
        logger.info(yaml.dump(dict(args), sort_keys=False, default_flow_style=False))
        # print debugging params
        logger.info("#"*40 + '%s-(%d/%d)' %(args.modelName, i+1, tune_times) + '#'*40)
        for k,v in args.items():
            if k in args.d_paras:
                logger.info(k + ':' + str(v))
        logger.info("#"*90)
        logger.info('Start running %s...' %(args.modelName))
        # restore existed paras
        if i == 0 and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in args.d_paras])
        # check paras
        cur_paras = [args[v] for v in args.d_paras]
        if cur_paras in has_debuged:
            logger.info('These paras have been used!')
            time.sleep(3)
            continue
        has_debuged.append(cur_paras)
        results = []
        try:
            for j, seed in enumerate(args.seeds):
                args.cur_time = j + 1
                setup_seed(seed)
                args.seed = seed
                # result, _ = run(args)
                result, _, _ = run(args)
                results.append(result)
        except Exception as e:
            logger.info(f"Error encontered with params, skip to next...")
            logger.info(f"Error:{e}")
            continue

        # save results to csv
        logger.info('Start saving results...')
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
        else:
            df = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in results[0].keys()])
        # stat results
        tmp = [args[c] for c in args.d_paras]
        for col in results[0].keys():
            values = [r[col] for r in results]
            tmp.append(round(sum(values) * 100 / len(values), 2))

        df.loc[len(df)] = tmp
        df.to_csv(save_path, index=None)
        logger.info('Results are saved to %s...\n\n\n\n\n' %(save_path))

def run_normal(args):
    # args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds
    # load other results
    save_path = os.path.join( os.path.join(args.res_save_dir, args.tune_normals), \
                        f'{args.time}/{args.datasetName}-{args.train_mode}.txt')
    args.save_path = save_path
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    
    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        # load config
        if args.train_mode == "regression":
            config = ConfigRegression(args)
        args = config.get_config()
        setup_seed(seed)
        args.seed = seed
        logger.info('Start running %s...' %(args.modelName))
        logger.info(yaml.dump(dict(args), sort_keys=False, default_flow_style=False))
        # runnning
        args.cur_time = i+1
        test_results = run(args)
        # restore results
        model_results.append(test_results)

    criterions = list(model_results[0].keys())
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Seed"] + criterions)
    # save results
    res_mean = ["mean"]
    res_std = ["std"]
    if not os.path.exists(os.path.dirname(args.save_path)):
                    os.makedirs(os.path.dirname(args.save_path))
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res_mean.append(mean)
        res_std.append(std)

    for i in range(len(model_results)):
        res = [str(seeds[i])] + list(np.round(np.array(list(model_results[i].values()))*100,2))
        df.loc[len(df)] = res
    df.loc[len(df)] = res_mean
    df.loc[len(df)] = res_std
    df.to_csv(save_path, sep='\t', index=None)
    with open(save_path, 'a') as f:
        f.write(f'\nsettings:\n{yaml.dump(dict(args), sort_keys=False, default_flow_style=False)}')
    logger.info('Results are added to %s...' %(save_path))

def set_log(args):
    log_file_path = f'logs/{args.time}-{args.modelName}-{args.datasetName}.log'
    if not os.path.exists(log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers[:]:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_tune', action="store_true",
                        help='tune parameters ?') # default False
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression / classification')
    parser.add_argument('--modelName', type=str, default='pdl',
                        help='support pdl')
    parser.add_argument('--datasetNames', nargs='+', default=['sims'],
                        help='support mosi/mosei/sims')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save models.')
    parser.add_argument('--res_save_dir', type=str, default='results/',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    
    # parser.add_argument('--pre_train_modality', type=list, default=['vision','audio'],
    #                     help="which modal to pretrain, choice:['text','vision','audio']")
    # parser.add_argument('--pre_train_use_uni_labels', action="store_false",
    #                     help='which labels to use in sims') # default True
    # parser.add_argument('--load_pretrain_model', action="store_true",
    #                     help='whether load pretained model or not') # default False
    
    # parser.add_argument('--use_FDS', action="store_true",
    #                     help='whether use FDS or not') # default False
    # parser.add_argument('--fusion_opt', type=str, default='qap',
    #                     help="fusion mono fetures in which way, choice:'concat'/'almt'/'qap'/'add'")
    # parser.add_argument('--hyper_layer_num', type=int, default=3,
    #                     help="the number of hyper-layer when fusion_opt=almt")
    parser.add_argument('--mode', type=list, default=['train','test'],
                        help="choice during running, train/test. Only 'test' means loading pretrained model from model_save_path and do test.")
    parser.add_argument('--model_load_path', type=str, default=None,
                        help="where to load pretrained fusion model")
    parser.add_argument('--seeds', nargs='+', type=int, default=[1234],
                        help="seeds")
    # qap
    # parser.add_argument('--qap_dropout', type=float, default=0.,
    #                     help="dropout rate in qap model")
    parser.add_argument('--use_anchor_prompt', action="store_true",
                        help="whether use prompt of text modality or not") # defalt False
    # parser.add_argument('--n_prompt', type=int, default=1,
    #                     help="the number of qap prompt")
    # parser.add_argument('--qap_depth', type=int, default=2,
    #                     help="the depth of qap layer")
    parser.add_argument('--as_anchor', type=str, default='t',
                        help="the anchor modality")
    # parser.add_argument('--adapter_gate', type=float, default=3.0,
    #                     help="")

    # add
    # parser.add_argument('--add_dropout', type=float, default=0.,
    #                     help="")
    
    # post_fusion_dim
    # parser.add_argument('--post_fusion_dim', type=int, default=128,
    #                     help="")
    # parser.add_argument('--ffn_dim', type=int, default=128,
    #                     help="the anchor modality")
    
    # parser.add_argument('--max_lr_times', type=float, default=1.0,
    #                     help="the depth of qap layer")
    # parser.add_argument('--loss_modality', nargs='+', default=['M'],
    #                     help="whitch modality to calculate loss.")
    # contrastive learning
    # parser.add_argument('--use_simi_matrix', action="store_true",
    #                     help='whether use use_simi_matrix or not') # default False
    # parser.add_argument('--dis_q_fuison_loss_weight', type=float, default=1.0,
    #                     help='the weight of distribution loss') # default False
    # loss
    # parser.add_argument('--conr_w', type=float, default=0.1,
    #                     help="the ts in conr")
    parser.add_argument('--lossLambda', type=float, default=0.1,
                         help="the ts in conr")
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    args.time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    args.pretrain_set = 'roberta'

    for data_name in args.datasetNames:
        args.datasetName = data_name
        logger = set_log(args)
        if args.is_tune:
            args.tune_normals = "tunes"
            run_tune(args, tune_times=80)
        else:
            args.tune_normals = 'normals'
            run_normal(args)