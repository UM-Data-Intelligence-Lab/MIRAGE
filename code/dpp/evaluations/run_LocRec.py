from recbole.quick_start import run_recbole, run_recboles
from recbole.config import Config
import argparse
from ast import arg
from pathlib import Path
import os

def run_LocRec_task(dataset_path,model,dataset,setting,cuda):
    parameter_dict = {
        'data_path' : dataset_path,
        'metrics': ['MRR', 'NDCG', 'Hit',],
        'valid_metric': 'MRR@10',
        'topk': [5,10],
        'load_col': {'inter':['user_id', 'item_id', 'rating'],}, 
        'epochs' : 50,
        'train_neg_sample_args': None if setting == '1' else {'distribution': 'uniform', 'sample_num': 1, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0},  
        'train_batch_size':512,
        'eval_batch_size':512, 
        'gpu_id':cuda
    }

    run_recbole(model=model, dataset=dataset,config_dict=parameter_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savepath", type=str, default='', help="name of savepath")
    parser.add_argument("--dataset_name", type=str, default='', help="name of dataset")
    parser.add_argument("--model_name", type=str, default='BPR', help="name of model")
    parser.add_argument("--change_setting", type=str, default='0', help="parameter changing")
    parser.add_argument("--cuda", type=str, default='0', help="index of cuda")
    args, _ = parser.parse_known_args()
    run_LocRec_task(args.savepath,args.model_name,args.dataset_name,args.change_setting,args.cuda)