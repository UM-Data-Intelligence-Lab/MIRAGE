import argparse
import os
from dpp.evaluations.preprocessing import preprocessing_data
from dpp.evaluations import Get_Statistical_Metrics,run_SemLoc_task,run_EpiSim_task
import torch
import pandas as pd
import numpy as np

dataset_file_path = os.path.abspath(os.path.dirname(os.getcwd()))

def collect_results(dataset_list,task_lists):
    def scan_result_files(file_dir):
        list_files=[]
        for files in os.listdir(file_dir):
            list_files.append(files)
        return list_files
    results = []
    for dataset_name in dataset_list:
        data_frame=pd.DataFrame(columns=['method','mrr@5','mrr@10','ndcg@5','ndcg@10','hit@5','hit@10'])
        for model in task_lists:
            files = scan_result_files(f"./log/{model}")
            file = [f for f in files if dataset_name in f][0]
            with open(f'./log/{model}/{file}','r') as f:
                data=f.readlines()
                res_list=eval((data[-1].strip('\n').split(': ')[1])[12:-1])
                data = [item[1] for item in res_list]
                data.insert(0,model)
                data_frame.loc[len(data_frame)]=data
        results.append(data_frame)
    real_res = results[0].iloc[:,1:]
    gene_res = results[1].iloc[:,1:]
    absolute_differences = (gene_res - real_res).abs()
    relative_differences = absolute_differences / real_res
    MAPE = relative_differences.mean().mean()
    relative_differences_2 = np.square(relative_differences)
    MSPE = relative_differences_2.mean().mean()
    return MAPE,MSPE

def run_LocRec(task,dataset,cuda,experiment_comments,generated_only):
    print(f'preprocessing data for {task} Task')
    datasets = [dataset+'_test.pkl',f'{dataset}_generated.pkl' if experiment_comments=="" else f'{dataset}_{experiment_comments}_generated.pkl']
    datasets_prefix = [dataset+'_test_for_general',f'{dataset}_generated_for_general' if experiment_comments=="" else f'{dataset}_{experiment_comments}_generated_for_general']
    for source_file in datasets:
        preprocessing_data(f'{dataset_file_path}/data/{dataset}', source_file, task == 'NexLoc')
    LocRec_Tasks=['DMF','LightGCN','MultiVAE','NeuMF','BPR']
    LocRec_settings=['0','0','0','0','0']    
    for index,dataset_name in enumerate(datasets_prefix):
        if generated_only and index==0:
            continue
        for idx,model_name in enumerate(LocRec_Tasks):
            cmd=(f'python ./dpp/evaluations/run_LocRec.py --savepath {dataset_file_path}/data/{dataset} --model_name {model_name} --dataset_name {dataset_name}  --change_setting {LocRec_settings[idx]} --cuda {cuda}')   
            os.system(cmd)
    LocRec_MAPE,LocRec_MSPE = collect_results(datasets_prefix,LocRec_Tasks)
    return LocRec_MAPE,LocRec_MSPE

def run_NexLoc(task,dataset,cuda,experiment_comments,generated_only):
    print(f'preprocessing data for {task} Task')
    datasets = [dataset+'_test.pkl',f'{dataset}_generated.pkl' if experiment_comments=="" else f'{dataset}_{experiment_comments}_generated.pkl']
    datasets_prefix = [dataset+'_test_for_sequential',f'{dataset}_generated_for_sequential' if experiment_comments=="" else f'{dataset}_{experiment_comments}_generated_for_sequential']
    for source_file in datasets:
        preprocessing_data(f'{dataset_file_path}/data/{dataset}',source_file, task == 'NexLoc')
    NexLoc_Tasks=['FPMC','BERT4Rec','Caser','SRGNN','SASRec']
    NexLoc_settings=['0','1','1','1','1'] 
    for index,dataset_name in enumerate(datasets_prefix):
        if generated_only and index==0:
            continue
        for idx,model_name in enumerate(NexLoc_Tasks):
            cmd=(f'python ./dpp/evaluations/run_NexLoc.py --savepath {dataset_file_path}/data/{dataset} --model_name {model_name} --dataset_name {dataset_name}  --change_setting {NexLoc_settings[idx]} --cuda {cuda}')   
            os.system(cmd)
    NexLoc_MAPE,NexLoc_MSPE = collect_results(datasets_prefix,NexLoc_Tasks)
    return NexLoc_MAPE,NexLoc_MSPE

def run_SemLoc(task,dataset, experiment_comments):
    print(f'run {task} Task')
    test_data = torch.load(f'{dataset_file_path}/data/{dataset}/{dataset}_test.pkl')
    generated_data_path = f'{dataset_file_path}/data/{dataset}/{dataset}_generated.pkl' if experiment_comments=="" else f'{dataset_file_path}/data/{dataset}/{dataset}_{experiment_comments}_generated.pkl'
    generated_data = torch.load(generated_data_path)
    test_seqs = test_data.get('sequences')
    generated_seqs = generated_data.get('sequences')
    data_all = torch.load(f'{dataset_file_path}/data/{dataset}/{dataset}.pkl')
    poi_label_dict = data_all['poi_category']
    SemLoc_MAPE,SemLoc_MSPE = run_SemLoc_task(test_seqs,generated_seqs,poi_label_dict)
    return SemLoc_MAPE,SemLoc_MSPE

def run_EpiSim(task,dataset, experiment_comments, init_exposed_num, exp_num, max_weeks):
    print(f'run {task} Task')
    if 'Tokyo' in dataset:
        max_weeks = 12
    test_data = torch.load(f'{dataset_file_path}/data/{dataset}/{dataset}_test.pkl')
    generated_data_path = f'{dataset_file_path}/data/{dataset}/{dataset}_generated.pkl' if experiment_comments=="" else f'{dataset_file_path}/data/{dataset}/{dataset}_{experiment_comments}_generated.pkl'
    generated_data = torch.load(generated_data_path)
    test_seqs = test_data.get('sequences')
    generated_seqs = generated_data.get('sequences')
    EpiSim_MAPE,EpiSim_MSPE = run_EpiSim_task(test_seqs,generated_seqs, init_exposed_num, exp_num, max_weeks)
    return EpiSim_MAPE,EpiSim_MSPE

def run_Statistical(dataset, experiment_comments):
    print(f'get Statistical Performance')
    test_data = torch.load(f'{dataset_file_path}/data/{dataset}/{dataset}_test.pkl')
    generated_data_path = f'{dataset_file_path}/data/{dataset}/{dataset}_generated.pkl' if experiment_comments=="" else f'{dataset_file_path}/data/{dataset}/{dataset}_{experiment_comments}_generated.pkl'
    generated_data = torch.load(generated_data_path)
    test_seqs = test_data.get('sequences')
    generated_seqs = generated_data.get('sequences')
    JSD_Values = Get_Statistical_Metrics(test_seqs,generated_seqs)
    return JSD_Values


def evaluation(task, dataset, cuda, results_log, experiment_comments, generated_only, init_exposed_num, exp_num, max_weeks):
    if task != "Stat":
        if task == "LocRec" :
            MAPE,MSPE = run_LocRec(task, dataset, cuda, experiment_comments, generated_only)
        elif task == "NexLoc" :
            MAPE,MSPE = run_NexLoc(task, dataset, cuda, experiment_comments, generated_only)
        elif task == "SemLoc" :
            MAPE,MSPE = run_SemLoc(task, dataset, experiment_comments)
        elif task == "EpiSim" :
            MAPE,MSPE = run_EpiSim(task, dataset, experiment_comments, init_exposed_num, exp_num, max_weeks)
        with open (results_log, "a+") as f:
            f.writelines(f"{dataset} {task} MAPE : {MAPE}, MSPE : {MSPE} \n")
    else:
        JSD_Values = run_Statistical(dataset, experiment_comments)
        with open (results_log, "a+") as f:
            f.writelines(f"Distance: {JSD_Values['Distance']}, Radius: {JSD_Values['Radius']}, Interval: {JSD_Values['Interval']}, DailyLoc: {JSD_Values['DailyLoc']}, Category: {JSD_Values['Category']}\n")      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',default='Stat', type=str, choices=['LocRec', 'NexLoc', 'SemLoc', 'EpiSim', 'Stat'])
    parser.add_argument('--cuda',default=0,type=str)
    parser.add_argument('--datasets',default='NewYork',type=str)
    parser.add_argument('--results',default='', type=str)
    parser.add_argument("--experiment_comments", type=str, default="")
    parser.add_argument('--init_exposed_num',default=50,type=int)
    parser.add_argument('--exp_num',default=15,type=int)
    parser.add_argument('--max_weeks',default=15,type=int)
    parser.add_argument("--generated_only",default=False,action='store_true')
    args = parser.parse_args()
    if len(args.results) == 0:
        args.results = args.datasets+"_"+args.experiment_comments + "_Evaluation_results.txt" if args.experiment_comments != '' else args.datasets +"_Evaluation_results.txt"
    evaluation(args.task,args.datasets,args.cuda,args.results,args.experiment_comments,args.generated_only,args.init_exposed_num,args.exp_num,args.max_weeks)
