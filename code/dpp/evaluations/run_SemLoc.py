import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

def calculate_seq_per_poi(dataset,poi_label_dict,t_day,t_granularity):
    checkin_seq_per_poi={}
    for uid,seq in enumerate(dataset):
        checkin_hour = (np.array(seq['arrival_times'])*t_granularity).astype(int)
        for pid,poi in enumerate(seq['checkins']):
            if poi != 0:
                if poi not in checkin_seq_per_poi.keys():       
                    # t_day*t_granularity is hour-in-week representation  for my setting             
                    checkin_seq_per_poi[poi] = {'chekcin_array':np.zeros((1,t_day*t_granularity)),'total_check_in':0,'label':poi_label_dict[poi]-1} # 0 is padding and does not exist in dictionary for my setting
                checkin_seq_per_poi[poi]['chekcin_array'][0,checkin_hour[pid]]+=1
                checkin_seq_per_poi[poi]['total_check_in']+=1
    return checkin_seq_per_poi

def generate_training_data(checkin_seq_per_poi,limit=1):
    X=[]
    y=[]
    for key,values in checkin_seq_per_poi.items():
        if values['total_check_in']>=limit:
            X.append(values['chekcin_array'])
            y.append(values['label'])
    X = np.concatenate(X)
    y = np.array(y)
    return X,y


def run_SemLoc_task(test_data,generated_data,poi_label_dict,t_day=7,t_granularity=24):
    data_transformed_list = []
    datalist = [test_data,generated_data]
    for idx,data in enumerate(datalist):
        checkin_seq_per_poi = calculate_seq_per_poi(data,poi_label_dict,t_day,t_granularity)
        data_transformed_list.append(checkin_seq_per_poi)

    all_result =[]
    models=[tree.DecisionTreeClassifier(),GaussianNB(),sklearn.neighbors.KNeighborsClassifier(),sklearn.svm.LinearSVC(),sklearn.linear_model.LogisticRegression(multi_class="multinomial")]
    model_names=['tree.DecisionTreeClassifier','naive_bayes.GaussianNB','neighbors.KNeighborsClassifier','svm.LinearSVC','linear_model.LogisticRegression']
    metrics_names=['accuracy',  'f1_micro',  'f1_macro', ]
    for idx,data in enumerate(data_transformed_list):
        results = pd.DataFrame(columns=model_names)
        X,y = generate_training_data(data,2)
        for modelid,model in enumerate(models):
            res=[]
            for metricid,metric in enumerate(metrics_names):
                cv_scores = cross_val_score(model, X, y, cv=10, scoring=metric)
                res.append(np.mean(cv_scores))
            results[model_names[modelid]] = res
        all_result.append(results)
    real_res = all_result[0]
    gene_res = all_result[1]
    absolute_differences = (gene_res - real_res).abs()
    relative_differences = absolute_differences / real_res
    MAPE = relative_differences.mean().mean()
    relative_differences_2 = np.square(relative_differences)
    MSPE = relative_differences_2.mean().mean()
    return MAPE,MSPE

