import torch
import numpy as np
import pandas as pd
import os

def preprocessing_data(dicts,source_file,mode):  
    target_file_prefix = source_file.split('.')[0]
    data=torch.load(dicts+'/'+source_file)
    seqs=data.get('sequences')
    user_ids=[]
    checkin_times=[]
    lons=[]
    lats=[]
    checkins=[]
    for i,item in enumerate(seqs):
        if len(item['checkins']) >1 :
            uid=[i]*len(item['checkins'])
            user_ids.extend(uid)
            checkin_times.extend(item['arrival_times'])
            lat=np.array(item['gps'])[:,0].tolist()
            lats.extend(lat)
            lon=np.array(item['gps'])[:,1].tolist()
            lons.extend(lon)
            checkins.extend(item['checkins'])
    data_transformed=pd.DataFrame()
    data_transformed['user_id']=user_ids
    data_transformed['checkin_times']=checkin_times
    data_transformed['lat']=lats
    data_transformed['lon']=lons
    data_transformed['checkins']=checkins

    if mode:
        if not os.path.exists(f'{dicts}/{target_file_prefix}_for_sequential'):
            os.makedirs(f'{dicts}/{target_file_prefix}_for_sequential')
        data_4_seqRec=data_transformed.loc[:,['user_id','checkins','checkin_times']]
        data_4_seqRec.columns=['user_id:token','item_id:token','timestamp:float']
        data_4_seqRec.to_csv(f'{dicts}/{target_file_prefix}_for_sequential/{target_file_prefix}_for_sequential.inter', index=False, sep='\t')
    else:
        if not os.path.exists(f'{dicts}/{target_file_prefix}_for_general'):
            os.makedirs(f'{dicts}/{target_file_prefix}_for_general')
        data_4_locRec=data_transformed.loc[:,['user_id','checkins']]
        data_4_locRec.columns=['user_id:token','item_id:token']
        data_4_locRec=data_4_locRec.groupby(['user_id:token','item_id:token']).size().reset_index(name='count')
        data_4_locRec.columns=['user_id:token','item_id:token','rating:float']
        data_4_locRec.to_csv(f'{dicts}/{target_file_prefix}_for_general/{target_file_prefix}_for_general.inter', index=False, sep='\t')