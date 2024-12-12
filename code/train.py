import dpp
import numpy as np
import torch
import os
import time
from dpp.utils import save_generated_seq, save_seqs, get_logger
import argparse
import setproctitle
from torch.utils.data import TensorDataset
from parameters import Setting

torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_printoptions(profile='full') 

args = Setting()
args.parse()

logger = get_logger(f"{args.dataset_name.split('.')[0]}_traing_"+time.strftime("%Y%m%d_%H%M%S")+'.log')
logger.info(vars(args))

# Config
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

proc_title = "MIRAGE"
setproctitle.setproctitle(proc_title)

dataset_file_path = os.path.abspath(os.path.dirname(os.getcwd()))
data_path = os.path.join(dataset_file_path, 'data')


# Load the data
dataset, num_marks, locations, sequence_count, poi_gps_dict, poi_category = dpp.data.load_dataset(args.dataset_name)
d_train, _, d_test = dataset.train_val_test_split(seed=seed)

sequence_count = d_train.re_calculate_seq_idx(True)
_ = d_test.re_calculate_seq_idx(False)


save_seqs(d_train,os.path.join(data_path,args.dataset_name.split('.')[0]),args.dataset_name.split('.')[0]+'_train.pkl',num_marks,locations,sequence_count,poi_gps_dict,logger)
save_seqs(d_test,os.path.join(data_path,args.dataset_name.split('.')[0]),args.dataset_name.split('.')[0]+'_test.pkl',num_marks,locations,sequence_count,poi_gps_dict,logger)


dl_train = d_train.get_dataloader(batch_size=args.batch_size, shuffle=True)
dl_test = d_test.get_dataloader(batch_size=args.batch_size, shuffle=False)

logger.info('Building model...')
mean_log_inter_time, std_log_inter_time = d_train.get_inter_time_statistics()

model = dpp.models.MODEL(num_marks=d_train.num_marks, 
                            sequence_count=sequence_count, 
                            num_locations=locations,
                            mean_log_inter_time=mean_log_inter_time, 
                            std_log_inter_time=std_log_inter_time, 
                            tpp_sequence_embedding_size=args.tpp_sequence_embedding_size, 
                            context_size=args.tpp_context_size, 
                            tpp_mark_embedding_size=args.tpp_mark_embedding_size, 
                            time_interval_embedding_size=args.time_interval_embedding_size,
                            tpp_loc_embedding_size=args.tpp_loc_embedding_size,
                            tpp_time_embedding_size=args.tpp_time_embedding_size,
                            tpp_num_mix_components=args.tpp_num_mix_components, 
                            vae_hidden=args.vae_hidden,
                            vae_latent=args.vae_latent,
                            vae_mse_lambda=args.vae_mse_lambda,
                            poi_category=poi_category,
                            poi_gps_dict=poi_gps_dict,
                            tpp_sequential_type=args.tpp_sequential_type,)

opt = torch.optim.Adam(model.parameters(), weight_decay=args.regularization, lr=args.learning_rate)

logger.info('Starting Training...')

def sampling(num, data_path, logger):
    generated_seqs=[]
    while len(generated_seqs)<num:
        generated=model.sample(t_end = 7, batch_size = 1)
        generated=generated.to_list(poi_gps_dict)
        if len(generated[0].checkins) >= args.min_seq_len:
            generated_seqs+=generated
    save_generated_seq(generated_seqs, data_path, logger, args.experiment_comments, args.dataset_name.split('.')[0])

traing_losses=[]
for epoch in range(1,args.max_epochs+1):
    model.train()
    losses=[]
    epoch_start = time.time()
    for batch in dl_train:
        opt.zero_grad()
        loss = model.log_prob(batch).mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    epoch_end = time.time()
    traing_losses.append(np.mean(losses))
    if epoch % args.display_step == 0:
        logger.info(f"Epoch {epoch:4d}: loss_train = {np.mean(losses):.4f}")
        logger.info('One training need {:.2f}s'.format(epoch_end - epoch_start))

logger.info('User VAE Training')
seq_idx = torch.tensor(np.arange(1,sequence_count))
dataset_seq_id = TensorDataset(seq_idx)
dataloader = torch.utils.data.DataLoader(dataset_seq_id, batch_size=args.vae_batch_size)
for epoch in range(args.vae_epochs):
    model.train()
    MSE = []
    KLD = []
    model.sequence_encoder.requires_grad = False
    for batch in dataloader:
        opt.zero_grad()
        s_seq_emb = model.sequence_encoder(batch[0].long())
        mse,kld = model.user_vae(s_seq_emb)
        MSE.append(mse.item())
        KLD.append(kld.item())
        loss = mse + kld
        loss.backward()
        opt.step()
    logger.info(f"Epoch {epoch:4d}: mse = {np.mean(MSE):.10f}, kld = {np.mean(KLD):.10f}")
logger.info('User VAE Training Done')
logger.info('\n')
model.eval()
logger.info('Start to generate data')
sampling(int(len(d_test)), data_path, logger)
logger.info('Generate data Done')