import dpp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from dpp.data.batch import Batch
from dpp.utils import diff

class VAE(nn.Module):
    def __init__(self,input_dim, hidden_dim, latent_dim,vae_mse_lambda):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc4 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.mseloss = nn.MSELoss(reduction = 'mean')
        self.vae_mse_lambda = vae_mse_lambda
    def encode(self, x):
        h1 = self.fc1(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.fc4(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return self.loss(reconstructed, x, mu, logvar)

    def loss(self,recon_x, x, mu, logvar):

        MSE = self.mseloss(recon_x, x.view(-1, self.input_dim))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE/self.vae_mse_lambda,KLD


class MIRAGE(nn.Module):
    def __init__(
        self,
        num_marks: int,
        sequence_count : int,
        num_locations : int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        tpp_sequence_embedding_size: int = 32,
        context_size: int = 32,
        tpp_mark_embedding_size: int = 32,
        time_interval_embedding_size: int = 32,
        tpp_loc_embedding_size: int = 32,
        tpp_time_embedding_size: int = 16,
        vae_hidden: int = 32,
        vae_latent: int = 2,
        vae_mse_lambda: int = 1,
        poi_category: dict = {},
        poi_gps_dict: dict = {},
        tpp_sequential_type: str = "GRU",

    ):
        super().__init__()
        self.num_marks=num_marks
        self.sequence_count=sequence_count
        self.num_locations=num_locations
        self.mean_log_inter_time=mean_log_inter_time
        self.std_log_inter_time=std_log_inter_time
        self.tpp_sequence_embedding_size=tpp_sequence_embedding_size
        self.context_size=context_size
        self.tpp_mark_embedding_size=tpp_mark_embedding_size
        self.time_interval_embedding_size=time_interval_embedding_size
        self.tpp_loc_embedding_size=tpp_loc_embedding_size
        self.tpp_time_embedding_size=tpp_time_embedding_size
        self.vae_hidden=vae_hidden
        self.vae_latent=vae_latent
        self.vae_mse_lambda=vae_mse_lambda
        self.poi_category=poi_category
        self.poi_gps_dict=poi_gps_dict
        self.tpp_sequential_type=tpp_sequential_type

        self.sequence_encoder = nn.Embedding(self.sequence_count,self.tpp_sequence_embedding_size)
        self.loc_embedding = nn.Embedding(self.num_locations, self.tpp_loc_embedding_size, padding_idx = 0)
        self.delta_t_hour_emb = nn.Embedding(169,self.time_interval_embedding_size)
        self.num_features = 1 + self.tpp_mark_embedding_size + self.tpp_loc_embedding_size
        self.mark_embedding = nn.Embedding(self.num_marks, self.tpp_mark_embedding_size, padding_idx = 0)
        self.mark_linear = nn.Linear(self.context_size+self.tpp_time_embedding_size+self.tpp_sequence_embedding_size, self.num_marks)
        self.time_embedding = nn.Embedding(169, self.tpp_time_embedding_size)
        self.context_init = nn.Parameter(torch.zeros(self.context_size))  # initial state of the RNN
        self.transform = nn.Linear(self.num_features, self.context_size)
        self.sequential = getattr(nn, self.tpp_sequential_type)(input_size=self.context_size, hidden_size=self.context_size, batch_first=True)
        self.revisit_explore_linear = nn.Sequential(
                nn.Linear(self.context_size + self.tpp_time_embedding_size+self.tpp_sequence_embedding_size+self.tpp_mark_embedding_size, 8*self.context_size),
                nn.ReLU(),
                nn.Linear(8*self.context_size, 2*self.context_size),
                nn.ReLU(),
                nn.Linear(2*self.context_size, 2),
        )
        self.delta_t_mlp_hour = nn.Sequential(nn.Linear(self.time_interval_embedding_size, 2*self.time_interval_embedding_size),
            nn.ReLU(),
            nn.Linear(2*self.time_interval_embedding_size, self.time_interval_embedding_size),
            nn.ReLU(),
            nn.Linear(self.time_interval_embedding_size, 1),
            )
        
        self.poi_linear = nn.Linear(self.context_size + self.tpp_time_embedding_size + self.tpp_sequence_embedding_size + self.tpp_mark_embedding_size, self.num_locations)
        self.user_vae = VAE(self.tpp_sequence_embedding_size, self.vae_hidden, self.vae_latent, self.vae_mse_lambda)

    def get_features(self, batch: dpp.data.batch) -> torch.Tensor:
        features = torch.log(batch.inter_times + 1e-8).unsqueeze(-1)  
        features = (features - self.mean_log_inter_time) / self.std_log_inter_time
        poi_emb = self.loc_embedding(batch.checkins)
        mark_emb = self.mark_embedding(batch.marks)  
        features = torch.cat([features, mark_emb], dim=-1)
            
        features = torch.cat([features,poi_emb], dim=-1)
        return features  

    def get_context(self, features: torch.Tensor, batch: dpp.data.batch, remove_last: bool = True) -> torch.Tensor:
        """
        Get the context (history) embedding from the sequence of events.

        Args:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)
            remove_last: Whether to remove the context embedding for the last event.

        Returns:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size) if remove_last == Ture
                shape (batch_size, seq_len + 1, context_size) if remove_last == False

        """
        features = self.transform(features.float())
        context = self.sequential(features)[0]        

        context_init = self.context_init[None, None, :].expand(context.shape[0], 1, -1)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        if remove_last:
            context = context[:, :-1, :]
        context = torch.cat([context_init, context], dim=1)
        return context

    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raise NotImplementedError()

    def get_delta_t_untill_next(self, arrival_times):
        arrival_time_prev = arrival_times[:,:-1]
        arrival_time_cur = arrival_times[:,1:]
        len_s = arrival_time_prev.shape[1]
        arrival_time_prev = arrival_time_prev.unsqueeze(1).repeat(1,len_s,1)
        arrival_time_cur = arrival_time_cur.unsqueeze(2)
        delta_t = (arrival_time_cur - arrival_time_prev).double()
        delta_t = torch.where(delta_t < 0.0, 0.0, delta_t)
        return delta_t

    def get_poi_init_score(self, context_init:torch.Tensor, sample: bool = False) -> torch.Tensor:
        init_score = self.poi_linear(context_init)
        if sample:
            init_score = init_score[:,:,1:]
        init_score = torch.softmax(init_score,dim=-1)
        return init_score


    def revisit_mechanism(self, context: torch.Tensor, poi: torch.Tensor, arrival_times: torch.Tensor, sample: bool = False) -> torch.Tensor:
        batch_size, seq_len = context.shape[0], context.shape[1]
        delta_t = self.get_delta_t_untill_next(arrival_times)*24
        poi_matrix= poi.unsqueeze(2).repeat(1,1,delta_t.shape[2]).transpose(-1,-2)
        delta_t_mask = delta_t.bool().float()
        poi_prev = (poi_matrix * delta_t_mask).float()

        delta_t = delta_t.long()
        delta_t = torch.where(delta_t > 168, 168, delta_t)
        delta_t = self.delta_t_hour_emb(delta_t)
        delta_t = self.delta_t_mlp_hour(delta_t).squeeze(3)

        #revisit score
        delta_t.masked_fill_(~(delta_t_mask.bool()), float('-inf'))
        delta_t_weights = F.softmax(delta_t, dim=-1)

        revisit_score = torch.zeros(batch_size, seq_len, self.num_locations, device=context.device)
        revisit_score.scatter_add_(2, poi_prev.long(), delta_t_weights)
        revisit_score_masked = (revisit_score == 0)
        aggregated_revisit_score = revisit_score

        if sample:
            aggregated_revisit_score = aggregated_revisit_score[:,:,1:]
        #explore score mask
        explore_score_masked = ~revisit_score_masked
        return aggregated_revisit_score, explore_score_masked


    def explore_mechanism(self, context: torch.Tensor, explore_score_masked: bool, sample: bool = False) -> torch.Tensor:
        explore_score = self.poi_linear(context)
        explore_score.masked_fill_(explore_score_masked, float('-inf'))
        if sample:
            explore_score = explore_score[:,:,1:]
        return torch.softmax(explore_score,dim=-1)

    def get_poi_prob(self, context: torch.Tensor, batch: dpp.data.batch, sample: bool = False)-> torch.Tensor:
        
        poi = batch.checkins
        arrival_times = batch.inter_times.cumsum(-1)
        revisits = batch.revisit
        if not sample:
            poi = poi[:,:-1]
        poi_init_score = self.get_poi_init_score(context[:,[0],:],sample)

        revisit_out = self.revisit_explore_linear(context[:,1:,:]).squeeze()
        revisit_out = torch.softmax(revisit_out,dim=-1)

        revisit_score, explore_score_masked = self.revisit_mechanism(context[:,1:,:], poi, arrival_times, sample)
        explore_score = self.explore_mechanism(context[:,1:,:], explore_score_masked, sample)
        if sample:
            revisit_dist = Categorical(probs=revisit_out)
            revisit_prob = revisit_dist.sample()
            revisit_prob = revisit_prob.unsqueeze(-1)
        else:
            revisit_prob = revisits[:,1:].unsqueeze(2)
        poi_final_score = revisit_score * revisit_prob + (1-revisit_prob)*explore_score
        poi_final_score = torch.concat([poi_init_score, poi_final_score],dim=1)
        return poi_final_score,revisit_out

    def log_prob(self, batch: dpp.data.batch) -> torch.Tensor:

        features = self.get_features(batch)
        context = self.get_context(features, batch)
        if isinstance(batch.seq_idx, list):
            seqs = torch.cat(batch.seq_idx)
        else :
            seqs = batch.seq_idx 

        zero_hour = torch.zeros(1).expand(context.shape[0], -1).long()
        hour = (batch.inter_times.cumsum(-1)*24).long()
        
        hour_next = torch.where(hour > 168, 168, hour)
        hour_current = torch.cat([zero_hour,hour_next],dim=-1)
        
        hour_emb_current =  self.time_embedding(hour_current)[:,:-1,:]
        hour_emb_next = self.time_embedding(hour_next)

        
        
        seq_emb = self.sequence_encoder(seqs).unsqueeze(1).repeat(1,features.shape[1],1)
        context_for_intertimes = torch.cat([context, seq_emb, hour_emb_current], dim=-1)

        inter_time_dist = self.get_inter_time_dist(context_for_intertimes)
        inter_times = batch.inter_times.clamp(1e-10)
        log_p_time = inter_time_dist.log_prob(inter_times)  

        last_event_idx = batch.mask.sum(-1, keepdim=True).long()  
        log_surv_all = inter_time_dist.log_survival_function(inter_times)  
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  

        context_for_category=torch.concat([context,seq_emb,hour_emb_next],dim=-1)
        mark_logits = torch.log_softmax(self.mark_linear(context_for_category), dim=-1)  
        mark_dist = Categorical(logits=mark_logits)
        log_p_time += mark_dist.log_prob(batch.marks)  

        log_p_time *= batch.mask  
        time_ll = log_p_time.sum(-1) + log_surv_last    


        next_mark_emb = self.mark_embedding(batch.marks)
        context_for_poi = torch.cat([context_for_category,next_mark_emb], dim=-1)

        poi_prob,revisit_prob = self.get_poi_prob(context_for_poi,batch)

        pois_dist = Categorical(probs=poi_prob)
        revisit_dist = Categorical(probs=revisit_prob)

        log_p_poi = pois_dist.log_prob(batch.checkins)
        log_p_poi *= batch.mask # (batch_size, seq_len)
        log_p_revisit = revisit_dist.log_prob(batch.revisit.float()[:,1:])
        log_p_revisit *= batch.mask[:,1:]
        poi_ll = log_p_poi.sum(-1) + log_p_revisit.sum(-1)
        
        return -time_ll-poi_ll

    def sample(self, t_end: float, batch_size: int = 1) -> Batch:  
        z = torch.randn(batch_size,self.vae_latent)
        tpp_seq_emb = self.user_vae.decode(z).view(batch_size,1,-1)

        tpp_context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)
        tpp_next_context = tpp_context_init

        zero_hour = torch.zeros(1).expand(batch_size, -1).long()
        hour_current = zero_hour       
        tpp_hours_emb_current =  self.time_embedding(hour_current)
        
        inter_times = torch.empty(batch_size, 0)
        pois = torch.empty(batch_size, 0)
        gps = torch.empty(batch_size, 0)
        marks = torch.empty(batch_size, 0, dtype=torch.long)
        
        generated = False
        while not generated:
            tpp_next_context_for_delta_t = torch.cat([tpp_next_context, tpp_seq_emb, tpp_hours_emb_current], dim=-1)[:,-1,:]
            inter_time_dist = self.get_inter_time_dist(tpp_next_context_for_delta_t)            
            next_inter_times = inter_time_dist.sample() 
            next_inter_times = next_inter_times.clamp_min_(1.2e-4)
            inter_times = torch.cat([inter_times, next_inter_times.unsqueeze(1)], dim=1)  

            hour = (inter_times.cumsum(-1)*24).long()
            hour = torch.where(hour > 168, 168, hour)
            tpp_hours_emb = self.time_embedding(hour)
            tpp_hours_emb_next = tpp_hours_emb[:, [-1], :]
            tpp_next_context_for_mark = torch.cat([tpp_next_context, tpp_seq_emb, tpp_hours_emb_next], dim=-1)
            mark_logits = torch.log_softmax(self.mark_linear(tpp_next_context_for_mark)[:,:,1:], dim=-1)[:,[-1],:] 
            mark_dist = Categorical(logits=mark_logits)
            next_marks = mark_dist.sample()+1  
            marks = torch.cat([marks, next_marks], dim=1)

            if pois.shape[1] == 0:
                next_mark_emb = self.mark_embedding(next_marks)
                tpp_next_context_for_poi = torch.cat([tpp_next_context_for_mark, next_mark_emb], dim=-1)
                poi_prob_next = self.get_poi_init_score(tpp_next_context_for_poi,True)[:, -1, :]
            else :
                mark_emb = self.mark_embedding(marks)
                seq_emb = tpp_seq_emb.repeat(1,context.shape[1],1)
                context_for_poi = torch.cat([context,seq_emb,tpp_hours_emb,mark_emb], dim=-1)
                one_batch = Batch(inter_times=inter_times, checkins=pois.long(), day_hour=None, seq_idx=None, mask=torch.ones_like(inter_times), marks=marks,)
                poi_prob,_ = self.get_poi_prob(context_for_poi,one_batch,sample=True)
                poi_prob_next = poi_prob[:, -1, :]
            poi_dist = Categorical(probs=poi_prob_next)
            next_pois = poi_dist.sample() + 1
            next_gps = []
            for idx in range(len(next_pois)):
                gps_str=self.poi_gps_dict[next_pois[idx].item()].split(',')
                next_gps.append(torch.tensor([float(gps_str[0]),float(gps_str[1])]))

            next_gps = torch.stack(next_gps).unsqueeze(1)
            if gps.shape[1] == 0:
                gps = next_gps
            else: 
                gps = torch.cat([gps, next_gps], dim=1)
            pois = torch.cat([pois, next_pois.unsqueeze(1)], dim=1)  
            
            next_marks = torch.zeros(batch_size, 1, dtype=torch.long)
            for i in range(batch_size):
                next_marks[i]=self.poi_category[next_pois[i].item()]
            marks[:,[-1]]=next_marks

            with torch.no_grad():
                generated = inter_times.sum(-1).min() > t_end
            one_batch = Batch(inter_times=inter_times, checkins=pois.long(), day_hour=None, seq_idx=None, mask=torch.ones_like(inter_times), marks=marks,)
            features = self.get_features(one_batch)  
            context = self.get_context(features, one_batch, remove_last=False)  
            tpp_next_context = context[:, [-1], :]  
            tpp_hours_emb_current = tpp_hours_emb_next
            
        arrival_times = inter_times.cumsum(-1)  
        inter_times = diff(arrival_times.clamp(max=t_end), dim=-1)
        mask = (arrival_times <= t_end).float()  
        marks = marks * mask 
        return Batch(inter_times=inter_times, checkins=pois, day_hour=None, mask=mask, marks=marks, gps=None, seq_idx=None)
