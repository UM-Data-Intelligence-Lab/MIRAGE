import torch
import torch.nn as nn

import torch.distributions as D
from dpp.distributions import Normal, MixtureSameFamily, TransformedDistribution
from dpp.utils import clamp_preserve_gradients

from .MIRAGE import MIRAGE


class LogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.
    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time
    z = exp(y)

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_scales: Logarithms of scale parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
    """
    def __init__(
        self,
        locs: torch.Tensor,
        log_scales: torch.Tensor,
        log_weights: torch.Tensor,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=log_scales.exp())

        GMM = MixtureSameFamily(mixture_dist, component_dist)
        if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)]
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        transforms.append(D.ExpTransform())
        super().__init__(GMM, transforms)

    @property
    def mean(self) -> torch.Tensor:
        a = self.std_log_inter_time
        b = self.mean_log_inter_time
        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.logits
        return (log_weights + a * loc + b + 0.5 * a**2 * variance).logsumexp(-1).exp()


class MODEL(MIRAGE):
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
        tpp_num_mix_components: int = 16,
        vae_hidden: int = 32,
        vae_latent: int = 2,
        vae_mse_lambda: int = 500,
        poi_category: dict = {},
        poi_gps_dict: dict = {},
        tpp_sequential_type: str = "GRU",
    ):
        super().__init__(
            num_marks=num_marks,
            sequence_count=sequence_count,
            num_locations=num_locations,
            mean_log_inter_time=mean_log_inter_time,
            std_log_inter_time=std_log_inter_time,
            tpp_sequence_embedding_size=tpp_sequence_embedding_size,
            context_size=context_size,
            tpp_mark_embedding_size=tpp_mark_embedding_size,
            time_interval_embedding_size=time_interval_embedding_size,
            tpp_loc_embedding_size=tpp_loc_embedding_size,
            tpp_time_embedding_size=tpp_time_embedding_size,
            vae_hidden=vae_hidden,
            vae_latent=vae_latent,
            vae_mse_lambda=vae_mse_lambda,
            poi_category=poi_category,
            poi_gps_dict=poi_gps_dict,
            tpp_sequential_type=tpp_sequential_type,
        )
        self.num_mix_components = tpp_num_mix_components
        self.linear = nn.Linear(self.context_size + tpp_sequence_embedding_size + self.tpp_time_embedding_size, 3 * self.num_mix_components)

    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raw_params = self.linear(context)  

        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)

        return LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time
        )
