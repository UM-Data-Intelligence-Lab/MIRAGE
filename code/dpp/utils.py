import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import Any, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import os
import random
import collections
import logging


def _size_repr(key: str, item: Any) -> str:
    """String containing the size / shape of an object (e.g. a tensor, array)."""
    if isinstance(item, torch.Tensor) and item.dim() == 0:
        out = item.item()
    elif isinstance(item, torch.Tensor):
        out = str(list(item.size()))
    elif isinstance(item, list):
        out = f"[{len(item)}]"
    else:
        out = str(item)

    return f"{key}={out}"


class DotDict:
    """Dictionary where elements can be accessed as dict.entry."""

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        info = [_size_repr(key, item) for key, item in self]
        return f"{self.__class__.__name__}({', '.join(info)})"


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()


def pad_sequence(
        sequences: List[torch.Tensor],
        padding_value: float = 0,
        max_len: Optional[int] = None,
):
    r"""Pad a list of variable length Tensors with ``padding_value``"""
    dtype = sequences[0].dtype
    device = sequences[0].device
    seq_shape = sequences[0].shape
    trailing_dims = seq_shape[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = torch.empty(*out_dims, dtype=dtype, device=device).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor

    return out_tensor


def diff(x, dim: int = -1):
    """Inverse of x.cumsum(dim=dim).
    Compute differences between subsequent elements of the tensor.
    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension over which to compute the difference, {-2, -1}.
    Returns:
        diff: Tensor of the the same shape as x.
    """
    if dim == -1:
        return x - F.pad(x, (1, 0))[..., :-1]
    elif dim == -2:
        return x - F.pad(x, (0, 0, 1, 0))[..., :-1, :]
    else:
        raise ValueError("dim must be equal to -1 or -2")

def save_generated_seq(seqs, savepath, logger, experiment_comments, data_name):
    sequences=[]
    generated_pois=[]
    generated_marks=[]
    seq_counter=1
    for seq in seqs:
        arrival_times=seq.inter_times.cumsum(-1)[:-1].cpu().numpy().tolist()
        checkins=seq.checkins.cpu().numpy().astype(int).tolist()
        gps=seq.gps.cpu().numpy().tolist()
        day_hour=seq.day_hour.cpu().numpy().tolist()
        marks=seq.marks.cpu().numpy().tolist()
        generated_pois+=checkins
        generated_marks+=marks
        sequences.append({"arrival_times":arrival_times,"checkins":checkins,"gps":gps,"day_hour":day_hour,"seq_idx":seq_counter,"t_start":0,"t_end":7,"marks":marks})
        seq_counter+=1
    num_pois=len(set(generated_pois))
    num_marks=len(set(generated_marks))
    dataset={"sequences":sequences,"num_marks":num_marks,"num_seq_idxs":seq_counter,"num_pois":num_pois,"poi_gps":{}}
    path_to_file = os.path.join(savepath,data_name)
    name = f'{data_name}_generated.pkl' if experiment_comments is None else f'{data_name}_{experiment_comments}_generated.pkl'
    torch.save(dataset, os.path.join(path_to_file, name))
    logger.info(f"generated data saved")



def save_seqs(seqs,path,dataset_name,num_marks,locations,sequence_count,poi_gps_dict,logger):
    sequences=[]
    all_pois=[]
    all_marks=[]
    seq_idx=1
    for seq in seqs:
        arrival_times=seq.inter_times.cumsum(-1)[:-1].cpu().numpy().tolist()
        checkins=seq.checkins.cpu().numpy().tolist()
        gps=seq.gps.cpu().numpy().tolist()
        day_hour=seq.day_hour.cpu().numpy().tolist()
        marks=seq.marks.cpu().numpy().tolist()
        all_pois+=checkins
        all_marks+=marks
        sequences.append({"arrival_times":arrival_times,"checkins":checkins,"gps":gps,"day_hour":day_hour,"seq_idx":[seq.seq_idx.item()],"t_start":0,"t_end":7,"marks":marks})
        seq_idx+=1
    all_marks=len(set(all_marks))
    dataset={"sequences":sequences,"num_marks":num_marks,"num_seqs":seq_idx,"num_pois":locations,"poi_gps":poi_gps_dict}
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(dataset, os.path.join(path,dataset_name))


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger