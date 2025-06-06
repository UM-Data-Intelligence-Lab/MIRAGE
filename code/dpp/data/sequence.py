import numpy as np
import torch

from typing import Optional

from dpp.utils import DotDict


class Sequence(DotDict):
    """
    A sequence of events with corresponding marks.

    IMPORTANT: last entry of inter_times must correspond to the survival time
    until the end of the observed interval. Because of this len(inter_times) == len(marks) + 1.

    Args:
        inter_times: Inter-event times. Last entry corresponds to the survival time
            until the end of the observed interval, shape (seq_len,)
        marks: Mark corresponding to each event. Note that the length is 1 shorter than
            for inter_times, shape (seq_len - 1,)
    """
    def __init__(self, inter_times: torch.Tensor, checkins: torch.Tensor, gps: torch.Tensor, day_hour: torch.Tensor, seq_idx: torch.Tensor, marks: Optional[torch.Tensor] = None, revisit: Optional[torch.Tensor] = None, **kwargs):
        if not isinstance(inter_times, torch.Tensor):
            inter_times = torch.tensor(inter_times)
        # The inter-event times should be at least 1e-10 to avoid numerical issues
        self.inter_times = inter_times.float().clamp(min=1e-10)

        #new added info for spatial module
        if not isinstance(checkins, torch.Tensor):
            checkins = torch.tensor(checkins)
        if not isinstance(gps, torch.Tensor):
            if gps is not None:
                self.gps = torch.tensor(gps)
            else:
                self.gps=None
        if not isinstance(day_hour, torch.Tensor):
            day_hour = torch.tensor(day_hour)
        if not isinstance(seq_idx, torch.Tensor):
            if seq_idx is not None:
                self.seq_idx = torch.tensor(seq_idx)
            else:
                self.seq_idx=None

        if revisit is not None:
            if not isinstance(revisit, torch.Tensor):
                revisit = torch.tensor(revisit)
            self.revisit = revisit.long()
        else:
            self.revisit = None

        self.checkins = checkins
        self.day_hour = day_hour

        if marks is not None:
            if not isinstance(marks, torch.Tensor):
                marks = torch.tensor(marks)
            self.marks = marks.long()
        else:
            self.marks = None

        for key, value in kwargs.items():
            self[key] = value

        self._validate_args()

    def __len__(self):
        return len(self.inter_times)

    def _validate_args(self):
        """Check if all tensors have correct shapes."""
        if self.inter_times.ndim != 1:
            raise ValueError(
                f"inter_times must be a 1-d tensor (got {self.inter_times.ndim}-d)"
            )
        if self.marks is not None:
            expected_marks_length = len(self.inter_times) - 1
            if self.marks.shape != (expected_marks_length,):
                raise ValueError(
                    f"marks must be of shape (seq_len - 1 = {expected_marks_length},)"
                    f"(got {self.marks.shape})"
                )

    def to(self, device: str):
        """Move the underlying data to the specified device."""
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)

