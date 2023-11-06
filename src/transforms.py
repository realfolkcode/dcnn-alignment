from typing import List, Tuple
import numpy as np
import torch
from torch import nn


def sample_jumps(beat_alignment: np.ndarray,
                 max_num_jumps: int = 2) -> List[Tuple[int, int]]:
    """Samples segments with jumps.

    Args:
        beat_alignment: Beatwise alignment array in frames of shape 
          (2, num_beats), where the first and second rows correspond 
          to performance and score, respectively.
        max_num_jumps: The maximum number of jumps.

    Returns:
        A list of pairs of segment timestamps in frames.
    """
    perf_beats = beat_alignment[0]

    timestamps = []

    indices = np.arange(len(perf_beats))
    start_idx = 0
    for _ in range(max_num_jumps):
        end_idx = torch.randint(start_idx, indices[-1], size=(1,)).item()
        timestamps.append((perf_beats[start_idx], perf_beats[end_idx]))
        start_idx = torch.randint(0, end_idx, size=(1,)).item()
    timestamps.append((perf_beats[start_idx], perf_beats[-1]))

    return timestamps


def augment_performance(perf_roll: torch.Tensor,
                        beat_alignment: np.ndarray,
                        segment_timestamps: List[Tuple[int, int]],
                        max_silence: int = 200) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """Augments a performance with jumps given the timestamps.

    Args:
        perf_roll: Performance piano roll tensor of shape (perf_frames, 128).
        beat_alignment: Beatwise alignment array in frames of shape 
          (2, num_beats), where the first and second rows correspond 
          to performance and score, respectively.
        segment_timestamps: A list of pairs of segment timestamps in frames.
        max_silence: The maximal duration of silence before jumps in frames.

    Returns:
        Augmented performance piano roll with jumps; the new beat alignment;
          and the inflection points of shape (2 * k - 2, 2), where `k` is the
          number of segments.
    """
    new_perf_roll = torch.zeros((0, 128))
    new_beat_alignment = np.zeros((2, 0)).astype('int')
    perf_beats = beat_alignment[0]

    inflection_points = torch.zeros((2 * len(segment_timestamps) - 2, 2))

    offset = 0
    for i, ts in enumerate(segment_timestamps):
        # Construct new piano roll from segments
        start_idx, end_idx = ts
        segment = perf_roll[start_idx:end_idx]
        new_perf_roll = torch.cat((new_perf_roll, segment), dim=0)
        # Add silence
        num_silence = torch.randint(0, max_silence, size=(1,)).item()
        new_perf_roll = torch.cat((new_perf_roll, torch.zeros((num_silence, 128))), dim=0)

        # Retrieve beat indices for alignment
        start_idx =  np.argwhere(perf_beats == start_idx).item()
        end_idx = np.argwhere(perf_beats == end_idx).item()
        # Construct new beat alignment from segment alignments
        segment_alignment = beat_alignment[:, start_idx:end_idx].copy()
        segment_alignment[0] -= segment_alignment[0, 0]
        segment_alignment[0] += offset
        new_beat_alignment = np.concatenate((new_beat_alignment, segment_alignment), axis=1)
        offset = new_beat_alignment[0, -1] + beat_alignment[0, end_idx] - beat_alignment[0, end_idx-1] + num_silence

        # Add inflection points
        if i > 0:
            inflection_points[i * 2 - 1] = torch.from_numpy(segment_alignment[:, 0])
        if i < len(segment_timestamps) - 1:
            inflection_points[i * 2] = torch.from_numpy(segment_alignment[:, -1])

    return new_perf_roll, new_beat_alignment, inflection_points


class RandomJumps(nn.Module):
    """Piano roll augmentation with structural repeats."""

    def __init__(self,
                 fs: int,
                 max_num_jumps: int = 2,
                 max_silence_s: float = 0):
        """Initializes an instance of RandomJumps transformation.

        Args:
            fs: Sampling frequency.
            max_num_jumps: The maximum number of jumps.
            max_silence_s: The maximal duration of silence before jumps 
              in seconds.
        """
        super().__init__()
        self.fs = fs
        self.max_num_jumps = max_num_jumps
        self.max_silence = int(max_silence_s * fs)

    def forward(self,
                perf_roll: torch.Tensor,
                beat_alignment: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        """Augments a performance with jumps.

        Args:
            perf_roll: Performance piano roll tensor of shape (perf_frames, 128).
            beat_alignment: Beatwise alignment array in frames of shape 
              (2, num_beats), where the first and second rows correspond 
              to performance and score, respectively.
        Returns:
            Augmented performance piano roll with jumps; the new beat alignment;
              and the inflection points of shape (2 * k - 2, 2), where `k` is the
              number of segments.
        """
        segment_timestamps = sample_jumps(beat_alignment, max_num_jumps=self.max_num_jumps)
        aug_perf_roll, aug_beat_alignment, inflection_points = augment_performance(perf_roll,
                                                                                   beat_alignment,
                                                                                   segment_timestamps,
                                                                                   max_silence=self.max_silence)
        return aug_perf_roll, aug_beat_alignment, inflection_points
