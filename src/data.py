from typing import List, Dict, Optional, Callable

from .utils import seconds_to_frames

import pretty_midi
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


def extract_piano_roll(midi_path: str, fs: int = 40) -> torch.Tensor:
    """Extracts piano roll representation from midi file.

    Args:
        midi_path: The path to a MIDI file.
        fs: Sampling frequency.

    Returns:
        A binary piano roll tensor of shape (num_frames, 128).
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = (pm.get_piano_roll(fs) > 0).astype('float32').T
    piano_roll = torch.from_numpy(piano_roll)
    return piano_roll


def calculate_cross_similarity(perf_roll: torch.Tensor,
                               score_roll: torch.Tensor, 
                               eps: float = 1e-6) -> torch.Tensor:
    """Calculates cross-similarity matrix between score and performance.

    Args:
        perf_roll: Performance piano roll tensor of shape (perf_frames, 128).
        score_roll: Score piano roll tensor of shape (score_frames, 128).
        eps: Small value to avoid division by zero.

    Returns:
        A cross-similarity matrix of shape (1, perf_frames, score_frames).
    """
    cross_sim = torch.sparse.mm(perf_roll.to_sparse(), score_roll.T)
    cross_sim /= (torch.norm(perf_roll, dim=1).unsqueeze(1) + eps)
    cross_sim /= (torch.norm(score_roll, dim=1).unsqueeze(1).T + eps)
    cross_sim = cross_sim.unsqueeze(0)
    return cross_sim


def construct_beat_alignment(perf_beats: np.ndarray, 
                             score_beats: np.ndarray,
                             fs: int) -> np.ndarray:
    """Constructs an aligned array of beats.

    Args:
        perf_beats: Beat timestamps in performance (in seconds).
        score_beats: Beat timestamps in score (in seconds).
        fs: Sampling frequency.
    
    Returns:
        Beatwise alignment array in frames of shape (2, num_beats), where 
          the first and second rows correspond to performance and score, 
          respectively.
    """
    assert len(perf_beats) == len(score_beats)

    perf_beats_f = seconds_to_frames(perf_beats, fs)
    score_beats_f = seconds_to_frames(score_beats, fs)

    beat_alignment = np.vstack((perf_beats_f, score_beats_f))
    return beat_alignment


class CrossSimilarityDataset(Dataset):
    """Performance-score cross-similarity matrices dataset."""

    def __init__(self,
                 pairs: List[Dict],
                 fs: int,
                 transform: Callable,
                 structural_transform: Optional[Callable] = None,
                 inference_only: bool = False):
        """Initializes an instance of dataset class.

        Args:
            pairs: A list of performance-score path pairs and (optionally) their aligned
              beat arrays. Each dictionary in a list must contain keys `perf`, `score`.
              If `structural_transform` is not None, then `perf_beats`, and `score_beats`
              must also be keys.
            fs: Piano roll sampling frequency.
            transform: Transformation to apply to cross-similarity matrices (e.g., resize).
            structural_transform: Structural transformations to apply to performance
              piano rolls.
            inference_only: If True, dataset does not contain ground truth inflection points.
        """
        self.pairs = pairs
        self.fs = fs
        self.transform = transform
        self.structural_transform = structural_transform
        self.inference_only = inference_only

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        perf_path = self.pairs[idx]['perf']
        score_path = self.pairs[idx]['score']
        if self.structural_transform is not None:
            perf_beats = np.array(self.pairs[idx]['perf_beats'])
            score_beats = np.array(self.pairs[idx]['score_beats'])

        perf_roll = extract_piano_roll(perf_path, fs=self.fs)
        score_roll = extract_piano_roll(score_path, fs=self.fs)

        if self.structural_transform is not None:
            beat_alignment = construct_beat_alignment(perf_beats, score_beats, self.fs)
            perf_roll, beat_alignment, inflection_points = self.structural_transform(perf_roll, score_roll, beat_alignment)

        cross_similarity = calculate_cross_similarity(perf_roll, score_roll)
        cross_similarity = self.transform(cross_similarity)
        sample = {'image': cross_similarity}

        if not self.inference_only:
            sample['target'] = inflection_points

        return sample
