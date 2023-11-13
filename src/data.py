from typing import List, Dict, Optional, Callable
import os

from .utils import seconds_to_frames

import pretty_midi
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


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


def make_dataset(data_dir: str,
                 pairs: List[Dict],
                 fs: int) -> None:
    """Calculates and saves cross-similarity matrices and beat alignments.

    Args:
        data_dir: The directory to store matrices.
        pairs: A list of pairs of performance and score and their alignment.
        fs: Piano roll sampling frequency.
    """
    os.makedirs(data_dir, exist_ok=True)

    for idx in tqdm(range(len(pairs))):
        perf_path = pairs[idx]['perf']
        score_path = pairs[idx]['score']

        perf_roll = extract_piano_roll(perf_path, fs=fs)
        score_roll = extract_piano_roll(score_path, fs=fs)
        cross_similarity = calculate_cross_similarity(perf_roll, score_roll)
        sample = {'image': cross_similarity}

        perf_beats = np.array(pairs[idx]['perf_beats'])
        score_beats = np.array(pairs[idx]['score_beats'])
        beat_alignment = construct_beat_alignment(perf_beats, score_beats, fs)
        sample['alignment'] = beat_alignment

        filepath = os.path.join(data_dir, f"sample_{idx}.pt")
        torch.save(sample, filepath)


class CrossSimilarityDataset(Dataset):
    """Performance-score cross-similarity matrices dataset."""

    def __init__(self,
                 data_dir: str,
                 transform: Callable,
                 structural_transform: Optional[Callable] = None,
                 inference_only: bool = False):
        """Initializes an instance of dataset class.

        Args:
            data_dir: The directory where cross-similarities and alignments are stored.
            transform: Transformation to apply to cross-similarity matrices (e.g., resize).
            structural_transform: Structural transformations to apply to performance
              piano rolls.
            inference_only: If True, dataset does not contain ground truth inflection points.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.structural_transform = structural_transform
        self.inference_only = inference_only
        self.data_paths = self._get_paths(data_dir)

    def _get_paths(self, data_dir: str) -> List[str]:
        """Retrieves all the .pt files in data directory.

        Args:
            data_dir: The directory where cross-similarities and alignments are stored.

        Returns:
            A list of paths to .pt files.
        """
        data_paths = os.listdir(data_dir)
        data_paths = list(filter(lambda s: s.endswith('.pt'), data_paths))
        data_paths = [os.path.join(self.data_dir, s) for s in data_paths]
        return data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        sample = torch.load(self.data_paths[idx])
        x = sample['image']
        beat_alignment = sample['alignment']

        if self.structural_transform is not None:
            x, beat_alignment, inflection_points = self.structural_transform(x, beat_alignment)
        
        x = self.transform(x)
        new_sample = {'image': x}

        if not self.inference_only:
            new_sample['target'] = inflection_points
        
        return new_sample
