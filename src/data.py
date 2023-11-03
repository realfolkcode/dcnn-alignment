from typing import List

from .utils import seconds_to_frames

import pretty_midi
import torch
import torch.nn.functional as F
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
    normalized_perf_roll = perf_roll / (torch.norm(perf_roll, dim=0) + eps)
    normalized_score_roll = score_roll / (torch.norm(score_roll, dim=0) + eps)
    cross_similarity = torch.cdist(normalized_perf_roll, normalized_score_roll)
    cross_similarity = cross_similarity.unsqueeze(0)
    return cross_similarity


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
