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


def calculate_cross_similarity(score_roll: torch.Tensor, 
                               perf_roll: torch.Tensor,
                               eps: float = 1e-6) -> torch.Tensor:
    """Calculates cross-similarity matrix between score and performance.

    Args:
        score_roll: Score piano roll tensor of shape (score_frames, 128).
        perf_roll: Performance piano roll tensor of shape (perf_frames, 128).
        eps: Small value to avoid division by zero.

    Returns:
        A cross-similarity matrix of shape (score_frames, perf_frames).
    """
    normalized_score_roll = score_roll / (torch.norm(score_roll, dim=0) + eps)
    normalized_perf_roll = perf_roll / (torch.norm(perf_roll, dim=0) + eps)
    cross_similarity = torch.cdist(normalized_score_roll, normalized_perf_roll)
    return cross_similarity
