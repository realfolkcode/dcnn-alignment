from typing import Optional

import librosa
import pretty_midi
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_piano_roll(piano_roll: torch.Tensor, fs: int, start_pitch: int = 0, end_pitch: int = 128) -> None:
    """Plots a piano roll representation.

    Args:
        piano_roll: A binary piano roll tensor of shape (num_frames, 128).
        fs: Sampling frequency.
        start_pitch: Minimal pitch to display (MIDI numeration).
        end_pitch: Maximal pitch to display (MIDI numeration)
    """
    librosa.display.specshow(piano_roll.T.numpy()[start_pitch:end_pitch + 1],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))


def plot_cross_similarity(cross_similarity: torch.Tensor,
                          beat_alignment: Optional[np.ndarray] = None,
                          inflection_points: Optional[np.ndarray] = None) -> None:
    """Plots a cross-similarity matrix.

    If the beat alignment is provided, additionally plots alignment path.

    Args:
        cross_similarity: A cross-similarity matrix of shape 
          (1, perf_frames, score_frames).
        beat_alignment: Beatwise alignment array in frames of shape
          (2, num_beats), where the first and second rows correspond to 
            performance and score, respectively.
        inflection_points: The inflection points of shape (n, 2).
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cross_similarity[0])
    
    if beat_alignment is not None:
        plt.plot(beat_alignment[1, :], beat_alignment[0, :], label='Alignment', color='white')
        plt.legend()
    
    if inflection_points is not None:
        plt.scatter(inflection_points[:, 1], inflection_points[:, 0], 
                    label='Inflection points', color='red', marker='x')
        plt.legend()

    plt.title('Cross-similarity matrix')
    plt.xlabel('Score frames')
    plt.ylabel('Performance frames')
    plt.colorbar()


def seconds_to_frames(t: np.ndarray, fs: int) -> np.ndarray:
    """Converts time in seconds to frames.

    Args:
        t: Time array in seconds.
        fs: Sampling frequency.
    
    Returns:
        Time array in frames.
    """
    return (t * fs).astype('int')
