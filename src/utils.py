import librosa
import pretty_midi
import torch
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


def plot_cross_similarity(cross_similarity: torch.Tensor) -> None:
    """Plots a cross-similarity matrix.

    Args:
        cross_similarity: A cross-similarity matrix of shape 
          (score_frames, perf_frames).
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cross_similarity)
    plt.title('Cross-similarity matrix')
    plt.xlabel('Performance frames')
    plt.ylabel('Score frames')
    plt.colorbar()
