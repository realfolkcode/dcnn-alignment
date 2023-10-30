import pretty_midi
import torch
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
    piano_roll = (pm.get_piano_roll(fs) > 0).astype('float').T
    piano_roll = torch.from_numpy(piano_roll)
    return piano_roll
