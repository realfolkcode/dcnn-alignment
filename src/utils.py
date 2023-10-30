import librosa
import pretty_midi


def plot_piano_roll(piano_roll, fs: int, start_pitch: int = 0, end_pitch: int = 128) -> None:
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
