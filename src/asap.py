from typing import Dict, Tuple, List
import json
import os
from sklearn.model_selection import train_test_split


def load_annotations(asap_dir: str) -> Dict:
    """Loads the annotations from ASAP dataset.

    Args:
        asap_dir: The directory where ASAP is stored.
    
    Returns:
        A dictionary containing the ASAP annotations.
    """
    annotations_path = os.path.join(asap_dir, 'asap_annotations.json')
    with open(annotations_path, "r", encoding="utf8") as f:
        annotations = json.load(f)
    return annotations


class ASAPWrapper:
    """A wrapper for ASAP dataset.

    Attributes:
        annotations: A dictionary containing the ASAP annotations.
        train_paths: A list of train paths and performance-score beat alignments.
        val_paths: A list of val paths and performance-score beat alignments.
    """

    def __init__(self,
                 asap_dir: str,
                 val_ratio: float = 0.2,
                 random_seed: int = 42):
        """Initializes an object of ASAPWrapper.

        Args:
            asap_dir: The directory where ASAP is stored.
            val_ratio: The ratio of the validation set.
            random_seed: Seed for deterministic split.
        """
        self.asap_dir = asap_dir
        self.annotations = load_annotations(asap_dir)

        piece_lst = self.get_piece_list()
        self.train_paths, self.val_paths = self.split_into_train_val(piece_lst,
                                                                     val_ratio,
                                                                     random_seed)

    def get_piece_list(self) -> List[str]:
        """Retrieves a list of distinct pieces from ASAP.

        Returns:
            Relative ASAP paths to distinct piece directories.
        """
        piece_lst = set()

        for perf_title in self.annotations.keys():
            piece_title = '/'.join(perf_title.split('/')[:-1])
            piece_lst.add(piece_title)

        piece_lst = list(piece_lst)
        return piece_lst

    def split_into_train_val(self,
                             piece_lst: List[str],
                             val_ratio: float = 0.2,
                             random_seed: int = 42) -> Tuple[List, List]:
        """Splits ASAP into train and val sets.

        Args:
            piece_lst: Relative ASAP paths to distinct piece directories.
            val_ratio: The ratio of the validation set.
            random_seed: Seed for deterministic split.

        Returns:
            A list of train paths and a list of val paths with their beat alignments.
        """
        train_piece_lst, val_piece_lst = train_test_split(piece_lst,
                                                          test_size=val_ratio,
                                                          random_state=random_seed)
        train_paths = []
        val_paths = []

        for perf_title in self.annotations.keys():
            piece_title = '/'.join(perf_title.split('/')[:-1])

            perf_score_pair = {}
            perf_score_pair['perf'] = os.path.join(self.asap_dir, perf_title)
            perf_score_pair['score'] = os.path.join(self.asap_dir, piece_title, 'midi_score.mid')
            perf_score_pair['perf_beats'] = self.annotations[perf_title]['performance_beats']
            perf_score_pair['score_beats'] = self.annotations[perf_title]['midi_score_beats']

            if self.annotations[perf_title]['score_and_performance_aligned']:
                assert len(perf_score_pair['perf_beats']) == len(perf_score_pair['score_beats'])
                if piece_title in train_piece_lst:
                    train_paths.append(perf_score_pair)
                elif piece_title in val_piece_lst:
                    val_paths.append(perf_score_pair)
        
        return train_paths, val_paths
