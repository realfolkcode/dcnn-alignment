from typing import Dict
import json
import os


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
