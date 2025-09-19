from keras_yamnet import params
import csv
from pathlib import Path

import settings

def second_to_index(sec):
    """
    Convert seconds to index in variable output in when loading data from gt.
    """
    return int(sec // params.PATCH_HOP_SECONDS)

def write_csv(path, rows):
    path = Path(path)
    fieldnames = ["filename","start_time","end_time","class" ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", delimiter="\t")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def class_name_to_index(class_name):
    """
    Convert class name to index based on the class names defined in the YAMNet model.
    """
    if class_name in settings.CLASS_NAMES:
        return settings.CLASS_NAMES.tolist().index(class_name)
    else:
        return len(settings.CLASS_NAMES)-1 # Index of 'Other' class