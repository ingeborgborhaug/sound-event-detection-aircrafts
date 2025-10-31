from keras_yamnet import params
import csv
from pathlib import Path

import settings

def second_to_index(sec):
    """
    Convert seconds to index in variable output in when loading data from gt.
    """
    return int(sec // params.PATCH_HOP_SECONDS)


def class_name_to_index(class_name):
    """
    Convert class name to index based on the class names defined in the YAMNet model.
    """
    if class_name in settings.CLASS_NAMES:
        return settings.CLASS_NAMES.tolist().index(class_name)
    else:
        raise ValueError(f"Class name '{class_name}' not found in CLASS_NAMES.")