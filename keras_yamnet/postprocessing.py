import numpy as np
from scipy.signal import medfilt
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import settings

def postprocess_output(predictions):

    binary_predictions = np.where(predictions >= settings.PREDICTION_THRESHOLD, 1, 0)

    #smoothed_preds = medfilt(binary_predictions, kernel_size=3)

    return binary_predictions
