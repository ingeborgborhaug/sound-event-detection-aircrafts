import os, sys, traceback
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.models.classifier_heads import CustomClassificationHead
import tensorflow as tf
p = r'D:\final_runs\aerosonic_to_norwegian_ex3\20260526-171718Z\radius_dual_pos_2km_neg_10km_wnone\aero_only_to_norwegian\fold_4_test_4\training\best_model.keras'
print('Attempting load with custom_objects mapping...')
try:
    m = tf.keras.models.load_model(p, compile=False, custom_objects={'CustomClassificationHead': CustomClassificationHead})
    print('Loaded model:', m)
except Exception as e:
    print('ERROR during load:')
    traceback.print_exc()
