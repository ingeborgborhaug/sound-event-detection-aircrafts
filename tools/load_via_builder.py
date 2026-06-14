import os, sys, zipfile, tempfile, traceback
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.models.yamnet_finetune import build_yamnet_classifier
import tensorflow as tf
p = r'D:\final_runs\aerosonic_to_norwegian_ex3\20260526-171718Z\radius_dual_pos_2km_neg_10km_wnone\aero_only_to_norwegian\fold_4_test_4\training\best_model.keras'
print('Building model via builder...')
model = build_yamnet_classifier(freeze_backbone=False)
print('Model built. Attempting to extract weights file from archive...')
with zipfile.ZipFile(p,'r') as z:
    if 'model.weights.h5' in z.namelist():
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        tmp.write(z.read('model.weights.h5'))
        tmp.flush(); tmp.close()
        wfile = tmp.name
        print('Extracted weights to', wfile)
        try:
            model.load_weights(wfile, by_name=True)
            print('Loaded weights by_name successfully')
        except Exception:
            print('Error while loading weights:')
            traceback.print_exc()
    else:
        print('No HDF5 weights inside archive')
