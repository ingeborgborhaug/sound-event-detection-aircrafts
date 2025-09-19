import pandas as pd
import os
from tqdm import tqdm
from .. import conversion_functions as cf

parent_of_cwd = os.path.dirname(os.getcwd())
gt_dir = os.path.join(parent_of_cwd, "AeroSonicDB-YPAD0523", "data", "raw", "sample_meta.csv")
gt_file = pd.read_csv(gt_dir)
gt_converted_path = os.path.join(parent_of_cwd, "sound-event-detection-aircrafts", "Dataset", "AeroSonicDB")

gt_rows_train = []
gt_rows_test = []

def class_name_converstion(idx_name):
        if idx_name == 0:
            return None
        elif idx_name == 1:
            return "Aircraft"


for _, row in tqdm(gt_file.iterrows(), total=len(gt_file), desc='Loading gt'):
        original_label = row['class']
        offset = row['offset']
        onset = offset + row['duration']

        if class_name_converstion(original_label) is None:
            continue
        else:
            class_label = class_name_converstion(original_label)

        if row['train-test'] == 'train':
            gt_rows_train.append({
                    "filename": row['filename'],
                    "start_time": offset,
                    "end_time": onset,
                    "class": class_label,
                    "fold": row['fold']
            })

        elif row['train-test'] == 'test':
            gt_rows_test.append({
                "filename": row['filename'],
                "start_time": offset,
                "end_time": onset,
                "class": class_label,
                "fold": row['fold']
            })
            
              
cf.write_csv(os.path.join(gt_converted_path, "gt_train.csv"), gt_rows_train)
cf.write_csv(os.path.join(gt_converted_path, "gt_test.csv"), gt_rows_test)

print(f"Converted gt files written to {gt_converted_path}")

            
        