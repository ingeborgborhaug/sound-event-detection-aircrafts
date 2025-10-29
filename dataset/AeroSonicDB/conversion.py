import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import csv
from pathlib import Path

parent_of_cwd = os.path.dirname(os.getcwd())

gt_dir_loc12 = os.path.join(parent_of_cwd, "AeroSonicDB-YPAD0523", "data", "raw", "sample_meta.csv")
gt_file_loc12 = pd.read_csv(gt_dir_loc12)
gt_dir_loc0 = os.path.join(parent_of_cwd, "AeroSonicDB-YPAD0523", "data", "raw", "environment_class_mappings.csv")

gt_converted_path = "dataset/AeroSonicDB"

gt_rows_train = []
gt_rows_test = []

def write_csv(path, rows):
    path = Path(path)
    fieldnames = ["filename","start_time","end_time","class" ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", delimiter="\t")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def class_name_converstion(idx_name):
        if idx_name == 0:
            return "No aircraft"
        elif idx_name == 1:
            return "Aircraft"


for _, row in tqdm(gt_file_loc12.iterrows(), total=len(gt_file_loc12), desc='Loading gt'):
        original_label = row['class']
        offset = row['offset']
        onset = offset + row['duration']

        #class_label = class_name_converstion(original_label)
        class_label = original_label
            
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
            
              
write_csv(os.path.join(gt_converted_path, "gt_train.csv"), gt_rows_train)
write_csv(os.path.join(gt_converted_path, "gt_test.csv"), gt_rows_test)

def process_environment_mappings(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file, header=None)
    
    # Initialize lists to store results
    results = []
    segment_length = 5 # seconds
    
    # Process each column (0-5)
    for col in range(6):
        start_time_1 = None
        start_time_0 = None
        
        for idx, value in enumerate(df[col][1:], start=1):

            if value != 'ignore':
                value = int(value) # 1 or 0
            
            # Start of new aircraft segment
            if value == 1 and start_time_1 is None:
                # print(f"Aircraft detected in channel {col+1} at segment {idx}, time {idx * segment_length}s")
                start_time_1 = (idx - 1) * segment_length


            # End of aircraft segment
            elif (value == 0 or value == 'ignore') and start_time_1 is not None:
                results.append({
                    'filename': f'{col+1}_AUDIO.wav',
                    'starttime': start_time_1,
                    'endtime': (idx - 1) * segment_length,
                    'class': 1
                })
                start_time_1 = None

            # Start of new no-aircraft segment
            if value == 0 and start_time_0 is None:
                start_time_0 = (idx - 1) * segment_length

            # End of no-aircraft segment
            elif (value == 1 or value == 'ignore') and start_time_0 is not None:
                results.append({
                    'filename': f'{col+1}_AUDIO.wav',
                    'starttime': start_time_0,
                    'endtime': (idx - 1) * segment_length,
                    'class': 0
                })
                start_time_0 = None
        
        # Handle case where segment extends to end of file
        if start_time_1 is not None:
            results.append({
                'filename': f'{col+1}_AUDIO.wav',
                'starttime': start_time_1,
                'endtime': (len(df) - 1) * segment_length,
                'class': 1
            })

        if start_time_0 is not None:
            results.append({
                'filename': f'{col+1}_AUDIO.wav',
                'starttime': start_time_0,
                'endtime': (len(df) - 1) * segment_length,
                'class': 0
            })
    
    # Convert results to DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(gt_converted_path,'env_audio_gt.csv'), sep='\t', index=False)

# Run the conversion
process_environment_mappings(gt_dir_loc0)

print(f"Converted gt files written to {gt_converted_path}")

            
        