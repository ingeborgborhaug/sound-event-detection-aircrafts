import pandas as pd
import os
from tqdm import tqdm
from .. import conversion_functions as cf
import numpy as np

parent_of_cwd = os.path.dirname(os.getcwd())
gt_converted_path = os.path.join(parent_of_cwd, "sound-event-detection-aircrafts", "Dataset", "AeroSonicDB")

gt_dir_loc12 = os.path.join(parent_of_cwd, "AeroSonicDB-YPAD0523", "data", "raw", "sample_meta.csv")
gt_file_loc12 = pd.read_csv(gt_dir_loc12)

gt_dir_loc0 = os.path.join(parent_of_cwd, "AeroSonicDB-YPAD0523", "data", "raw", "environment_class_mappings.csv")

gt_rows_train = []
gt_rows_test = []

def class_name_converstion(idx_name):
        if idx_name == 0:
            return "N"
        elif idx_name == 1:
            return "Aircraft"


for _, row in tqdm(gt_file_loc12.iterrows(), total=len(gt_file_loc12), desc='Loading gt'):
        original_label = row['class']
        offset = row['offset']
        onset = offset + row['duration']

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

def process_environment_mappings(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file, header=None)
    
    # Initialize lists to store results
    results = []
    
    # Process each column (0-5)
    for col in range(6):
        current_segment = []
        start_time = None
        
        # Iterate through rows
        for idx, value in enumerate(df[col]):
            # Skip 'ignore' values
            if value == 'ignore':
                if start_time is not None:
                    # End current segment if we were tracking one
                    results.append({
                        'filename': f'channel_{col}',
                        'starttime': start_time,
                        'endtime': idx - 1,
                        'class': 1
                    })
                    start_time = None
                continue
                
            value = int(value)
            
            # Start of new aircraft segment
            if value == 1 and start_time is None:
                start_time = idx
            # End of aircraft segment
            elif value == 0 and start_time is not None:
                results.append({
                    'filename': f'channel_{col}',
                    'starttime': start_time,
                    'endtime': idx - 1,
                    'class': 1
                })
                start_time = None
        
        # Handle case where segment extends to end of file
        if start_time is not None:
            results.append({
                'filename': f'channel_{col}',
                'starttime': start_time,
                'endtime': len(df) - 1,
                'class': 1
            })
    
    # Convert results to DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_csv('processed_aircraft_segments.csv', index=False)

# Run the conversion
process_environment_mappings(gt_dir_loc0)

            
        