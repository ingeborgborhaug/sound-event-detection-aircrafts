import zipfile, json, sys
p = r'D:\final_runs\aerosonic_to_norwegian_ex3\20260526-171718Z\radius_dual_pos_2km_neg_10km_wnone\aero_only_to_norwegian\fold_4_test_4\training\best_model.keras'
try:
    with zipfile.ZipFile(p,'r') as z:
        print('ZIPLIST:')
        for n in z.namelist():
            print(n)
        for candidate in ['metadata.json','config.json','model_config','model_config.json','keras_metadata.pb','saved_model.pb','config/model_config.json']:
            if candidate in z.namelist():
                print('\n--- CONTENT of', candidate, '---')
                data = z.read(candidate)
                try:
                    txt = data.decode('utf-8')
                    print(txt[:20000])
                    # write full content to workspace for inspection
                    outp = 'tools/extracted_'+candidate.replace('/','_')
                    with open(outp,'w',encoding='utf-8') as f:
                        f.write(txt)
                    print('WROTE', outp)
                except Exception as e:
                    print('Could not decode', e)
                # continue to try other candidates
        # If HDF5 weights present, report that too
        for n in z.namelist():
            if n.endswith('.h5') or n.endswith('.hdf5'):
                print('\nFound HDF5 weights file:', n)
except Exception as e:
    print('ERROR:', e)
