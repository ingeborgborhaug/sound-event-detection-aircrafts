import zipfile, h5py, io
p = r'D:\final_runs\aerosonic_to_norwegian_ex3\20260526-171718Z\radius_dual_pos_2km_neg_10km_wnone\aero_only_to_norwegian\fold_4_test_4\training\best_model.keras'
with zipfile.ZipFile(p,'r') as z:
    with z.open('model.weights.h5') as f:
        data = f.read()
        bio = io.BytesIO(data)
        with h5py.File(bio,'r') as hf:
            def walk(name, obj):
                print(name)
            hf.visititems(lambda name, obj: print(name))
