# check_image_orientation.py
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# HDF5에서 샘플 이미지 로드
hdf5_files = glob.glob("/home/jihoonmun/LIBERO/libero/datasets_processed/libero_object/**/*.hdf5", recursive=True)
hdf5_path = hdf5_files[0] if hdf5_files else None

if not hdf5_path:
    print("No HDF5 files found!")
    exit(1)

print(f"Using: {hdf5_path}")

with h5py.File(hdf5_path, 'r') as f:
    print(f"Keys in HDF5: {list(f.keys())}")
    if 'observation' in f:
        print(f"Keys in observation: {list(f['observation'].keys())}")
        
        # plan.py에서 사용하는 main_view는 'third_image' 또는 'agentview_image'
        view_key = None
        for key in f['observation'].keys():
            if 'image' in key.lower():
                view_key = key
                break
        
        if not view_key:
            print("No image key found in observation!")
            exit(1)
        
        print(f"Using view key: {view_key}")
        img_data = f['observation'][view_key][0]  # 첫 프레임
        
        # 디코딩
        if isinstance(img_data, np.ndarray) and img_data.dtype == np.uint8:
            buf = img_data.reshape(-1)
        elif isinstance(img_data, (bytes, bytearray)):
            buf = np.frombuffer(img_data, dtype=np.uint8)
        elif isinstance(img_data, np.void):
            buf = np.frombuffer(img_data.tobytes(), dtype=np.uint8)
        else:
            raise TypeError(f"Unsupported image type: {type(img_data)}")
        
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 저장
        plt.imsave('hdf5_image_original.png', img_rgb)
        plt.imsave('hdf5_image_flipped.png', np.flipud(img_rgb))
        print("\n✅ Saved: hdf5_image_original.png and hdf5_image_flipped.png")
        print("👀 Check which one looks correct (robot should be upright)!")