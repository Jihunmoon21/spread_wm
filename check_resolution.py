#!/usr/bin/env python
"""
원본 HDF5 파일에서 이미지의 실제 해상도 확인
"""
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# HDF5 파일 경로
hdf5_dir = Path("/home/jihoonmun/LIBERO/libero/datasets/libero_object")
hdf5_files = list(hdf5_dir.rglob("*.hdf5"))

if not hdf5_files:
    print(f"⚠️ No HDF5 files found in {hdf5_dir}")
else:
    # 첫 번째 파일 확인
    hdf5_file = hdf5_files[0]
    print(f"📁 Checking: {hdf5_file}\n")
    
    with h5py.File(hdf5_file, 'r') as f:
        print("🔍 Top-level keys:", list(f.keys()))
        
        # 'data' 키 확인
        if 'data' in f:
            data = f['data']
            
            if isinstance(data, h5py.Group):
                # 첫 번째 demo 확인
                demo_keys = sorted(data.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
                first_demo_key = demo_keys[0]
                print(f"\n📷 Checking first demo: {first_demo_key}")
                
                demo_group = data[first_demo_key]
                print(f"📂 Demo keys: {list(demo_group.keys())}")
                
                # 'obs'가 Group인지 확인
                if 'obs' in demo_group:
                    obs_group = demo_group['obs']
                    print(f"\n📂 'obs' type: {type(obs_group)}")
                    
                    if isinstance(obs_group, h5py.Group):
                        print(f"📂 'obs' keys: {list(obs_group.keys())}")
                        
                        # 이미지 키 찾기
                        for key in obs_group.keys():
                            dataset = obs_group[key]
                            if isinstance(dataset, h5py.Dataset):
                                print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                                
                                # 카메라 이미지 찾기
                                if 'image' in key.lower() or 'rgb' in key.lower():
                                    if len(dataset) > 0:
                                        first_img = dataset[0]
                                        print(f"\n✅ Found {key}!")
                                        print(f"   Image shape: {first_img.shape}")
                                        print(f"   Actual resolution: {first_img.shape[1]}x{first_img.shape[0]} pixels")
                                        
                                        # 이미지 저장
                                        plt.imsave('original_hdf5_sample.png', first_img)
                                        print("   💾 Saved: original_hdf5_sample.png")
                                        break
    
    print(f"\n📊 Total HDF5 files: {len(hdf5_files)}")