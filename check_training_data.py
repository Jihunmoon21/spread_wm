#!/usr/bin/env python
"""
학습 데이터 로더에서 실제로 어떤 이미지가 로드되는지 확인하는 스크립트
"""
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datasets.libero_dset import load_libero_slice_train_val
from datasets.img_transforms import default_transform

print("=" * 60)
print("학습 데이터 로더 확인")
print("=" * 60)

# Dataset 생성 (학습 시와 동일한 방식)
print("\n1. 데이터셋 로드 중...")
data_path = "/home/jihoonmun/LIBERO/libero/datasets_processed/libero_object"
transform = default_transform(img_size=256)

train_dset, val_dset = load_libero_slice_train_val(
    transform=transform,
    data_path=data_path,
    train_fraction=0.9,
    random_seed=42,
    num_frames=10,  # num_hist + num_pred
    frameskip=1,
    main_view='third_image'
)

print(f"  Train dataset type: {type(train_dset)}")
print(f"  Val dataset type: {type(val_dset)}")

# dict로 반환되므로 'train' 키로 접근
if isinstance(train_dset, dict):
    actual_train_dset = train_dset  # 이미 dict
    print(f"  Dataset keys: {train_dset.keys()}")
    # DataLoader를 만들어야 하지만, 간단히 하나의 샘플만 확인
    # 대신 LiberoDataset을 직접 생성
    from datasets.libero_dset import LiberoDataset
    direct_dset = LiberoDataset(
        data_path="/home/jihoonmun/LIBERO/libero/datasets_processed/libero_object",
        transform=transform,
        main_view='third_image',
        normalize=True,
        num_frames=10,
        frameskip=1
    )
    train_dset = direct_dset

print(f"  Actual dataset size: {len(train_dset)}")

# 첫 번째 샘플 가져오기
print("\n2. 첫 번째 배치 로드 중...")
sample = train_dset[0]
if sample is None:
    print("  ERROR: Sample is None!")
    sys.exit(1)

obs, actions, state, meta = sample

print(f"  Visual shape: {obs['visual'].shape}")  # (T, C, H, W)
print(f"  Proprio shape: {obs['proprio'].shape}")
print(f"  Actions shape: {actions.shape}")

# 첫 번째 프레임 추출 및 저장
print("\n3. 이미지 저장 중...")
first_frame = obs['visual'][0]  # (C, H, W)

# Denormalize (학습 시 Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 적용됨)
first_frame = first_frame * 0.5 + 0.5  # [-1, 1] -> [0, 1]
first_frame = torch.clamp(first_frame, 0, 1)

# (C, H, W) -> (H, W, C) for matplotlib
first_frame_np = first_frame.permute(1, 2, 0).cpu().numpy()

# 저장
plt.imsave('training_data_frame0.png', first_frame_np)
plt.imsave('training_data_frame0_flipped.png', np.flipud(first_frame_np))

print("  ✅ Saved:")
print("     - training_data_frame0.png (학습 데이터 원본)")
print("     - training_data_frame0_flipped.png (flip 버전)")
print("\n" + "=" * 60)
print("👀 두 이미지를 확인하세요:")
print("   - 학습 데이터가 정상 방향이면: training_data_frame0.png가 올바른 방향")
print("   - 학습 데이터가 flip된 방향이면: training_data_frame0_flipped.png가 올바른 방향")
print("=" * 60)

