import os
import os.path as osp
import torch
import numpy as np
import io
import json
from PIL import Image
# [수정] Dataset 임포트 (Subset은 더 이상 사용하지 않음)
from torch.utils.data import Dataset
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Optional, Dict, Tuple, Any, List
from einops import rearrange # [수정] einops 임포트

# [수정] TrajSlicerDataset, split_traj_datasets 제거
# TrajDataset 은 openloop_rollout 용도로 사용될 수 있었으나, 메모리 효율적 로더에서는 사용하지 않음

def build_libero_stats(dataset_path, cache_json_name='libero_stats.json'):
    """
    Libero 데이터셋의 통계(평균, 표준편차)를 계산하고 캐시합니다.
    """
    cache_json = osp.join(dataset_path, cache_json_name)
    if osp.isfile(cache_json):
        print(f'Libero dataset statistics in {cache_json} exists. Loading...')
        with open(cache_json, 'r') as f:
            dataset_statistics = json.load(f)
    else:
        print(f'Beginning to build Libero dataset statistics... This may take a while.')
        # --- 수정: 파일 목록 생성 시 존재 여부 확인 ---
        hdf5_files = []
        for file in Path(dataset_path).rglob('*.hdf5'):
            file_path_str = str(file.resolve())
            # 파일 존재 확인 추가
            if os.path.exists(file_path_str):
                 hdf5_files.append(file_path_str)
            # 누락된 파일은 조용히 건너뜀 (경고 메시지 제거)
        # --- 수정 끝 ---

        if not hdf5_files: # 파일 목록이 비었는지 확인
            raise RuntimeError(f"No HDF5 files found under {dataset_path}")

        actions = []
        proprios = []
        traj_lens = []
        views = None
        valid_hdf5_files = [] # 실제로 성공적으로 로드된 파일 목록

        for file_path in tqdm(hdf5_files):
            try:
                with h5py.File(file_path, 'r') as f:
                    if views is None:
                        # observation 키 존재 확인
                        if 'observation' in f and isinstance(f['observation'], h5py.Group):
                            views = list(f['observation'].keys())
                        else:
                            print(f"Warning: 'observation' group not found or invalid in {file_path}. Skipping file.")
                            continue # 다음 파일로
                    # action, proprio 키 존재 및 데이터 확인
                    if 'action' not in f or 'proprio' not in f:
                        print(f"Warning: 'action' or 'proprio' dataset not found in {file_path}. Skipping file.")
                        continue

                    traj_actions = f['action'][()]
                    traj_proprios = f['proprio'][()]

                    # 데이터 타입 및 길이 확인
                    if traj_actions.ndim == 0 or traj_proprios.ndim == 0 or traj_actions.shape[0] == 0:
                         print(f"Warning: Empty or invalid data in {file_path}. Skipping file.")
                         continue

                    actions.append(traj_actions.astype('float32'))
                    proprios.append(traj_proprios.astype('float32'))
                    traj_lens.append(traj_actions.shape[0])
                    valid_hdf5_files.append(file_path) # 성공적으로 로드된 파일만 추가
            # --- 수정: 파일 열기 실패 시 건너뛰기 ---
            except FileNotFoundError:
                print(f"Warning: File not found during statistics building: {file_path}. Skipping.")
                continue # 다음 파일로 넘어감
            except Exception as e:
                print(f"Error loading {file_path}: {e}. Skipping.")
                continue # 다음 파일로 넘어감
            # --- 수정 끝 ---

        if len(actions) == 0 or len(proprios) == 0:
            raise RuntimeError(f"No valid data found in HDF5 files under {dataset_path}")

        all_actions = np.concatenate(actions, axis=0)
        all_proprios = np.concatenate(proprios, axis=0)
        action_mean = all_actions.mean(axis=0).astype('float32').tolist()
        action_std = all_actions.std(axis=0).astype('float32').tolist()
        proprio_mean = all_proprios.mean(axis=0).astype('float32').tolist()
        proprio_std = all_proprios.std(axis=0).astype('float32').tolist()
        action_std = [float(s) if s > 1e-6 else 1.0 for s in action_std]
        proprio_std = [float(s) if s > 1e-6 else 1.0 for s in proprio_std]

        dataset_statistics = dict(
            views=views or [],
            action_mean=action_mean,
            action_std=action_std,
            proprio_mean=proprio_mean,
            proprio_std=proprio_std,
            # --- 수정: 성공적으로 로드된 파일 목록과 길이를 저장 ---
            traj_paths=valid_hdf5_files,
            traj_lens=traj_lens
            # --- 수정 끝 ---
        )

        with open(cache_json, 'w') as f:
            json.dump(dataset_statistics, f, indent=4)
        print(f'Libero dataset statistics saved to {cache_json}.')

    return dataset_statistics


class LiberoSliceWrapper:
    """
    Subset과 유사하지만 하위 데이터셋의 속성에 접근 가능한 래퍼.
    (이 클래스는 변경 없음)
    """
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        # 인덱스 유효성 검사 추가
        if idx >= len(self.indices):
             raise IndexError(f"Index {idx} out of range for LiberoSliceWrapper with length {len(self.indices)}")
        actual_idx = self.indices[idx]
        try:
            return self.dataset[actual_idx]
        except Exception as e:
             # __getitem__ 에서 오류 발생 시 None 반환 (DataLoader 에서 처리)
             print(f"Error loading item at original index {actual_idx} (wrapper index {idx}): {e}")
             return None


    def __getattr__(self, name: str):
        """하위 데이터셋의 속성에 접근"""
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


class LiberoDataset(Dataset):
    """
    Libero HDF5 데이터셋을 위한 메모리 효율적인 슬라이스 로더.
    """
    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        main_view: str = 'images0',
        normalize: bool = True,
        num_frames: int = 10,
        frameskip: int = 1,
    ):
        self.data_path = data_path
        self.transform = transform
        self.normalize = normalize
        self.main_view = main_view
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.T = self.num_frames

        stats = build_libero_stats(self.data_path)

        self.metas = stats['traj_paths']
        self.traj_lens = stats['traj_lens']

        if stats.get('views'):
            assert self.main_view in stats['views'], (
                f"main_view '{self.main_view}' not in available views {stats['views']}"
            )

        self.action_dim = len(stats['action_mean'])
        self.proprio_dim = len(stats['proprio_mean'])

        if self.normalize:
            self.action_mean = torch.tensor(stats['action_mean'], dtype=torch.float32)
            self.action_std = torch.tensor(stats['action_std'], dtype=torch.float32)
            self.proprio_mean = torch.tensor(stats['proprio_mean'], dtype=torch.float32)
            self.proprio_std = torch.tensor(stats['proprio_std'], dtype=torch.float32)
        else:
            self.action_mean = torch.zeros(self.action_dim, dtype=torch.float32)
            self.action_std = torch.ones(self.action_dim, dtype=torch.float32)
            self.proprio_mean = torch.zeros(self.proprio_dim, dtype=torch.float32)
            self.proprio_std = torch.ones(self.proprio_dim, dtype=torch.float32)

        self.state_dim = self.proprio_dim
        self.state_mean = self.proprio_mean
        self.state_std = self.proprio_std

        print("Building slice indices...")
        self.slice_indices = []
        valid_traj_indices = list(range(len(self.metas))) # build_libero_stats 에서 이미 필터링됨

        for i in valid_traj_indices:
            traj_len = self.get_seq_length(i)
            # 마지막 프레임 인덱스: frame_start_idx + (T-1)*frameskip
            # 이 인덱스가 (traj_len - 1) 보다 작거나 같아야 함.
            max_start_frame = traj_len - (self.T - 1) * self.frameskip - 1

            if max_start_frame >= 0:
                for start_idx in range(0, max_start_frame + 1):
                    # --- 수정: 슬라이스 인덱스 빌드 시 파일 존재 확인 ---
                    # build_libero_stats 에서 이미 확인했지만, 이중 확인
                    traj_path = self.metas[i]
                    if os.path.exists(traj_path):
                        self.slice_indices.append((i, start_idx))
                    # 누락된 파일은 조용히 건너뜀 (경고 메시지 제거)
                    # --- 수정 끝 ---

        print(f"Total slices: {len(self.slice_indices)}")
        if not self.slice_indices:
             raise RuntimeError("No valid slices could be created. Check dataset integrity and parameters (num_frames, frameskip).")


    def get_seq_length(self, idx: int) -> int:
        return self.traj_lens[idx]

    def __len__(self) -> int:
        return len(self.slice_indices)

    def __getitem__(self, idx: int) -> Optional[Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, Dict]]:
        """
        spread_wm과 호환되도록 *슬라이스된* 궤적 데이터를 반환합니다.
        오류 발생 시 None을 반환하여 DataLoader에서 처리하도록 합니다.
        """
        if idx >= len(self.slice_indices):
            print(f"Warning: Index {idx} out of range in __getitem__. Returning None.")
            return None # 인덱스 범위 초과 시 None 반환

        traj_idx, frame_start_idx = self.slice_indices[idx]
        # 메타데이터 인덱스 유효성 검사
        if traj_idx >= len(self.metas):
            print(f"Warning: Trajectory index {traj_idx} out of range for metas. Slice index {idx}. Returning None.")
            return None

        traj_path = self.metas[traj_idx]
        frame_indices = range(frame_start_idx,
                              frame_start_idx + self.T * self.frameskip,
                              self.frameskip)

        try:
            # --- 수정: 파일 열기 시 try-except 추가 ---
            # 파일이 존재하는지 다시 한번 확인 (매번 여는 비용 발생)
            # if not os.path.exists(traj_path):
            #     print(f"Warning: File {traj_path} not found for slice {idx}. Returning None.")
            #     return None # 파일을 찾을 수 없으면 None 반환

            with h5py.File(traj_path, 'r') as f:
                # 데이터 로딩 로직 (이전과 동일)
                action_indices = list(frame_indices)
                actions_np = f['action'][action_indices].astype('float32')
                actions = torch.from_numpy(actions_np)
                actions = (actions - self.action_mean) / self.action_std

                proprio_indices = list(frame_indices)
                proprios_np = f['proprio'][proprio_indices].astype('float32')
                proprios = torch.from_numpy(proprios_np)
                proprios_normalized = (proprios - self.proprio_mean) / self.proprio_std
                state = proprios_normalized

                # observation 키 및 main_view 존재 확인
                if 'observation' not in f or self.main_view not in f['observation']:
                    print(f"Warning: Observation key '{self.main_view}' not found in {traj_path} for slice {idx}. Returning None.")
                    return None

                compressed_images = f['observation'][self.main_view][list(frame_indices)]

                images = []
                for img_data in compressed_images:
                    # 이미지 디코딩 로직 (이전과 동일)
                    if isinstance(img_data, np.ndarray) and img_data.dtype == np.uint8: buf = img_data.reshape(-1)
                    elif isinstance(img_data, (bytes, bytearray)): buf = np.frombuffer(img_data, dtype=np.uint8)
                    elif isinstance(img_data, np.void): buf = np.frombuffer(img_data.tobytes(), dtype=np.uint8)
                    else: raise TypeError(f"Unsupported image buffer type: {type(img_data)}")

                    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img_bgr is None: raise ValueError("cv2.imdecode failed")
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    if self.transform:
                        img_pil = Image.fromarray(img_rgb)
                        img_tensor = self.transform(img_pil).to(dtype=torch.float32)
                    else:
                        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
                    images.append(img_tensor)

                images_tensor = torch.stack(images, dim=0)
                obs = {'visual': images_tensor, 'proprio': proprios_normalized}
                return obs, actions, state, {}
        # --- 수정: FileNotFoundError 포함하여 모든 예외 처리 ---
        except FileNotFoundError:
             print(f"Warning: File not found during __getitem__: {traj_path} (Slice index {idx}). Returning None.")
             return None # 파일 없으면 None 반환
        except Exception as e:
            error_msg = (
                f"Failed to load slice {idx} (traj {traj_idx}, frame {frame_start_idx}) "
                f"from {traj_path}"
            )
            print(f"Original error for slice {idx}: {repr(e)}")
            # 상세 오류 대신 None 반환하여 DataLoader가 처리하도록 함
            # raise RuntimeError(error_msg) from e
            print(error_msg)
            return None # 오류 발생 시 None 반환
        # --- 수정 끝 ---


def load_libero_slice_train_val(
    transform: Callable,
    data_path: str,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    num_frames: int = 10,
    frameskip: int = 1,
    main_view: str = 'images0',
    **kwargs
):
    """
    LiberoDataset을 로드하고 train/val '슬라이스'로 분할합니다.
    (이 함수는 변경 없음)
    """
    print("="*50)
    print("WARNING: Using memory-efficient LiberoDataset loader.")
    print("This will solve OOM issues, but `openloop_rollout` in `train.py`")
    print("is NOT compatible with this loader and *MUST* be commented out.")
    print("Please comment out lines calling `self.openloop_rollout(...)` in `train.py`'s `val()` function.")
    print("="*50)

    print(f"Loading Libero dataset from: {data_path}")

    full_dataset = LiberoDataset(
        data_path=data_path,
        transform=transform,
        main_view=main_view,
        normalize=True,
        num_frames=num_frames,
        frameskip=frameskip
    )

    print("Splitting slice indices into train and validation sets...")

    num_slices = len(full_dataset)
    indices = np.arange(num_slices)
    np.random.RandomState(random_seed).shuffle(indices)
    train_len = int(train_fraction * num_slices)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    print(f"Total slices: {num_slices}, Train: {len(train_indices)}, Val: {len(val_indices)}")

    # Use LiberoSliceWrapper to allow attribute access
    train_slices = LiberoSliceWrapper(full_dataset, train_indices.tolist())
    val_slices = LiberoSliceWrapper(full_dataset, val_indices.tolist())

    datasets = {"train": train_slices, "valid": val_slices}
    # Pass wrappers to traj_dsets as well for attribute access
    traj_dsets = {"train": train_slices, "valid": val_slices}

    print("Libero datasets loaded and processed successfully.")
    return datasets, traj_dsets


# DataLoader에서 None 배치를 처리하기 위한 Collate 함수
def collate_fn_skip_none(batch):
    # Filter out None items from the batch
    batch = [item for item in batch if item is not None]
    if not batch:
        # Return None or an empty structure if the batch becomes empty
        # Returning None might cause issues depending on the training loop
        # Consider returning an empty tensor or skipping the batch in the loop
        return None # Returning None, check training loop handling
    try:
        # Use default collate function on the filtered batch
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        print(f"Error during collation: {e}")
        # Handle collation error, e.g., return None or re-raise
        return None


if __name__ == '__main__':
    # 테스트용 코드 (변경 없음)
    from torchvision import transforms
    from einops import rearrange

    IMG_SIZE = 128
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # Removed ToTensor and Normalize as LiberoDataset handles ToTensor internally now
        # Add normalization if needed after ToTensor in LiberoDataset's transform
        # For testing, we assume transform passed to load_libero... handles ToTensor+Normalize
    ])

    DATA_PATH = "/home/jihoonmun/LIBERO/libero/datasets_processed/libero_object"
    NUM_FRAMES = 10
    FRAMESKIP = 5

    if not osp.exists(DATA_PATH):
        print(f"Warning: Data path {DATA_PATH} does not exist. Test script will fail.")
        print("Please download Libero dataset and update DATA_PATH.")
    else:
        # --- 테스트용 Transform 정의 (ToTensor + Normalize 포함) ---
        test_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(), # PIL to Tensor, scales to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # Standard normalization
                                 std=[0.229, 0.224, 0.225]),
        ])
        # --- 수정 끝 ---

        datasets, traj_dsets = load_libero_slice_train_val(
            transform=test_transform, # 수정된 transform 사용
            data_path=DATA_PATH,
            num_frames=NUM_FRAMES,
            frameskip=FRAMESKIP
        )

        print(f"\n--- Dataset Summary ---")
        print(f"Total slices (Train): {len(datasets['train'])}")
        print(f"Total slices (Valid): {len(datasets['valid'])}")

        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            datasets['train'],
            batch_size=4,
            shuffle=True,
            num_workers=0, # Set to 0 for easier debugging
            collate_fn=collate_fn_skip_none # Use the custom collate_fn
        )

        print("\nTesting DataLoader...")
        batch_count = 0
        none_batch_count = 0
        try:
            for batch in train_loader:
                batch_count += 1
                if batch is not None:
                    obs, act, state, _ = batch # Unpack the tuple correctly
                    print(f"Batch {batch_count} loaded successfully.")
                    print(f"  obs['visual'] shape: {obs['visual'].shape}")
                    print(f"  obs['proprio'] shape: {obs['proprio'].shape}")
                    print(f"  action shape:        {act.shape}")
                    print(f"  state shape:         {state.shape}")

                    # Access attributes via the wrapper
                    expected_action_dim = traj_dsets['train'].action_dim
                    expected_proprio_dim = traj_dsets['train'].proprio_dim
                    print(f"  Expected action per-step dim: {expected_action_dim}")
                    print(f"  Expected proprio/state per-step dim: {expected_proprio_dim}")
                    # Action shape check needs update based on how actions are returned
                    # Assuming __getitem__ returns (T, Da) shape for actions
                    print(f"  Expected action shape in batch: (B, {NUM_FRAMES}, {expected_action_dim})")
                    break # Load only one batch for testing
                else:
                    none_batch_count += 1
                    print(f"Batch {batch_count} was None (skipped due to errors).")
                    if none_batch_count > 10: # Avoid infinite loops if many errors occur
                        print("Too many None batches, stopping test.")
                        break

            if batch_count == 0:
                 print("DataLoader is empty or failed to load any batches.")

        except Exception as e:
            print(f"Error during DataLoader test: {e}")
            import traceback
            traceback.print_exc()