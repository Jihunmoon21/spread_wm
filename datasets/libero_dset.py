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
from .traj_dset import TrajDataset # TrajDataset은 openloop_rollout을 위해 남겨둘 수 있으나, 여기서는 제거

def build_libero_stats(dataset_path, cache_json_name='libero_stats.json'):
    """
    Libero 데이터셋의 통계(평균, 표준편차)를 계산하고 캐시합니다.
    (이 함수는 변경 없음)
    """
    cache_json = osp.join(dataset_path, cache_json_name)
    if osp.isfile(cache_json):
        print(f'Libero dataset statistics in {cache_json} exists. Loading...')
        with open(cache_json, 'r') as f:
            dataset_statistics = json.load(f)
    else:
        print(f'Beginning to build Libero dataset statistics... This may take a while.')
        hdf5_files = [str(file.resolve()) for file in Path(dataset_path).rglob('*.hdf5')]

        actions = []
        proprios = []
        traj_lens = []
        views = None

        for file_path in tqdm(hdf5_files):
            try:
                with h5py.File(file_path, 'r') as f:
                    if views is None:
                        views = list(f['observation'].keys())
                    traj_actions = f['action'][()].astype('float32')
                    traj_proprios = f['proprio'][()].astype('float32')
                    actions.append(traj_actions)
                    proprios.append(traj_proprios)
                    traj_lens.append(traj_actions.shape[0])
            except Exception as e:
                print(f"Error loading {file_path}: {e}. Skipping.")

        if len(actions) == 0 or len(proprios) == 0:
            raise RuntimeError(f"No valid HDF5 found under {dataset_path}")

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
            traj_paths=hdf5_files,
            traj_lens=traj_lens
        )

        with open(cache_json, 'w') as f:
            json.dump(dataset_statistics, f, indent=4)
        print(f'Libero dataset statistics saved to {cache_json}.')

    return dataset_statistics


class LiberoSliceWrapper:
    """
    Subset과 유사하지만 하위 데이터셋의 속성에 접근 가능한 래퍼.
    
    plan.py의 PlanWorkspace에서 dset.action_mean, dset.action_std, 
    dset.state_mean, dset.state_std, dset.proprio_mean, dset.proprio_std, 
    dset.transform 등의 속성에 접근해야 하기 때문에 필요합니다.
    
    torch.utils.data.Subset은 __getattr__를 구현하지 않아
    하위 데이터셋의 속성에 접근할 수 없습니다.
    """
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]
    
    def __getattr__(self, name: str):
        """하위 데이터셋의 속성에 접근"""
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


# [수정] TrajDataset 대신 torch.utils.data.Dataset 상속
class LiberoDataset(Dataset):
    """
    [수정] Libero HDF5 데이터셋을 위한 메모리 효율적인 슬라이스 로더.
    TrajSlicerDataset의 역할을 직접 수행합니다.
    """
    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        main_view: str = 'images0',
        normalize: bool = True,
        # --- [추가] 슬라이싱 인자 ---
        num_frames: int = 10,
        frameskip: int = 1,
    ):
        self.data_path = data_path
        self.transform = transform
        self.normalize = normalize
        self.main_view = main_view
        # --- [추가] 슬라이싱 인자 저장 ---
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.T = self.num_frames # SPREAD의 TrajSlicerDataset과 호환

        # 데이터셋 통계 로드
        stats = build_libero_stats(self.data_path)

        self.metas = stats['traj_paths']
        self.traj_lens = stats['traj_lens']

        # (가드) 뷰 검증
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
        
        # state는 proprio와 동일하게 처리 (plan.py의 PlanWorkspace에서 필요)
        self.state_mean = self.proprio_mean
        self.state_std = self.proprio_std
        
        # --- [추가] TrajSlicerDataset 로직을 여기로 가져옴 ---
        print("Building slice indices...")
        self.slice_indices = []
        traj_indices = list(range(len(self.metas)))
        
        for i in traj_indices:
            traj_len = self.get_seq_length(i)
            # T*frameskip 만큼의 프레임이 필요함
            # (T-1)*frameskip + 1 이 마지막 프레임 인덱스이므로,
            # T*frameskip (또는 (T-1)*frameskip + 1) 만큼의 길이가 필요함
            # TrajSlicerDataset은 T개의 프레임을 샘플링
            # (0, fs, 2fs, ..., (T-1)fs)
            # 마지막 인덱스는 (T-1)*frameskip
            # 따라서 필요한 총 길이는 (T-1)*frameskip + 1 입니다.
            # 하지만 action은 T개가 필요할 수 있습니다. 
            # SPREAD의 num_hist=3, num_pred=1 -> num_frames=4
            # (0, fs, 2fs, 3fs) -> 4개 프레임
            # 필요한 길이: 3*fs + 1
            
            # TrajSlicerDataset 로직을 정확히 따름: T개의 프레임이 필요.
            # 0, 1*fs, ..., (T-1)*fs.
            # 마지막 프레임 인덱스: frame_start_idx + (T-1)*frameskip
            # 이 인덱스가 (traj_len - 1) 보다 작거나 같아야 함.
            # frame_start_idx <= traj_len - 1 - (T-1)*frameskip
            
            # SPREAD의 TrajSlicerDataset은 T * frameskip 길이의 궤적이 필요
            required_len = self.T * self.frameskip
            max_start_idx = traj_len - required_len
            
            # SPREAD의 action 처리는 T*fs 만큼의 action을 T*Da로 concat
            # 이미지는 T개를 샘플링 (0, fs, ..., (T-1)fs)
            # 코드를 다시 보니, num_frames = num_hist + num_pred
            # T = num_frames
            # obs: (B, T, C, H, W), act: (B, T, Da), state: (B, T, Dp)
            # TrajSlicerDataset이 T개 프레임을 반환.
            # frame_indices = range(start_idx, start_idx + T * frameskip, frameskip)
            
            max_start_frame = traj_len - (self.T - 1) * self.frameskip - 1
            
            if max_start_frame >= 0:
                # 유효한 모든 시작 인덱스를 추가
                for start_idx in range(0, max_start_frame + 1):
                    self.slice_indices.append((i, start_idx)) # (궤적 인덱스, 프레임 시작 인덱스)
            
        print(f"Total slices: {len(self.slice_indices)}")
        # --- [추가] 종료 ---


    def get_seq_length(self, idx: int) -> int:
        return self.traj_lens[idx]

    # [수정] __len__은 전체 궤적 수가 아닌, 총 슬라이스 수
    def __len__(self) -> int:
        return len(self.slice_indices)

    # [수정] __getitem__은 이제 전체 궤적이 아닌 슬라이스를 반환
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, Dict]:
        """
        spread_wm과 호환되도록 *슬라이스된* 궤적 데이터를 반환합니다.
        """
        
        # --- 슬라이스 인덱스 가져오기 ---
        traj_idx, frame_start_idx = self.slice_indices[idx]
        traj_path = self.metas[traj_idx]
        
        # --- 필요한 프레임 인덱스 계산 (T개) ---
        frame_indices = range(frame_start_idx, 
                              frame_start_idx + self.T * self.frameskip, 
                              self.frameskip)
        
        try:
            with h5py.File(traj_path, 'r') as f:
                
                # 1) Action 로드 및 정규화
                # SPREAD의 "concat" 모드는 T*frameskip 길이의 action을 (T*Da*frameskip)로 압축
                # train.py는 num_hist=3, num_pred=1 -> T=4
                # frameskip=5
                # TrajSlicerDataset(process_actions="concat")은
                # (T, Da)를 (T*Da)로 만듦. (frameskip은 이미 궤적 자체에 적용됨)
                # 아, frameskip은 여기서 적용하는 게 맞음.
                # TrajSlicerDataset은 (T, Da)를 반환. (T*fs, Da) -> (T, fs*Da)
                
                # SPREAD의 TrajSlicerDataset(process_actions="concat") 로직:
                # T=num_frames, fs=frameskip
                # action: (T*fs, Da) -> (T, fs*Da)
                # obs: (T*fs, ...) -> (T, ...) (fs 간격으로 샘플링)
                
                # [수정] SPREAD 로직에 맞게 수정
                
                # 1) Action: (frame_start_idx) 부터 (T * fs) 개의 action 로드
                action_start = frame_start_idx
                action_end = frame_start_idx + self.T * self.frameskip
                
                # [수정] 원본 TrajSlicerDataset은 T개의 action만 반환 (이미 fs가 적용된 궤적에서)
                # 이 코드는 원본 HDF5를 다루므로, fs를 적용해야 함.
                
                # 1.1) TrajSlicerDataset(process_actions="concat") 로직
                # (T, Da) -> (T*Da)
                # 여기서 T = num_frames (e.g., 4)
                # LiberoDataset은 (T, Da)를 반환했었음.
                # 이 T는 궤적 전체 길이.
                # TrajSlicerDataset은 (T_slice, Da_action)을 (T_slice * Da_action)으로.
                # T_slice = num_frames (e.g., 4)
                
                action_indices = list(frame_indices) # (T,)
                actions_np = f['action'][action_indices].astype('float32') # (T, Da)
                actions = torch.from_numpy(actions_np)
                actions = (actions - self.action_mean) / self.action_std
                
                # 2) Proprio (State) 로드 및 정규화
                proprio_indices = list(frame_indices) # (T,)
                proprios_np = f['proprio'][proprio_indices].astype('float32')     # (T, Dp)
                proprios = torch.from_numpy(proprios_np)             # float32
                proprios_normalized = (proprios - self.proprio_mean) / self.proprio_std
                state = proprios_normalized                          # (T, Dp)

                # 3) Image 로드 및 변환
                img_indices = list(frame_indices) # (T,)
                compressed_images = f['observation'][self.main_view][img_indices] # (T, ...)
                
                images = []
                # [수정] 이제 루프는 T번만 (e.g., 4번) 돕니다.
                for img_data in compressed_images:
                    # img_data -> 1D uint8 버퍼로 보장
                    if isinstance(img_data, np.ndarray) and img_data.dtype == np.uint8:
                        buf = img_data.reshape(-1)
                    elif isinstance(img_data, (bytes, bytearray)):
                        buf = np.frombuffer(img_data, dtype=np.uint8)
                    elif isinstance(img_data, np.void):
                        buf = np.frombuffer(img_data.tobytes(), dtype=np.uint8)
                    else:
                        raise TypeError(
                            f"Unsupported image buffer type: {type(img_data)}, "
                            f"dtype: {getattr(img_data, 'dtype', None)}"
                        )

                    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        raise ValueError("cv2.imdecode failed; check image buffer/content")

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    if self.transform:
                        img_pil = Image.fromarray(img_rgb)
                        img_tensor = self.transform(img_pil).to(dtype=torch.float32)
                    else:
                        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(dtype=torch.float32) / 255.0

                    images.append(img_tensor)

                images_tensor = torch.stack(images, dim=0)  # (T, C, H, W) float32

                # 4) Obs dict 생성 (pusht_dset.py 형식)
                obs = {
                    'visual': images_tensor,
                    'proprio': proprios_normalized
                }

                # (obs, act, state, {}) 튜플 반환
                return obs, actions, state, {}

        except Exception as e:
            error_msg = (
                f"Failed to load slice {idx} (traj {traj_idx}, frame {frame_start_idx}) "
                f"from {traj_path}"
            )
            print(f"Original error for slice {idx}: {repr(e)}")
            raise RuntimeError(error_msg) from e


# [수정] load_libero_slice_train_val 함수 수정
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
    [수정] LiberoDataset을 로드하고 train/val '슬라이스'로 분할합니다.
    TrajSlicerDataset을 더 이상 사용하지 않습니다.
    """
    print("="*50)
    print("WARNING: Using memory-efficient LiberoDataset loader.")
    print("This will solve OOM issues, but `openloop_rollout` in `train.py`")
    print("is NOT compatible with this loader and *MUST* be commented out.")
    print("Please comment out lines calling `self.openloop_rollout(...)` in `train.py`'s `val()` function.")
    print("="*50)
    
    print(f"Loading Libero dataset from: {data_path}")
    
    # [수정] LiberoDataset이 슬라이싱 인자를 받도록 수정
    full_dataset = LiberoDataset(
        data_path=data_path,
        transform=transform,
        main_view=main_view,
        normalize=True,
        num_frames=num_frames,
        frameskip=frameskip
    )

    print("Splitting slice indices into train and validation sets...")
    
    # [수정] 궤적(trajectory)이 아닌 슬라이스(slice) 인덱스를 분할
    num_slices = len(full_dataset)
    indices = np.arange(num_slices)
    
    # SPREAD의 TrajSlicerDataset 셔플 로직을 따름
    np.random.RandomState(random_seed).shuffle(indices) 
    
    train_len = int(train_fraction * num_slices)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]
    
    print(f"Total slices: {num_slices}, Train: {len(train_indices)}, Val: {len(val_indices)}")

    # [수정] LiberoSliceWrapper를 사용하여 full_dataset 공유
    # Subset 대신 LiberoSliceWrapper를 사용하는 이유:
    # plan.py의 PlanWorkspace에서 dset.action_mean, dset.transform 등의
    # 속성에 접근해야 하는데, Subset은 __getattr__를 구현하지 않아
    # 하위 데이터셋의 속성에 접근할 수 없습니다.
    train_slices = LiberoSliceWrapper(full_dataset, train_indices.tolist())
    val_slices = LiberoSliceWrapper(full_dataset, val_indices.tolist())

    datasets = {"train": train_slices, "valid": val_slices}
    
    # [수정] traj_dsets는 train.py에서 속성 접근(action_dim) 및 롤아웃에 사용
    # 롤아웃은 비활성화해야 하지만, 속성 접근을 위해 래퍼를 그대로 전달
    traj_dsets = {"train": train_slices, "valid": val_slices}

    print("Libero datasets loaded and processed successfully.")
    return datasets, traj_dsets


# DataLoader에서 None 배치를 처리하기 위한 Collate 함수
# (이 함수는 변경 없음)
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    # 테스트용: (변경 없음)
    from torchvision import transforms
    from einops import rearrange # 테스트용 임포트

    IMG_SIZE = 128
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    DATA_PATH = "/home/jihun/LIBERO/libero/datasets_processed/libero_object"
    NUM_FRAMES = 10 # num_hist + num_pred
    FRAMESKIP = 5

    if not osp.exists(DATA_PATH):
        print(f"Warning: Data path {DATA_PATH} does not exist. Test script will fail.")
        print("Please download Libero dataset and update DATA_PATH.")
    else:
        datasets, traj_dsets = load_libero_slice_train_val(
            transform=transform,
            data_path=DATA_PATH,
            num_frames=NUM_FRAMES,
            frameskip=FRAMESKIP
        )

        print(f"\n--- Dataset Summary ---")
        # [수정] traj_dsets는 이제 슬라이스 데이터셋(Subset)을 가리킴
        print(f"Total slices (Train): {len(datasets['train'])}")
        print(f"Total slices (Valid): {len(datasets['valid'])}")

        # DataLoader 테스트
        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            datasets['train'],
            batch_size=4,
            shuffle=True, # Subset을 사용하므로 shuffle=True가 의미 있음
            num_workers=0,
            collate_fn=collate_fn_skip_none
        )

        print("\nTesting DataLoader...")
        try:
            batch = next(iter(train_loader))
            if batch:
                obs, act, state = batch
                print(f"Batch loaded successfully.")
                print(f"  obs['visual'] shape: {obs['visual'].shape}")   # (B, T, C, H, W)
                print(f"  obs['proprio'] shape: {obs['proprio'].shape}") # (B, T, Dp)
                print(f"  action shape:        {act.shape}")              # [수정] (B, T*Da)
                print(f"  state shape:         {state.shape}")            # (B, T, Dp)
                
                # [수정] Subset의 하위 dataset에서 속성 접근
                expected_action_dim = traj_dsets['train'].dataset.action_dim
                expected_proprio_dim = traj_dsets['train'].dataset.proprio_dim
                print(f"  Expected action per-step dim: {expected_action_dim}")
                print(f"  Expected proprio/state per-step dim: {expected_proprio_dim}")
                print(f"  Expected action shape: (B, {NUM_FRAMES * expected_action_dim})")
            else:
                print("DataLoader returned an empty batch (possibly due to errors).")

        except StopIteration:
            print("DataLoader is empty. Check data path and dataset implementation.")
        except Exception as e:
            print(f"Error during DataLoader test: {e}")