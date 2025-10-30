import os
import torch
import random
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from typing import Callable, Dict, List, Tuple, Any, Sequence, Union, Optional
import psutil
from collections import deque
import copy

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)  # Memory usage in MB

def get_available_ram():
    mem = psutil.virtual_memory()
    return mem.available / (1024 * 1024 * 1024)  # Available memory in MB

def dict_to_namespace(cfg_dict):
    args = argparse.Namespace()
    for key in cfg_dict:
        setattr(args, key, cfg_dict[key])
    return args

def move_to_device(dct, device):
    for key, value in dct.items():
        if isinstance(value, torch.Tensor):
            dct[key] = value.to(device)
    return dct

def slice_trajdict_with_t(data_dict, start_idx=0, end_idx=None, step=1):
    if end_idx is None:
        end_idx = max(arr.shape[1] for arr in data_dict.values())
    return {key: arr[:, start_idx:end_idx:step, ...] for key, arr in data_dict.items()}

def concat_trajdict(dcts):
    full_dct = {}
    for k in dcts[0].keys():
        if isinstance(dcts[0][k], np.ndarray):
            full_dct[k] = np.concatenate([dct[k] for dct in dcts], axis=1)
        elif isinstance(dcts[0][k], torch.Tensor):
            full_dct[k] = torch.cat([dct[k] for dct in dcts], dim=1)
        else:
            raise TypeError(f"Unsupported data type: {type(dcts[0][k])}")
    return full_dct

ArrayLike = Union[np.ndarray, torch.Tensor]
TrajDict = Dict[str, ArrayLike]

def _ensure_same_keys(dcts: List[TrajDict]) -> None:
    """모든 딕셔너리가 같은 키를 가지는지 확인"""
    if len(dcts) <= 1:
        return
    
    first_keys = set(dcts[0].keys())
    for i, dct in enumerate(dcts[1:], 1):
        if set(dct.keys()) != first_keys:
            raise ValueError(f"Dictionary {i} has different keys: {set(dct.keys())} vs {first_keys}")

def _is_torch(arr) -> bool:
    """배열이 torch.Tensor인지 확인"""
    import torch
    return isinstance(arr, torch.Tensor)

def stack_trajdict(dcts: List[TrajDict], axis: int = 1) -> TrajDict:
    """
    여러 시점(또는 블록)을 새 시간축처럼 쌓기.
    - 값이 [B, ...]이면 stack으로 새 축 생성 → [B, L, ...]
    - 값이 [B, K, ...]이면 concat으로 이어붙임 → [B, K_total, ...]
    """
    assert len(dcts) > 0, "Empty dcts"
    _ensure_same_keys(dcts)
    out: TrajDict = {}
    for k in dcts[0].keys():
        arrs = [d[k] for d in dcts]
        a0 = arrs[0]
        if a0.ndim == 2:  # [B, D] 같은 경우
            if _is_torch(a0):
                out[k] = torch.stack(arrs, dim=axis)
            else:
                out[k] = np.stack(arrs, axis=axis)
        else:
            if _is_torch(a0):
                out[k] = torch.cat(arrs, dim=axis)
            else:
                out[k] = np.concatenate(arrs, axis=axis)
    return out

def aggregate_dct(dcts):
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct

def sample_tensors(tensors, n, indices=None):
    if indices is None:
        b = tensors[0].shape[0]
        indices = torch.randperm(b)[:n]
    indices = torch.tensor(indices)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            tensors[i] = tensor[indices]
    return tensors

def cfg_to_dict(cfg):
    cfg_dict = OmegaConf.to_container(cfg)
    for key in cfg_dict:
        if isinstance(cfg_dict[key], list):
            # --- 수정된 부분 ---
            # 리스트가 비어있지 않고, 첫 번째 항목이 문자열(str)인지 확인
            if cfg_dict[key] and isinstance(cfg_dict[key][0], str):
                cfg_dict[key] = ",".join(cfg_dict[key])
            # 그 외의 경우 (예: int 리스트)는 리스트를 그대로 둠
            # --- 수정 끝 ---
    return cfg_dict

def reduce_dict(f: Callable, d: Dict):
    return {k: reduce_dict(f, v) if isinstance(v, dict) else f(v) for k, v in d.items()}

def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")

def select_data_from_indices(data_dict, indices, traj_len):
    """
    numpy 배열로 구성된 딕셔너리에서 특정 프레임 인덱스의 데이터를 선택합니다.
    음수 인덱스(예: -1)를 지원합니다.

    Args:
        data_dict (dict): 키가 데이터 이름이고 값이 (T, ...) 형태의 numpy 배열인 딕셔너리.
                          예: {'visual': (T, C, H, W), 'proprio': (T, D)}
        indices (list[int]): 선택할 프레임 인덱스 리스트 (예: [0, 10, -1]).
        traj_len (int): 궤적의 전체 길이 (T).

    Returns:
        dict: 선택된 데이터가 쌓인 새로운 딕셔너리.
              예: {'visual': (N_indices, C, H, W), 'proprio': (N_indices, D)}
    """
    absolute_indices = []
    for idx in indices:
        if idx < 0:
            # 음수 인덱스를 양수 인덱스로 변환 (예: -1 -> traj_len - 1)
            abs_idx = traj_len + idx
        else:
            abs_idx = idx
        
        # 인덱스가 유효한 범위 내에 있는지 확인
        if 0 <= abs_idx < traj_len:
            absolute_indices.append(abs_idx)
        else:
            # 유효하지 않은 인덱스 경고 (선택 사항)
            print(f"Warning: Index {idx} (absolute: {abs_idx}) is out of bounds for traj_len {traj_len}. Skipping.")

    if not absolute_indices:
        # 유효한 인덱스가 없으면 빈 배열 반환
        return {key: np.array([]) for key in data_dict.keys()}

    selected_data = {}
    for key, data in data_dict.items():
        try:
            # numpy의 고급 인덱싱을 사용하여 한 번에 모든 프레임 선택
            selected_data[key] = data[absolute_indices]
        except Exception as e:
            print(f"Error selecting indices {absolute_indices} from key {key} with shape {data.shape}: {e}")
            selected_data[key] = np.array([]) # 오류 발생 시 빈 배열

    return selected_data