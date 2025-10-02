import torch
from einops import rearrange

class Preprocessor:
    def __init__(self, 
        action_mean,
        action_std,
        state_mean,
        state_std,
        proprio_mean,
        proprio_std,
        transform,
    ):
        self.action_mean = action_mean
        self.action_std = action_std
        self.state_mean = state_mean
        self.state_std = state_std
        self.proprio_mean = proprio_mean
        self.proprio_std = proprio_std
        self.transform = transform

    def normalize_actions(self, actions):
        '''
        actions: (b, t, action_dim)  
        '''
        # 디바이스 일치시키기
        action_mean = self.action_mean.to(actions.device)
        action_std = self.action_std.to(actions.device)
        return (actions - action_mean) / action_std

    def denormalize_actions(self, actions):
        '''
        actions: (b, t, action_dim)  
        '''
        # 디바이스 일치시키기
        action_mean = self.action_mean.to(actions.device)
        action_std = self.action_std.to(actions.device)
        return actions * action_std + action_mean
    
    def normalize_proprios(self, proprio):
        '''
        input shape (..., proprio_dim)
        '''
        # 디바이스 일치시키기
        proprio_mean = self.proprio_mean.to(proprio.device)
        proprio_std = self.proprio_std.to(proprio.device)
        return (proprio - proprio_mean) / proprio_std

    def normalize_states(self, state):
        '''
        input shape (..., state_dim)
        '''
        # 디바이스 일치시키기
        state_mean = self.state_mean.to(state.device)
        state_std = self.state_std.to(state.device)
        return (state - state_mean) / state_std

    def preprocess_obs_visual(self, obs_visual):
        # 4D 입력 처리
        if obs_visual.ndim == 4:
            B, T, H, C = obs_visual.shape
            
            # 메모리 사용량 체크 (67GB 할당 방지)
            expected_memory_gb = (B * T * H * H * C * 4) / (1024**3)  # 4 bytes per float32
            if expected_memory_gb > 10:  # 10GB 이상이면 건너뛰기
                raise ValueError(f"Memory allocation too large: {expected_memory_gb:.1f}GB. "
                               f"Skipping visualization for shape {obs_visual.shape}")
            
            # 원시 이미지인지 확인 (C=3이고 H가 이미지 크기인 경우)
            if C == 3 and H > 50:  # 원시 이미지로 판단
                # W 차원이 누락된 경우, H와 같은 크기로 확장
                obs_visual = obs_visual.unsqueeze(-2).expand(B, T, H, H, C)
            else:
                # 이미 처리된 feature 데이터인 경우, 5D로 변환하지 않고 에러 발생
                raise ValueError(f"Unexpected 4D visual data shape: {obs_visual.shape}. "
                               f"Expected 5D for raw images or different processing for features.")
        
        return rearrange(obs_visual, "b t h w c -> b t c h w") / 255.0

    def transform_obs_visual(self, obs_visual):
        transformed_obs_visual = torch.tensor(obs_visual)
        transformed_obs_visual = self.preprocess_obs_visual(transformed_obs_visual)
        transformed_obs_visual = self.transform(transformed_obs_visual)
        return transformed_obs_visual
    
    def transform_obs_visual_for_visualization(self, obs_visual):
        """
        시각화를 위한 관측 데이터 변환 (메모리 효율적)
        일반적인 transform과 달리 시각화에 최적화됨
        """
        try:
            # numpy array를 tensor로 변환
            if not isinstance(obs_visual, torch.Tensor):
                obs_visual = torch.tensor(obs_visual)
            
            # 메모리 사용량 체크
            if obs_visual.numel() * 4 > 10 * (1024**3):  # 10GB 이상
                print(f"[WARNING] Skipping visualization due to large memory requirement: {obs_visual.shape}")
                return obs_visual
            
            # 기본 전처리만 수행 (transform은 생략하여 메모리 절약)
            transformed_obs_visual = self.preprocess_obs_visual(obs_visual)
            return transformed_obs_visual
            
        except Exception as e:
            print(f"[WARNING] Visualization preprocessing failed: {e}")
            return obs_visual
    
    def transform_obs(self, obs):
        '''
        np arrays to tensors
        '''
        transformed_obs = {}
        transformed_obs['visual'] = self.transform_obs_visual(obs['visual'])
        transformed_obs['proprio'] = self.normalize_proprios(torch.tensor(obs['proprio']))
        return transformed_obs
