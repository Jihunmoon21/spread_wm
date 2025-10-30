import torch
import numpy as np
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
        return rearrange(obs_visual, "b t h w c -> b t c h w") / 255.0

    def transform_obs_visual(self, obs_visual):
        # obs_visual은 numpy array (B, T, H, W, C), uint8 형식
        if not isinstance(obs_visual, np.ndarray):
            obs_visual = np.array(obs_visual)
        
        b, t = obs_visual.shape[:2]
        
        # transform을 각 이미지에 개별적으로 적용 (numpy -> PIL -> transform -> tensor)
        if self.transform is not None:
            # (B, T, H, W, C) -> (B*T, H, W, C)
            flat_imgs = obs_visual.reshape(-1, *obs_visual.shape[2:])
            
            # 각 이미지에 transform 적용 (PIL Image로 변환 후)
            from PIL import Image
            transformed_imgs = []
            for img in flat_imgs:
                # numpy (H, W, C) uint8 -> PIL Image -> transform -> tensor
                pil_img = Image.fromarray(img.astype(np.uint8))
                transformed_img = self.transform(pil_img)
                transformed_imgs.append(transformed_img)
            
            # 리스트를 텐서로 변환하고 reshape
            transformed_imgs = torch.stack(transformed_imgs)
            # (B*T, C, H', W') -> (B, T, C, H', W')
            transformed_obs_visual = transformed_imgs.reshape(b, t, *transformed_imgs.shape[1:])
        else:
            # transform이 없으면 수동으로 정규화
            transformed_obs_visual = torch.tensor(obs_visual)
            transformed_obs_visual = self.preprocess_obs_visual(transformed_obs_visual)
        
        return transformed_obs_visual
        
    def transform_obs(self, obs):
        '''
        np arrays to tensors
        '''
        transformed_obs = {}
        transformed_obs['visual'] = self.transform_obs_visual(obs['visual'])
        transformed_obs['proprio'] = self.normalize_proprios(torch.tensor(obs['proprio']))
        return transformed_obs
