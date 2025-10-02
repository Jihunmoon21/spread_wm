import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True


# class DinoV3Encoder(nn.Module):
#     def __init__(
#         self,
#         name: str = "dinov3_vits16",
#         repo_name: str = "facebookresearch/dinov3",
#         feature_key: str = "x_norm_patchtokens",
#     ):
#         """
#         DINOv3 모델을 로드하고 World Model을 위한 인코더 역할을 하는 클래스입니다.

#         Args:
#             name (str): PyTorch Hub에서 불러올 DINOv3 모델의 이름.
#             repo_name (str): DINOv3 모델이 있는 Hub 저장소 이름.
#             feature_key (str): 사용할 특징의 종류 ('x_norm_patchtokens' 또는 'x_norm_clstoken').
#         """
#         super().__init__()
#         self.name = name
#         self.repo_name = repo_name
#         self.feature_key = feature_key
        
#         # PyTorch Hub를 통해 DINOv3 모델을 로드합니다.
#         self.base_model = torch.hub.load(self.repo_name, self.name)
        
#         # DINOv3 모델의 임베딩 차원을 가져옵니다.
#         self.emb_dim = self.base_model.embed_dim 
        
#         # 출력 텐서의 차원 수를 결정합니다. (패치 토큰은 2D, CLS 토큰은 1D)
#         if feature_key == "x_norm_patchtokens":
#             self.latent_ndim = 2
#         elif feature_key == "x_norm_clstoken":
#             self.latent_ndim = 1
#         else:
#             raise ValueError(f"Invalid feature key: {feature_key}")

#         self.patch_size = self.base_model.patch_size

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         입력 이미지 텐서를 받아 DINOv3 특징 벡터를 반환합니다.

#         Args:
#             x (torch.Tensor): (B, C, H, W) 형태의 이미지 텐서.

#         Returns:
#             torch.Tensor: 처리된 특징 벡터.
#         """
#         # DINOv3 모델로부터 중간 레이어 출력을 가져옵니다.
#         # 디버깅을 통해 이 출력이 이미 CLS와 레지스터 토큰이 제거된
#         # 순수한 196개의 이미지 패치 토큰임을 확인했습니다.
#         features = self.base_model.get_intermediate_layers(x)[0]
        
#         if self.feature_key == "x_norm_patchtokens":
#             # 추가적인 슬라이싱 없이, 받은 features를 그대로 사용합니다.
#             emb = features
#         elif self.feature_key == "x_norm_clstoken":
#             # 현재 get_intermediate_layers 출력 방식에서는 CLS 토큰을 지원하지 않습니다.
#             raise NotImplementedError("x_norm_clstoken is not supported with this forward pass.")
#         else:
#             raise ValueError(f"Invalid feature key: {self.feature_key}")

#         # VQVAE와 같은 디코더는 (B, T, N, E) 형태를 기대하므로,
#         # 이 클래스의 출력은 (B, N, E)가 되어야 합니다. 
#         # 따라서 latent_ndim 관련 로직은 여기서는 필요하지 않습니다.
#         # (visual_world_model.py에서 rearrange를 통해 처리됨)
        
#         return emb
    
class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb