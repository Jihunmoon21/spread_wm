# ------------------------------------------
# https://github.com/JamesQFreeman/LoRA-ViT.git
# Sheng Wang at Feb 22 2023
# ------------------------------------------
# Modification:
# Added code for Online LoRA. 
#       -- Xiwen Wei
# ------------------------------------------

# models/lora.py

import math
import torch
import torch.nn as nn
from models.vit import ViTPredictor as ViT
from torch import Tensor

class _LoRA_qkv_timm(nn.Module):
    def __init__(self, qkv: nn.Module, online_mode: bool = True):
        super().__init__()
        self.qkv = qkv
        self.online_mode = online_mode
        self.dim = qkv.in_features
        self.inner_dim = qkv.out_features // 3
        
        # 일반 LoRA 가중치 (A, B)
        self.lora_a_q = None
        self.lora_b_q = None
        self.lora_a_v = None
        self.lora_b_v = None

        # 온라인 LoRA 전용 가중치
        if self.online_mode:
            self.w_a_q = None
            self.w_b_q = None
            self.w_a_v = None
            self.w_b_v = None
            self.wnew_a_q = None
            self.wnew_b_q = None
            self.wnew_a_v = None
            self.wnew_b_v = None
    @property
    def in_features(self):
        return self.qkv.in_features

    @property
    def out_features(self):
        return self.qkv.out_features

    @property
    def weight(self):
        return self.qkv.weight

    @property
    def bias(self):
        return self.qkv.bias
    
    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        if self.online_mode:
            # 온라인 모드: w (고정) + w_new (학습)
            q = q + self.w_b_q(self.w_a_q(x)) + self.wnew_b_q(self.wnew_a_q(x))
            v = v + self.w_b_v(self.w_a_v(x)) + self.wnew_b_v(self.wnew_a_v(x))
        else:
            # 일반 모드: lora (학습)
            q = q + self.lora_b_q(self.lora_a_q(x))
            v = v + self.lora_b_v(self.lora_a_v(x))
            
        return torch.cat([q, k, v], dim=-1)

# models/lora.py

class LoRA_ViT_spread(nn.Module):
    def __init__(self, vit_model: ViT, r: int = 4, online_mode: bool = True):
        super(LoRA_ViT_spread, self).__init__()
        assert r > 0
        self.online_mode = online_mode

        # 기본 ViT 모델의 모든 파라미터를 동결
        for param in vit_model.parameters():
            param.requires_grad = False

        # LoRA 가중치를 저장할 리스트 초기화
        self.lora_As = []
        self.lora_Bs = []
        if self.online_mode:
            self.w_As, self.w_Bs = [], []
            self.wnew_As, self.wnew_Bs = [], []

        # --- 추가: vit_model의 디바이스를 가져옵니다. ---
        device = next(vit_model.parameters()).device

        for layer in vit_model.transformer.layers:
            attn_block = layer[0]
            w_qkv_linear = attn_block.to_qkv
            dim = w_qkv_linear.in_features
            
            # 기존 to_qkv 레이어를 LoRA 기능이 추가된 레이어로 교체
            attn_block.to_qkv = _LoRA_qkv_timm(w_qkv_linear, online_mode=self.online_mode)
            inner_dim = attn_block.to_qkv.inner_dim

            if self.online_mode:
                # 온라인 LoRA 가중치 생성하고 바로 GPU로 보냅니다.
                w_a_q = nn.Linear(dim, r, bias=False).to(device)
                w_b_q = nn.Linear(r, inner_dim, bias=False).to(device)
                w_a_v = nn.Linear(dim, r, bias=False).to(device)
                w_b_v = nn.Linear(r, inner_dim, bias=False).to(device)
                wnew_a_q = nn.Linear(dim, r, bias=False).to(device)
                wnew_b_q = nn.Linear(r, inner_dim, bias=False).to(device)
                wnew_a_v = nn.Linear(dim, r, bias=False).to(device)
                wnew_b_v = nn.Linear(r, inner_dim, bias=False).to(device)

                # w 가중치는 동결, w_new 가중치만 학습
                for param in w_a_q.parameters(): param.requires_grad = False
                for param in w_b_q.parameters(): param.requires_grad = False
                for param in w_a_v.parameters(): param.requires_grad = False
                for param in w_b_v.parameters(): param.requires_grad = False
                
                self.w_As.extend([w_a_q, w_a_v])
                self.w_Bs.extend([w_b_q, w_b_v])
                self.wnew_As.extend([wnew_a_q, wnew_a_v])
                self.wnew_Bs.extend([wnew_b_q, wnew_b_v])

                # _LoRA_qkv_timm 모듈에 가중치 할당
                attn_block.to_qkv.w_a_q, attn_block.to_qkv.w_b_q = w_a_q, w_b_q
                attn_block.to_qkv.w_a_v, attn_block.to_qkv.w_b_v = w_a_v, w_b_v
                attn_block.to_qkv.wnew_a_q, attn_block.to_qkv.wnew_b_q = wnew_a_q, wnew_b_q
                attn_block.to_qkv.wnew_a_v, attn_block.to_qkv.wnew_b_v = wnew_a_v, wnew_b_v
            else:
                # 일반 LoRA 가중치 생성하고 바로 GPU로 보냅니다.
                lora_a_q = nn.Linear(dim, r, bias=False).to(device)
                lora_b_q = nn.Linear(r, inner_dim, bias=False).to(device)
                lora_a_v = nn.Linear(dim, r, bias=False).to(device)
                lora_b_v = nn.Linear(r, inner_dim, bias=False).to(device)

                self.lora_As.extend([lora_a_q, lora_a_v])
                self.lora_Bs.extend([lora_b_q, lora_b_v])

                # _LoRA_qkv_timm 모듈에 가중치 할당
                attn_block.to_qkv.lora_a_q, attn_block.to_qkv.lora_b_q = lora_a_q, lora_b_q
                attn_block.to_qkv.lora_a_v, attn_block.to_qkv.lora_b_v = lora_a_v, lora_b_v

        self.lora_vit = vit_model
        self.reset_parameters()
        self.lora_vit.pos_embedding.requires_grad = False
        self._assert_equivalence_once()
    @property
    def pos_embedding(self):
        return self.lora_vit.pos_embedding

    @property
    def transformer(self):
        return self.lora_vit.transformer

    # --- 범용 프록시: 정의되지 않은 속성은 원본으로 위임 ---
    # def __getattr__(self, name):
    #     # 주의: __getattr__은 *기존 속성을 못 찾았을 때만* 호출됩니다.
    #     try:
    #         return super().__getattribute__(name)
    #     except AttributeError:
    #         return getattr(self.lora_vit, name)

    # (선택) dir 지원: IDE 자동완성/디버깅 편의
    def __dir__(self):
        return sorted(set(list(super().__dir__()) + list(dir(self.lora_vit))))

    @torch.no_grad()
    def _assert_equivalence_once(self, atol: float = 1e-5):
        self.eval()
        for li, layer in enumerate(self.lora_vit.transformer.layers):
            attn = layer[0]
            qkv = attn.to_qkv                             # _LoRA_qkv_timm
            in_dim = qkv.qkv.in_features                  # 원본 qkv의 in_features
            
            # --- 아래 라인을 수정하여 NameError를 해결합니다 ---
            x = torch.randn(2, 3, in_dim, device=next(qkv.parameters()).device, dtype=qkv.qkv.weight.dtype)

            # 원본 경로 출력
            y_ref = qkv.qkv(x)

            # LoRA 경로 출력 (초기 델타=0이면 동일해야 함)
            y_new = qkv(x)

            diff = (y_ref - y_new).abs().max().item()
            if diff > atol:
                # 실패 시 에러 로그 출력 (기존 assert와 동일하게 에러 발생)
                error_msg = f"[LoRA Equivalence Check FAILED] Layer {li}: Max difference is {diff} ( > {atol})"
                print(error_msg)
                raise AssertionError(error_msg)
            else:
                pass

    def reset_parameters(self) -> None:
        if self.online_mode:
            for w_A in self.w_As: nn.init.zeros_(w_A.weight)
            for w_B in self.w_Bs: nn.init.zeros_(w_B.weight)
            for wnew_A in self.wnew_As: nn.init.kaiming_uniform_(wnew_A.weight, a=math.sqrt(5))
            for wnew_B in self.wnew_Bs: nn.init.zeros_(wnew_B.weight)
        else:
            # 일반 LoRA 초기화 (논문 방식)
            for lora_A in self.lora_As: nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
            for lora_B in self.lora_Bs: nn.init.zeros_(lora_B.weight)
    
    def update_and_reset_lora_parameters(self):
        if not self.online_mode:
            print("Warning: update_and_reset_lora_parameters is called in non-online mode.")
            return
        
        with torch.no_grad():
            for i in range(len(self.w_As)):
                self.w_As[i].weight.data += self.wnew_As[i].weight.data
                self.w_Bs[i].weight.data += self.wnew_Bs[i].weight.data
        
        for wnew_A in self.wnew_As: nn.init.kaiming_uniform_(wnew_A.weight, a=math.sqrt(5))
        for wnew_B in self.wnew_Bs: nn.init.zeros_(wnew_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)
