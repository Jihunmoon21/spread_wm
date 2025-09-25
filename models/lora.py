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

class _LoRA_qkv_timm(nn.Module):
    def __init__(self, qkv: nn.Module, online_mode: bool = True):
        super().__init__()
        self.qkv = qkv
        self.online_mode = online_mode
        self.dim = qkv.in_features
        
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

        for layer in vit_model.transformer.layers:
            attn_block = layer[0]
            w_qkv_linear = attn_block.to_qkv
            dim = w_qkv_linear.in_features
            
            # 기존 to_qkv 레이어를 LoRA 기능이 추가된 레이어로 교체
            attn_block.to_qkv = _LoRA_qkv_timm(w_qkv_linear, online_mode=self.online_mode)

            if self.online_mode:
                # 온라인 LoRA 가중치 생성
                w_a_q, w_b_q = nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)
                w_a_v, w_b_v = nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)
                wnew_a_q, wnew_b_q = nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)
                wnew_a_v, wnew_b_v = nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)
                
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
                # 일반 LoRA 가중치 생성 (모두 학습 가능)
                lora_a_q, lora_b_q = nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)
                lora_a_v, lora_b_v = nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)
                
                self.lora_As.extend([lora_a_q, lora_a_v])
                self.lora_Bs.extend([lora_b_q, lora_b_v])

                # _LoRA_qkv_timm 모듈에 가중치 할당
                attn_block.to_qkv.lora_a_q, attn_block.to_qkv.lora_b_q = lora_a_q, lora_b_q
                attn_block.to_qkv.lora_a_v, attn_block.to_qkv.lora_b_v = lora_a_v, lora_b_v

        self.lora_vit = vit_model
        self.reset_parameters()
        self.lora_vit.pos_embedding.requires_grad = True

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
    
# class LoRA_ViT(nn.Module):
#     """Applies low-rank adaptation to a vision transformer.

#     Args:
#         vit_model: a vision transformer model, see base_vit.py
#         r: rank of LoRA
#         num_classes: how many classes the model output, default to the vit model
#         lora_layer: which layer we apply LoRA.

#     Examples::
#         >>> model = ViT('B_16_imagenet1k')
#         >>> lora_model = LoRA_ViT(model, r=4)
#         >>> preds = lora_model(img)
#         >>> print(preds.shape)
#         torch.Size([1, 1000])
#     """

#     def __init__(self, vit_model: ViT, r: int, num_classes: int = 0, lora_layer=None):
#         super(LoRA_ViT, self).__init__()

#         assert r > 0
#         base_vit_dim = vit_model.transformer.blocks[0].attn.proj_q.in_features
#         dim = base_vit_dim
#         if lora_layer:
#             self.lora_layer = lora_layer
#         else:
#             self.lora_layer = list(range(len(vit_model.transformer.blocks)))
#         # create for storage, then we can init them or load weights
#         self.w_As = []  # These are linear layers
#         self.w_Bs = []
#         self.wnew_As = []
#         self.wnew_Bs = []

#         # lets freeze first
#         for param in vit_model.parameters():
#             param.requires_grad = False

#         # Here, we do the surgery
#         for t_layer_i, blk in enumerate(vit_model.transformer.blocks):
#             # If we only want few lora layer instead of all
#             if t_layer_i not in self.lora_layer:
#                 continue
#             w_q_linear = blk.attn.proj_q
#             w_v_linear = blk.attn.proj_v
#             w_a_linear_q = nn.Linear(dim, r, bias=False)
#             w_b_linear_q = nn.Linear(r, dim, bias=False)
#             w_a_linear_v = nn.Linear(dim, r, bias=False)
#             w_b_linear_v = nn.Linear(r, dim, bias=False)
#             # Freeze w_A and w_B parameters
#             for param in w_a_linear_q.parameters(): 
#                 param.requires_grad = False
#             for param in w_a_linear_v.parameters(): 
#                 param.requires_grad = False
#             for param in w_b_linear_q.parameters(): 
#                 param.requires_grad = False
#             for param in w_b_linear_v.parameters(): 
#                 param.requires_grad = False

#             wnew_a_linear_q = nn.Linear(dim, r, bias=False)
#             wnew_b_linear_q = nn.Linear(r, dim, bias=False)
#             wnew_a_linear_v = nn.Linear(dim, r, bias=False)
#             wnew_b_linear_v = nn.Linear(r, dim, bias=False)
#             # Assign custom attributes to wnew_a and wnew_b
#             setattr(wnew_a_linear_q.weight, '_is_wnew_a', True)
#             setattr(wnew_a_linear_v.weight, '_is_wnew_a', True)
#             setattr(wnew_b_linear_q.weight, '_is_wnew_b', True)
#             setattr(wnew_b_linear_v.weight, '_is_wnew_b', True)
#             # Assign w_a and w_b
#             setattr(w_a_linear_q.weight, '_is_w_a', True)
#             setattr(w_a_linear_v.weight, '_is_w_a', True)
#             setattr(w_b_linear_q.weight, '_is_w_b', True)
#             setattr(w_b_linear_v.weight, '_is_w_b', True)

#             self.w_As.append(w_a_linear_q)
#             self.w_Bs.append(w_b_linear_q)
#             self.w_As.append(w_a_linear_v)
#             self.w_Bs.append(w_b_linear_v)
#             self.wnew_As.append(wnew_a_linear_q)
#             self.wnew_Bs.append(wnew_b_linear_q)
#             self.wnew_As.append(wnew_a_linear_v)
#             self.wnew_Bs.append(wnew_b_linear_v)
#             blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q, wnew_a_linear_q, wnew_b_linear_q)
#             blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v, wnew_a_linear_v, wnew_b_linear_v)

#         self.reset_parameters()
#         self.lora_vit = vit_model
#         if num_classes > 0:
#             self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

#     def save_fc_parameters(self, filename: str) -> None:
#         r"""Only safetensors is supported now.

#         pip install safetensor if you do not have one installed yet.
#         """
#         assert filename.endswith(".safetensors")
#         _in = self.lora_vit.fc.in_features
#         _out = self.lora_vit.fc.out_features
#         fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.fc.weight}
#         save_file(fc_tensors, filename)

#     def load_fc_parameters(self, filename: str) -> None:
#         r"""Only safetensors is supported now.

#         pip install safetensor if you do not have one installed yet.
#         """

#         assert filename.endswith(".safetensors")
#         _in = self.lora_vit.fc.in_features
#         _out = self.lora_vit.fc.out_features
#         with safe_open(filename, framework="pt") as f:
#             saved_key = f"fc_{_in}in_{_out}out"
#             try:
#                 saved_tensor = f.get_tensor(saved_key)
#                 self.lora_vit.fc.weight = Parameter(saved_tensor)
#             except ValueError:
#                 print("this fc weight is not for this model")

#     def save_lora_parameters(self, filename: str) -> None:
#         r"""Only safetensors is supported now.

#         pip install safetensor if you do not have one installed yet.
#         """

#         assert filename.endswith(".safetensors")

#         num_layer = len(self.w_As)  # actually, it is half
#         a_tensors = {f"w_a_{i:03d}": (self.w_As[i].weight + self.wnew_As[i].weight) for i in range(num_layer)}
#         b_tensors = {f"w_b_{i:03d}": (self.w_Bs[i].weight + self.wnew_Bs[i].weight) for i in range(num_layer)}

        
#         _in = self.lora_vit.fc.in_features
#         _out = self.lora_vit.fc.out_features
#         fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.fc.weight}
        
#         merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
#         save_file(merged_dict, filename)

#     def load_lora_parameters(self, filename: str) -> None:
#         r"""Only safetensors is supported now.

#         pip install safetensor if you do not have one installed yet.
#         """

#         assert filename.endswith(".safetensors")

#         with safe_open(filename, framework="pt") as f:
#             for i, w_A_linear in enumerate(self.w_As):
#                 saved_key = f"w_a_{i:03d}"
#                 saved_tensor = f.get_tensor(saved_key)
#                 w_A_linear.weight = Parameter(saved_tensor)

#             for i, w_B_linear in enumerate(self.w_Bs):
#                 saved_key = f"w_b_{i:03d}"
#                 saved_tensor = f.get_tensor(saved_key)
#                 w_B_linear.weight = Parameter(saved_tensor)
                
#             _in = self.lora_vit.fc.in_features
#             _out = self.lora_vit.fc.out_features
#             saved_key = f"fc_{_in}in_{_out}out"
#             try:
#                 saved_tensor = f.get_tensor(saved_key)
#                 self.lora_vit.fc.weight = Parameter(saved_tensor)
#             except ValueError:
#                 print("this fc weight is not for this model")

#     def reset_parameters(self) -> None:
#         for w_A in self.w_As:
#             nn.init.zeros_(w_A.weight)
#         for w_B in self.w_Bs:
#             nn.init.zeros_(w_B.weight)
#         for wnew_A in self.wnew_As:
#             nn.init.kaiming_uniform_(wnew_A.weight, a=math.sqrt(5))
#         for wnew_B in self.wnew_Bs:
#             nn.init.zeros_(wnew_B.weight)


#     def forward(self, x: Tensor, use_new: bool = True) -> Tensor:
#         if use_new:
#             return self.lora_vit(x)
    
# class _LoRA_qkv_timm_x(nn.Module):
#     """In timm it is implemented as
#     self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

#     B, N, C = x.shape
#     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#     q, k, v = qkv.unbind(0)

#     """

#     def __init__(
#         self,
#         qkv: nn.Module,
#         linear_a_qs,
#         linear_b_qs,
#         linear_a_vs,
#         linear_b_vs,
#         linear_new_a_qs,
#         linear_new_b_qs,
#         linear_new_a_vs,
#         linear_new_b_vs,
#     ):
#         super().__init__()
#         self.qkv = qkv
#         for i in range(len(linear_a_qs)):
#             setattr(self, f'linear_a_q_{i}', linear_a_qs[i])
#             setattr(self, f'linear_b_q_{i}', linear_b_qs[i])
#             setattr(self, f'linear_a_v_{i}', linear_a_vs[i])
#             setattr(self, f'linear_b_v_{i}', linear_b_vs[i])
#             setattr(self, f'linear_new_a_q_{i}', linear_new_a_qs[i])
#             setattr(self, f'linear_new_b_q_{i}', linear_new_b_qs[i])
#             setattr(self, f'linear_new_a_v_{i}', linear_new_a_vs[i])
#             setattr(self, f'linear_new_b_v_{i}', linear_new_b_vs[i])
#         self.dim = qkv.in_features
#         self.w_identity = torch.eye(qkv.in_features)
#         self.lora_id = 0
    
#     def change_lora(self, num):
#         self.lora_id = num

#     def forward(self, x):
#         qkv = self.qkv(x)  # B,N,3*org_C
#         linear_a_q = getattr(self, f'linear_a_q_{self.lora_id}')
#         linear_b_q = getattr(self, f'linear_b_q_{self.lora_id}')
#         linear_a_v = getattr(self, f'linear_a_v_{self.lora_id}')
#         linear_b_v = getattr(self, f'linear_b_v_{self.lora_id}')
#         linear_new_a_q = getattr(self, f'linear_new_a_q_{self.lora_id}')
#         linear_new_b_q = getattr(self, f'linear_new_b_q_{self.lora_id}')
#         linear_new_a_v = getattr(self, f'linear_new_a_v_{self.lora_id}')
#         linear_new_b_v = getattr(self, f'linear_new_b_v_{self.lora_id}')
#         new_q = linear_b_q(linear_a_q(x))
#         new_v = linear_b_v(linear_a_v(x))
#         new_q += linear_new_b_q(linear_new_a_q(x))
#         new_v += linear_new_b_v(linear_new_a_v(x))
#         qkv[:, :, : self.dim] += new_q
#         qkv[:, :, -self.dim :] += new_v
#         return qkv
        
# class LoRA_ViT_timm_x(nn.Module):
#     def __init__(self, vit_model: timm_ViT, lora_files: list, lora_layer=None):
#         super(LoRA_ViT_timm_x, self).__init__()

#         self.lora_layer = list(range(len(vit_model.blocks)))

#         self.w_As = []  # These are linear layers
#         self.w_Bs = []
#         self.wnew_As = []
#         self.wnew_Bs = []
        
#         self.fc_loras = []
#         self.num_classes = []

#         # lets freeze first
#         for param in vit_model.parameters():
#             param.requires_grad = False
        
#         self.lora_vit = vit_model

#         # Here, we do the surgery
#         for t_layer_i, blk in enumerate(vit_model.blocks):
#             # If we only want few lora layer instead of all
#             if t_layer_i not in self.lora_layer:
#                 continue
#             w_qkv_linear = blk.attn.qkv
#             self.dim = w_qkv_linear.in_features
            
#             w_a_linear_qs = []
#             w_b_linear_qs = []
#             w_a_linear_vs = []
#             w_b_linear_vs = []
#             wnew_a_linear_qs = []
#             wnew_b_linear_qs = []
#             wnew_a_linear_vs = []
#             wnew_b_linear_vs = []
#             for file_path in lora_files:
#                 with safe_open(file_path, framework="pt") as f:
#                     melo_info = file_path.split("/")[-1].split("_")
                    
#                     r = int(melo_info[3])
                    
#                     w_a_linear_q = nn.Linear(self.dim, r, bias=False)
#                     w_b_linear_q = nn.Linear(r, self.dim, bias=False)
#                     w_a_linear_v = nn.Linear(self.dim, r, bias=False)
#                     w_b_linear_v = nn.Linear(r, self.dim, bias=False)
#                     wnew_a_linear_q = nn.Linear(self.dim, r, bias=False)
#                     wnew_b_linear_q = nn.Linear(r, self.dim, bias=False)
#                     wnew_a_linear_v = nn.Linear(self.dim, r, bias=False)
#                     wnew_b_linear_v = nn.Linear(r, self.dim, bias=False)
                    
#                     w_a_linear_q.weight = Parameter(f.get_tensor(f"w_a_{t_layer_i * 2:03d}"))
#                     w_b_linear_q.weight = Parameter(f.get_tensor(f"w_b_{t_layer_i * 2:03d}"))
#                     w_a_linear_v.weight = Parameter(f.get_tensor(f"w_a_{t_layer_i * 2 + 1:03d}"))
#                     w_b_linear_v.weight = Parameter(f.get_tensor(f"w_b_{t_layer_i * 2 + 1:03d}"))
#                     # Freeze w_A and w_B parameters
#                     w_a_linear_q.weight.requires_grad = False
#                     w_b_linear_q.weight.requires_grad = False
#                     w_a_linear_v.weight.requires_grad = False
#                     w_b_linear_v.weight.requires_grad = False
#                     wnew_a_linear_q.weight = Parameter(f.get_tensor(f"wnew_a_{t_layer_i * 2:03d}"))
#                     wnew_b_linear_q.weight = Parameter(f.get_tensor(f"wnew_b_{t_layer_i * 2:03d}"))
#                     wnew_a_linear_v.weight = Parameter(f.get_tensor(f"wnew_a_{t_layer_i * 2 + 1:03d}"))
#                     wnew_b_linear_v.weight = Parameter(f.get_tensor(f"wnew_b_{t_layer_i * 2 + 1:03d}"))
#                     # Assign custom attributes to wnew_a and wnew_b
#                     setattr(wnew_a_linear_q.weight, '_is_wnew_a', True)
#                     setattr(wnew_a_linear_v.weight, '_is_wnew_a', True)
#                     setattr(wnew_b_linear_q.weight, '_is_wnew_b', True)
#                     setattr(wnew_b_linear_v.weight, '_is_wnew_b', True)
#                     # Assign custom attributes to w_a and w_b
#                     setattr(w_a_linear_q.weight, '_is_w_a', True)
#                     setattr(w_a_linear_v.weight, '_is_w_a', True)
#                     setattr(w_b_linear_q.weight, '_is_w_b', True)
#                     setattr(w_b_linear_v.weight, '_is_w_b', True)
                    
#                     w_a_linear_qs.append(w_a_linear_q)
#                     w_b_linear_qs.append(w_b_linear_q)
#                     w_a_linear_vs.append(w_a_linear_v)
#                     w_b_linear_vs.append(w_b_linear_v)
#                     wnew_a_linear_qs.append(wnew_a_linear_q)
#                     wnew_b_linear_qs.append(wnew_b_linear_q)
#                     wnew_a_linear_vs.append(wnew_a_linear_v)
#                     wnew_b_linear_vs.append(wnew_b_linear_v)
                    
#                     _in = self.lora_vit.head.in_features
#                     _out = int(melo_info[4])
#                     self.num_classes.append(_out)
#                     self.fc_loras.append(f.get_tensor(f"fc_{_in}in_{_out}out"))
            
#             blk.attn.qkv = _LoRA_qkv_timm_x(
#                 w_qkv_linear,
#                 w_a_linear_qs,
#                 w_b_linear_qs,
#                 w_a_linear_vs,
#                 w_b_linear_vs,
#                 wnew_a_linear_qs,
#                 wnew_b_linear_qs,
#                 wnew_a_linear_vs,
#                 wnew_b_linear_vs,
#             )

#         for file_path in lora_files:
#             with safe_open(file_path, framework="pt") as f:
#                 for key in f.keys():
#                     if 'fc_' in key:
#                         self.fc_loras.append(f.get_tensor(key))
#                         break
    
#     def swith_lora(self, idx:int):
#         for t_layer_i, blk in enumerate(self.lora_vit.blocks):
#             blk.attn.qkv.change_lora(idx)
#         self.lora_vit.reset_classifier(num_classes=self.num_classes[idx])
#         self.lora_vit.head.weight = Parameter(self.fc_loras[idx])

#     def forward(self, x: Tensor) -> Tensor:
#         return self.lora_vit(x)

# % --------------------------SwinTransformer LoRA -----------------------------------------------------------%

# class LoRA_Swin_timm(nn.Module):
#     def __init__(self, swin_model: timm_swin, r: int = 4, num_classes: int = 0, lora_layer=None):
#         super(LoRA_Swin_timm, self).__init__()

#         assert r > 0

#         self.w_As = []  # These are linear layers
#         self.w_Bs = []
#         self.wnew_As = []
#         self.wnew_Bs = []

#         # lets freeze first
#         for param in swin_model.parameters():
#             param.requires_grad = False

#         # Here, we do the surgery
#         for _, stage in enumerate(swin_model.layers):
#             for _, blk in enumerate(stage.blocks):
#                 w_qkv_linear = blk.attn.qkv
#                 self.dim = w_qkv_linear.in_features
#                 w_a_linear_q = nn.Linear(self.dim, r, bias=False)
#                 w_b_linear_q = nn.Linear(r, self.dim, bias=False)
#                 w_a_linear_v = nn.Linear(self.dim, r, bias=False)
#                 w_b_linear_v = nn.Linear(r, self.dim, bias=False)
#                 # Freeze w_A and w_B parameters
#                 for param in w_a_linear_q.parameters(): 
#                     param.requires_grad = False
#                 for param in w_a_linear_v.parameters(): 
#                     param.requires_grad = False
#                 for param in w_b_linear_q.parameters(): 
#                     param.requires_grad = False
#                 for param in w_b_linear_v.parameters(): 
#                     param.requires_grad = False
#                 # Add: new set of lora
#                 wnew_a_linear_q = nn.Linear(self.dim, r, bias=False)
#                 wnew_b_linear_q = nn.Linear(r, self.dim, bias=False)
#                 wnew_a_linear_v = nn.Linear(self.dim, r, bias=False)
#                 wnew_b_linear_v = nn.Linear(r, self.dim, bias=False)
#                 # Assign custom attributes to wnew_a and wnew_b
#                 setattr(wnew_a_linear_q.weight, '_is_wnew_a', True)
#                 setattr(wnew_a_linear_v.weight, '_is_wnew_a', True)
#                 setattr(wnew_b_linear_q.weight, '_is_wnew_b', True)
#                 setattr(wnew_b_linear_v.weight, '_is_wnew_b', True)
#                 # Assign custom attributes to w_a and w_b
#                 setattr(w_a_linear_q.weight, '_is_w_a', True)
#                 setattr(w_a_linear_v.weight, '_is_w_a', True)
#                 setattr(w_b_linear_q.weight, '_is_w_b', True)
#                 setattr(w_b_linear_v.weight, '_is_w_b', True)
#                 self.w_As.append(w_a_linear_q)
#                 self.w_Bs.append(w_b_linear_q)
#                 self.w_As.append(w_a_linear_v)
#                 self.w_Bs.append(w_b_linear_v)
#                 self.wnew_As.append(wnew_a_linear_q)
#                 self.wnew_Bs.append(wnew_b_linear_q)
#                 self.wnew_As.append(wnew_a_linear_v)
#                 self.wnew_Bs.append(wnew_b_linear_v)
#                 blk.attn.qkv = _LoRA_qkv_timm(
#                     w_qkv_linear,
#                     w_a_linear_q,
#                     w_b_linear_q,
#                     w_a_linear_v,
#                     w_b_linear_v,
#                     wnew_a_linear_q,
#                     wnew_b_linear_q,
#                     wnew_a_linear_v,
#                     wnew_b_linear_v,
#                 )
#         self.reset_parameters()
#         self.lora_swin = swin_model
#         self.proj_3d = nn.Linear(num_classes * 30, num_classes)
#         if num_classes > 0:
#             self.lora_swin.reset_classifier(num_classes=num_classes)

#     def save_lora_parameters(self, filename: str) -> None:
#         r"""Only safetensors is supported now.

#         pip install safetensor if you do not have one installed yet.
        
#         save both lora and fc parameters.
#         """

#         assert filename.endswith(".safetensors")

#         num_layer = len(self.w_As)  # actually, it is half
#         a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
#         b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
#         _in = self.lora_swin.head.in_features
#         _out = self.lora_swin.head.fc.weight.shape[0]
#         fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_swin.head.fc.weight}
        
#         merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
#         save_file(merged_dict, filename)

#     def load_lora_parameters(self, filename: str) -> None:
#         r"""Only safetensors is supported now.

#         pip install safetensor if you do not have one installed yet.\
            
#         load both lora and fc parameters.
#         """

#         assert filename.endswith(".safetensors")

#         with safe_open(filename, framework="pt") as f:
#             for i, w_A_linear in enumerate(self.w_As):
#                 saved_key = f"w_a_{i:03d}"
#                 saved_tensor = f.get_tensor(saved_key)
#                 w_A_linear.weight = Parameter(saved_tensor)

#             for i, w_B_linear in enumerate(self.w_Bs):
#                 saved_key = f"w_b_{i:03d}"
#                 saved_tensor = f.get_tensor(saved_key)
#                 w_B_linear.weight = Parameter(saved_tensor)
                
#             _in = self.lora_swin.head.in_features
#             _out = self.lora_swin.head.fc.weight.shape[0]
#             saved_key = f"fc_{_in}in_{_out}out"
#             try:
#                 saved_tensor = f.get_tensor(saved_key)
#                 self.lora_swin.head.fc.weight = Parameter(saved_tensor)
#             except ValueError:
#                 print("this fc weight is not for this model")

#     def update_and_reset_lora_parameters(self):
#         # Update w_A and w_B weights and reset wnew_A and wnew_B weights
#         for i in range(len(self.w_As)):
#             self.w_As[i].weight.data += self.wnew_As[i].weight.data
#             self.w_Bs[i].weight.data += self.wnew_Bs[i].weight.data
#             self.reset_lora_new_parameters(i)

#     def reset_lora_new_parameters(self, index):
#         # Reset the wnew_A and wnew_B weights to zeros or initial state
#         # nn.init.kaiming_uniform_(self.wnew_As[index].weight, a=math.sqrt(5))
#         nn.init.zeros_(self.wnew_As[index].weight)
#         nn.init.zeros_(self.wnew_Bs[index].weight)

#     def reset_parameters(self) -> None:
#         for w_A in self.w_As:
#             nn.init.zeros_(w_A.weight)
#         for w_B in self.w_Bs:
#             nn.init.zeros_(w_B.weight)
#         for wnew_A in self.wnew_As:
#             nn.init.kaiming_uniform_(wnew_A.weight, a=math.sqrt(5))
#         for wnew_B in self.wnew_Bs:
#             nn.init.zeros_(wnew_B.weight)

#     def forward(self, x: Tensor, use_new: bool = True) -> Tensor:
#         if use_new:
#             return self.lora_swin(x)