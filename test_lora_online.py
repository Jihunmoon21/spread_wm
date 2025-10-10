#!/usr/bin/env python3
"""
LoRA 온라인 학습 테스트 스크립트
"""

import torch
import torch.nn as nn
import numpy as np
from models.lora import LoRA_ViT_spread
from models.vit import ViTPredictor

def test_lora_online_learning():
    """LoRA 온라인 학습이 정상적으로 작동하는지 테스트"""
    
    print("=== LoRA Online Learning Test ===")
    
    # 1. 기본 ViT 모델 생성
    print("1. Creating base ViT model...")
    vit_model = ViTPredictor(
        image_size=224,
        patch_size=16,
        num_classes=384,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
    )
    
    # 2. LoRA 래퍼 적용
    print("2. Applying LoRA wrapper...")
    lora_model = LoRA_ViT_spread(vit_model=vit_model, r=4, online_mode=True)
    
    # 3. 파라미터 상태 확인
    print("3. Checking parameter status...")
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.4f}%")
    
    # 4. LoRA 파라미터 상세 정보
    if hasattr(lora_model, 'wnew_As'):
        lora_params = sum(p.numel() for p in lora_model.wnew_As + lora_model.wnew_Bs)
        print(f"LoRA parameters: {lora_params:,}")
    
    # 5. 더미 데이터로 forward pass 테스트
    print("4. Testing forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = lora_model(dummy_input)
        print(f"Output shape: {output.shape}")
    
    # 6. 그래디언트 테스트
    print("5. Testing gradient computation...")
    dummy_input.requires_grad = True
    output = lora_model(dummy_input)
    
    # 손실 계산
    target = torch.randn_like(output)
    loss = nn.MSELoss()(output, target)
    
    # 역전파
    loss.backward()
    
    # 그래디언트 확인
    grad_norm = 0
    trainable_count = 0
    for param in lora_model.parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
            trainable_count += 1
    
    grad_norm = grad_norm ** 0.5
    print(f"Gradient norm: {grad_norm:.6f}")
    print(f"Parameters with gradients: {trainable_count}")
    
    # 7. 파라미터 업데이트 테스트
    print("6. Testing parameter update...")
    optimizer = torch.optim.Adam([p for p in lora_model.parameters() if p.requires_grad], lr=1e-4)
    
    # 초기 파라미터 저장
    initial_params = []
    for p in lora_model.wnew_As + lora_model.wnew_Bs:
        if p.requires_grad:
            initial_params.append(p.clone())
    
    # 업데이트
    optimizer.zero_grad()
    output = lora_model(dummy_input)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()
    
    # 파라미터 변화량 확인
    param_changes = []
    current_idx = 0
    for p in lora_model.wnew_As + lora_model.wnew_Bs:
        if p.requires_grad:
            change = torch.norm(p - initial_params[current_idx]).item()
            param_changes.append(change)
            current_idx += 1
    
    avg_change = sum(param_changes) / len(param_changes)
    print(f"Average parameter change: {avg_change:.8f}")
    print(f"Loss: {loss.item():.6f}")
    
    print("=== Test Complete ===")
    
    # 결과 요약
    if grad_norm > 0 and avg_change > 0:
        print("✅ LoRA online learning is working correctly!")
        print(f"   - Gradients computed: {grad_norm:.6f}")
        print(f"   - Parameters updated: {avg_change:.8f}")
        print(f"   - Trainable parameters: {trainable_params:,}")
    else:
        print("❌ LoRA online learning has issues!")
        print(f"   - Gradients: {grad_norm}")
        print(f"   - Parameter changes: {avg_change}")

if __name__ == "__main__":
    test_lora_online_learning()
