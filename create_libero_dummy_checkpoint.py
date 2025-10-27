#!/usr/bin/env python3
"""
Libero용 Dummy 체크포인트 생성 스크립트
Planning 테스트를 위한 최소한의 모델 체크포인트를 생성합니다.
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.dummy import DummyModel, DummyRepeatActionEncoder
from models.vit import ViTPredictor
from models.vqvae import VQVAE

# 디렉토리 생성
output_dir = Path("/home/jihoonmun/spread/outputs/libero_dummy/checkpoints")
output_dir.mkdir(parents=True, exist_ok=True)

print("Creating dummy checkpoint for libero planning test...")
print("Instantiating model components...")

# 실제 모델 인스턴스 생성
try:
    encoder = DummyModel(emb_dim=384)
    print("  ✓ Encoder (DummyModel) instantiated")
except Exception as e:
    print(f"  ✗ Encoder error: {e}")
    encoder = None

try:
    action_encoder = DummyRepeatActionEncoder(in_chans=35, emb_dim=10)
    print("  ✓ Action encoder (DummyRepeatActionEncoder) instantiated")
except Exception as e:
    print(f"  ✗ Action encoder error: {e}")
    action_encoder = None

try:
    proprio_encoder = DummyModel(emb_dim=10)
    print("  ✓ Proprio encoder (DummyModel) instantiated")
except Exception as e:
    print(f"  ✗ Proprio encoder error: {e}")
    proprio_encoder = None

try:
    # ViTPredictor 필수 인자 계산
    # img_size=256, patch_size=16 (DinoV2 기본값)
    # num_patches = (256 // 16) ** 2 = 256
    # num_frames = num_hist + num_pred = 3 + 1 = 4
    # dim = encoder output dim = 384
    predictor = ViTPredictor(
        num_patches=256,
        num_frames=4,
        dim=384,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0,
        pool='mean'
    )
    print("  ✓ Predictor (ViTPredictor) instantiated")
except Exception as e:
    print(f"  ✗ Predictor error: {e}")
    predictor = None

try:
    decoder = VQVAE(
        channel=384,
        n_embed=2048,
        n_res_block=4,
        n_res_channel=128,
        quantize=False
    )
    print("  ✓ Decoder (VQVAE) instantiated")
except Exception as e:
    print(f"  ✗ Decoder error: {e}")
    decoder = None

# Checkpoint 생성
checkpoint = {
    'encoder': encoder,
    'action_encoder': action_encoder,
    'proprio_encoder': proprio_encoder,
    'predictor': predictor,
    'decoder': decoder,
    'epoch': 0,
}

# 저장
checkpoint_path = output_dir / "model_latest.pth"
torch.save(checkpoint, checkpoint_path)
print(f"\n✅ Dummy checkpoint saved to {checkpoint_path}")
print(f"   Size: {checkpoint_path.stat().st_size / 1024:.1f} KB")
print("\nNote: This is a dummy checkpoint for testing planning infrastructure only.")
print("It will not produce meaningful planning results.")

