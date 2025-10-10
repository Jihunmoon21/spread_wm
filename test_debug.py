#!/usr/bin/env python3
"""
액션 디버깅 유틸리티 테스트 스크립트
"""
import torch
import numpy as np
from debug_utils import debug_actions, debug_state_change, print_debug_summary, save_debug_logs

def test_action_debugging():
    """액션 디버깅 기능 테스트"""
    print("=" * 60)
    print("액션 디버깅 유틸리티 테스트")
    print("=" * 60)
    
    # 정상적인 액션 테스트
    normal_actions = torch.randn(2, 5, 4) * 0.5
    debug_actions(normal_actions, "NORMAL_ACTIONS")
    
    # 모든 값이 0인 액션 테스트
    zero_actions = torch.zeros(2, 5, 4)
    debug_actions(zero_actions, "ZERO_ACTIONS")
    
    # NaN이 포함된 액션 테스트
    nan_actions = torch.randn(2, 5, 4)
    nan_actions[0, 0, 0] = float('nan')
    debug_actions(nan_actions, "NAN_ACTIONS")
    
    # Inf가 포함된 액션 테스트
    inf_actions = torch.randn(2, 5, 4)
    inf_actions[0, 0, 0] = float('inf')
    debug_actions(inf_actions, "INF_ACTIONS")
    
    # 상태 변화 테스트
    initial_state = np.array([1.0, 2.0, 3.0])
    final_state = np.array([1.5, 2.5, 3.5])
    debug_state_change(initial_state, final_state, "STATE_CHANGE_TEST")
    
    # 상태 변화가 없는 경우 테스트
    no_change_state = np.array([1.0, 2.0, 3.0])
    debug_state_change(initial_state, no_change_state, "NO_STATE_CHANGE")
    
    print("\n" + "=" * 60)
    print("디버깅 요약 리포트:")
    print_debug_summary()
    
    # 디버깅 로그 저장
    save_debug_logs("test_debug_logs.json")
    
    print("\n테스트 완료!")

if __name__ == "__main__":
    test_action_debugging()

