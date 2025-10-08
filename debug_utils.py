"""
액션 디버깅을 위한 유틸리티 함수들
"""
import torch
import numpy as np
from typing import Dict, Any, Optional


class ActionDebugger:
    """액션 디버깅을 위한 클래스"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.debug_logs = []
    
    def log_action_stats(self, 
                        actions: torch.Tensor, 
                        stage: str, 
                        iteration: Optional[int] = None,
                        trajectory_id: Optional[int] = None) -> Dict[str, Any]:
        """
        액션 통계를 로깅하고 반환
        
        Args:
            actions: 액션 텐서
            stage: 디버깅 단계 (예: "CEM_INIT", "GD_UPDATE", "MPC_EXEC")
            iteration: 반복 횟수
            trajectory_id: 궤적 ID
        
        Returns:
            액션 통계 딕셔너리
        """
        stats = {
            "stage": stage,
            "iteration": iteration,
            "trajectory_id": trajectory_id,
            "shape": list(actions.shape),
            "min": actions.min().item(),
            "max": actions.max().item(),
            "mean": actions.mean().item(),
            "std": actions.std().item(),
            "is_all_zero": torch.allclose(actions, torch.zeros_like(actions), atol=1e-6),
            "has_nan": torch.isnan(actions).any(),
            "has_inf": torch.isinf(actions).any(),
            "first_action": actions[0, :3].cpu().numpy().tolist() if actions.numel() > 0 else [],
        }
        
        if self.verbose:
            prefix = f"[{stage}]"
            if iteration is not None:
                prefix += f" Iter {iteration}"
            if trajectory_id is not None:
                prefix += f" Traj {trajectory_id}"
            
            print(f"{prefix} Shape: {stats['shape']}")
            print(f"{prefix} Stats - min: {stats['min']:.4f}, max: {stats['max']:.4f}, mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
            print(f"{prefix} First action: {stats['first_action']}")
            
            if stats['is_all_zero']:
                print(f"{prefix} ⚠️ WARNING: All actions are zero!")
            if stats['has_nan']:
                print(f"{prefix} ⚠️ WARNING: Actions contain NaN values!")
            if stats['has_inf']:
                print(f"{prefix} ⚠️ WARNING: Actions contain Inf values!")
        
        self.debug_logs.append(stats)
        return stats
    
    def log_state_change(self, 
                        initial_state: np.ndarray, 
                        final_state: np.ndarray, 
                        stage: str) -> Dict[str, Any]:
        """
        상태 변화를 로깅하고 반환
        
        Args:
            initial_state: 초기 상태
            final_state: 최종 상태
            stage: 디버깅 단계
        
        Returns:
            상태 변화 통계 딕셔너리
        """
        state_change = np.linalg.norm(final_state - initial_state)
        
        stats = {
            "stage": stage,
            "initial_state": initial_state.tolist(),
            "final_state": final_state.tolist(),
            "change_magnitude": float(state_change),
            "has_change": state_change > 1e-6,
        }
        
        if self.verbose:
            print(f"[{stage}] State change magnitude: {stats['change_magnitude']:.4f}")
            if not stats['has_change']:
                print(f"[{stage}] ⚠️ WARNING: No state changes detected!")
        
        self.debug_logs.append(stats)
        return stats
    
    def generate_summary_report(self) -> str:
        """
        디버깅 로그를 기반으로 요약 리포트 생성
        
        Returns:
            요약 리포트 문자열
        """
        if not self.debug_logs:
            return "No debug logs available."
        
        report = ["=" * 60]
        report.append("ACTION DEBUGGING SUMMARY REPORT")
        report.append("=" * 60)
        
        # 액션 관련 통계
        action_logs = [log for log in self.debug_logs if 'shape' in log]
        if action_logs:
            report.append("\n📊 ACTION STATISTICS:")
            report.append("-" * 30)
            
            for log in action_logs:
                stage = log['stage']
                report.append(f"\n{stage}:")
                report.append(f"  Shape: {log['shape']}")
                report.append(f"  Range: [{log['min']:.4f}, {log['max']:.4f}]")
                report.append(f"  Mean: {log['mean']:.4f}, Std: {log['std']:.4f}")
                
                warnings = []
                if log['is_all_zero']:
                    warnings.append("ALL ZERO")
                if log['has_nan']:
                    warnings.append("HAS NaN")
                if log['has_inf']:
                    warnings.append("HAS Inf")
                
                if warnings:
                    report.append(f"  ⚠️ WARNINGS: {', '.join(warnings)}")
        
        # 상태 변화 관련 통계
        state_logs = [log for log in self.debug_logs if 'change_magnitude' in log]
        if state_logs:
            report.append("\n🔄 STATE CHANGES:")
            report.append("-" * 30)
            
            for log in state_logs:
                stage = log['stage']
                magnitude = log['change_magnitude']
                report.append(f"\n{stage}:")
                report.append(f"  Change magnitude: {magnitude:.4f}")
                if not log['has_change']:
                    report.append("  ⚠️ WARNING: No state changes!")
        
        # 문제점 요약
        report.append("\n🚨 ISSUES SUMMARY:")
        report.append("-" * 30)
        
        issues = []
        for log in self.debug_logs:
            if log.get('is_all_zero', False):
                issues.append(f"{log['stage']}: All actions are zero")
            if log.get('has_nan', False):
                issues.append(f"{log['stage']}: Actions contain NaN")
            if log.get('has_inf', False):
                issues.append(f"{log['stage']}: Actions contain Inf")
            if log.get('change_magnitude', 0) < 1e-6:
                issues.append(f"{log['stage']}: No state changes")
        
        if issues:
            for issue in issues:
                report.append(f"  • {issue}")
        else:
            report.append("  ✅ No major issues detected")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_debug_logs(self, filename: str = "debug_logs.json"):
        """디버깅 로그를 JSON 파일로 저장"""
        import json
        
        with open(filename, 'w') as f:
            json.dump(self.debug_logs, f, indent=2)
        
        print(f"Debug logs saved to {filename}")


# 전역 디버거 인스턴스
debugger = ActionDebugger(verbose=True)


def debug_actions(actions: torch.Tensor, 
                 stage: str, 
                 iteration: Optional[int] = None,
                 trajectory_id: Optional[int] = None) -> Dict[str, Any]:
    """액션 디버깅을 위한 편의 함수"""
    return debugger.log_action_stats(actions, stage, iteration, trajectory_id)


def debug_state_change(initial_state: np.ndarray, 
                      final_state: np.ndarray, 
                      stage: str) -> Dict[str, Any]:
    """상태 변화 디버깅을 위한 편의 함수"""
    return debugger.log_state_change(initial_state, final_state, stage)


def print_debug_summary():
    """디버깅 요약 리포트 출력"""
    print(debugger.generate_summary_report())


def save_debug_logs(filename: str = "debug_logs.json"):
    """디버깅 로그 저장"""
    debugger.save_debug_logs(filename)
