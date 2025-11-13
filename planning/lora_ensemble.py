import torch
import torch.nn as nn
import math
import time
import os
import gzip
import pickle
import io
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict, deque
from contextlib import redirect_stdout
from utils import move_to_device


class LoRAEnsembleManager:
    """
    LoRA 앙상블을 관리하는 핵심 클래스
    
    기능:
    - 여러 LoRA 모델을 앙상블로 관리
    - 메모리 효율적인 저장/로드
    - 앙상블 멤버 성능 추적
    - LoRA 가중치 통합
    """
    
    def __init__(self, base_model, max_ensemble_size: int = 11, 
                 cache_dir: str = "./lora_cache", max_memory_mb: int = 200):
        self.base_model = base_model
        self.max_ensemble_size = max_ensemble_size
        self.cache_dir = cache_dir
        self.max_memory_mb = max_memory_mb
        
        # 앙상블 멤버 관리
        self.ensemble_members = OrderedDict()  # {task_id: member_info}
        self.current_task_id = 0
        self.memory_usage = 0
        
        # 성능 추적
        self.performance_history = {}  # {task_id: [performance_scores]}
        self.access_frequency = {}   # {task_id: access_count}
        
        # 캐시 디렉토리 생성
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"LoRAEnsembleManager initialized:")
        print(f"  - Max ensemble size: {max_ensemble_size}")
        print(f"  - Cache directory: {cache_dir}")
        print(f"  - Max memory usage: {max_memory_mb}MB")
    
    def add_ensemble_member(self, task_id: int, lora_weights: Dict, 
                           performance: Dict, metadata: Optional[Dict] = None) -> bool:
        """
        새로운 앙상블 멤버 추가
        
        Args:
            task_id: 태스크 ID
            lora_weights: LoRA 가중치 딕셔너리
            performance: 성능 지표 {'loss': float, 'accuracy': float, ...}
            metadata: 추가 메타데이터
            
        Returns:
            bool: 추가 성공 여부
        """
        try:
            if task_id in self.ensemble_members:
                print(f"ℹ️  Ensemble member for Task {task_id} already exists. Skipping update to preserve original weights.")
                return False

            # 앙상블 크기 제한 확인
            if len(self.ensemble_members) >= self.max_ensemble_size:
                self._remove_oldest_member()
            
            # 메모리 사용량 확인
            lora_size_mb = self._calculate_lora_size(lora_weights)
            if self.memory_usage + lora_size_mb > self.max_memory_mb:
                self._cleanup_memory()
            
            # 앙상블 멤버 정보 생성
            member_info = {
                'task_id': task_id,
                'lora_weights': lora_weights,
                'performance': performance,
                'metadata': metadata or {},
                'created_at': time.time(),
                'last_accessed': time.time(),
                'access_count': 0,
                'size_mb': lora_size_mb
            }
            
            # 안전한 깊은 복사로 저장 (참조 오염 방지)
            safe_weights = {}
            for layer_key, layer_vals in lora_weights.items():
                if isinstance(layer_vals, dict):
                    w_A = layer_vals.get('w_A', None)
                    w_B = layer_vals.get('w_B', None)
                    if w_A is not None and w_B is not None:
                        safe_weights[layer_key] = {
                            'w_A': w_A.clone().detach().cpu(),
                            'w_B': w_B.clone().detach().cpu(),
                        }
            member_info['lora_weights'] = safe_weights

            # 앙상블에 추가
            self.ensemble_members[task_id] = member_info
            self.memory_usage += lora_size_mb
            
            # 성능 히스토리 업데이트
            if task_id not in self.performance_history:
                self.performance_history[task_id] = []
            self.performance_history[task_id].append(performance)
            
            # 접근 빈도 초기화
            self.access_frequency[task_id] = 0
            
            print(f"✅ Added ensemble member: Task {task_id}")
            print(f"   - Performance: {performance}")
            print(f"   - Size: {lora_size_mb:.2f}MB")
            print(f"   - Total members: {len(self.ensemble_members)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to add ensemble member: {e}")
            return False
    
    def get_best_member(self, input_data: torch.Tensor, 
                       metric: str = 'loss') -> Optional[Dict]:
        """
        입력 데이터에 대해 가장 좋은 성능의 앙상블 멤버 선택
        
        Args:
            input_data: 입력 데이터
            metric: 성능 지표 ('loss', 'accuracy', etc.)
            
        Returns:
            Dict: 최적의 앙상블 멤버 정보
        """
        if not self.ensemble_members:
            print("⚠️  No ensemble members available")
            return None
        
        best_member = None
        best_score = float('inf') if metric == 'loss' else float('-inf')
        
        for task_id, member_info in self.ensemble_members.items():
            # 접근 빈도 업데이트
            self.access_frequency[task_id] += 1
            member_info['last_accessed'] = time.time()
            member_info['access_count'] += 1
            
            # 성능 점수 계산
            performance = member_info['performance']
            score = performance.get(metric, float('inf'))
            
            # 최적 성능 확인
            is_better = (score < best_score) if metric == 'loss' else (score > best_score)
            if is_better:
                best_score = score
                best_member = member_info
        
        if best_member:
            print(f"🎯 Selected best member: Task {best_member['task_id']} "
                  f"(score: {best_score:.6f})")
        
        return best_member
    
    def save_ensemble_to_disk(self, save_path: str) -> bool:
        """
        앙상블을 디스크에 저장
        
        Args:
            save_path: 저장 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            ensemble_data = {
                'ensemble_members': dict(self.ensemble_members),
                'performance_history': self.performance_history,
                'access_frequency': self.access_frequency,
                'current_task_id': self.current_task_id,
                'metadata': {
                    'max_ensemble_size': self.max_ensemble_size,
                    'cache_dir': self.cache_dir,
                    'max_memory_mb': self.max_memory_mb,
                    'saved_at': time.time()
                }
            }
            
            # 압축하여 저장
            compressed_data = gzip.compress(pickle.dumps(ensemble_data))
            
            with open(save_path, 'wb') as f:
                f.write(compressed_data)
            
            print(f"💾 Ensemble saved to: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save ensemble: {e}")
            return False
    
    def load_ensemble_from_disk(self, load_path: str) -> bool:
        """
        디스크에서 앙상블 로드
        
        Args:
            load_path: 로드 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            with open(load_path, 'rb') as f:
                compressed_data = f.read()
            
            ensemble_data = pickle.loads(gzip.decompress(compressed_data))
            
            # 앙상블 데이터 복원
            self.ensemble_members = OrderedDict(ensemble_data['ensemble_members'])
            self.performance_history = ensemble_data['performance_history']
            self.access_frequency = ensemble_data['access_frequency']
            self.current_task_id = ensemble_data['current_task_id']
            
            # 메모리 사용량 재계산
            self.memory_usage = sum(
                member['size_mb'] for member in self.ensemble_members.values()
            )
            
            print(f"📂 Ensemble loaded from: {load_path}")
            print(f"   - Members: {len(self.ensemble_members)}")
            print(f"   - Memory usage: {self.memory_usage:.2f}MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load ensemble: {e}")
            return False
    
    def get_ensemble_info(self) -> Dict:
        """
        앙상블 정보 반환
        
        Returns:
            Dict: 앙상블 상태 정보
        """
        return {
            'member_count': len(self.ensemble_members),
            'max_ensemble_size': self.max_ensemble_size,
            'memory_usage_mb': self.memory_usage,
            'max_memory_mb': self.max_memory_mb,
            'current_task_id': self.current_task_id,
            'members': list(self.ensemble_members.keys()),
            'cache_dir': self.cache_dir
        }
    
    def _calculate_lora_size(self, lora_weights: Dict) -> float:
        """LoRA 가중치 크기 계산 (MB 단위)"""
        total_params = 0
        for layer_name, weights in lora_weights.items():
            if isinstance(weights, torch.Tensor):
                total_params += weights.numel()
            elif isinstance(weights, dict):
                for sub_weights in weights.values():
                    if isinstance(sub_weights, torch.Tensor):
                        total_params += sub_weights.numel()
        
        # float32 기준으로 계산
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb
    
    def _remove_oldest_member(self):
        """가장 오래된 앙상블 멤버 제거"""
        if not self.ensemble_members:
            return
        
        # 가장 오래된 멤버 찾기
        oldest_task_id = min(self.ensemble_members.keys(), 
                           key=lambda x: self.ensemble_members[x]['created_at'])
        
        oldest_member = self.ensemble_members[oldest_task_id]
        self.memory_usage -= oldest_member['size_mb']
        
        # 디스크에 저장 후 메모리에서 제거
        self._save_member_to_disk(oldest_task_id, oldest_member)
        del self.ensemble_members[oldest_task_id]
        
        print(f"🗑️  Removed oldest member: Task {oldest_task_id}")
    
    def _cleanup_memory(self):
        """메모리 정리"""
        if self.memory_usage <= self.max_memory_mb:
            return
        
        # 접근 빈도가 낮은 멤버들을 디스크로 이동
        sorted_members = sorted(
            self.ensemble_members.items(),
            key=lambda x: x[1]['access_count']
        )
        
        for task_id, member_info in sorted_members:
            if self.memory_usage <= self.max_memory_mb * 0.8:  # 80%까지 정리
                break
            
            self._save_member_to_disk(task_id, member_info)
            self.memory_usage -= member_info['size_mb']
            print(f"💾 Moved to disk: Task {task_id}")
    
    def _save_member_to_disk(self, task_id: int, member_info: Dict):
        """멤버를 디스크에 저장 (이미 존재하면 건너뜀)"""
        save_path = os.path.join(self.cache_dir, f"lora_task_{task_id}.pth")
        
        # 파일이 이미 존재하면 덮어쓰지 않음
        if os.path.exists(save_path):
            print(f"ℹ️  LoRA file for Task {task_id} already exists at {save_path}. Skipping save to preserve original weights.")
            return
        
        # LoRA 가중치만 저장 (메타데이터는 별도 관리)
        torch.save(member_info['lora_weights'], save_path)
        print(f"💾 Saved LoRA member for Task {task_id} to {save_path}")
    


class EnsembleOnlineLora:
    """
    앙상블 기반 Online-LoRA 학습 시스템
    
    기존 OnlineLora를 내부적으로 사용하면서
    앙상블 기반의 지능적인 LoRA 적층을 수행합니다.
    """
    
    def __init__(self, workspace):
        """
        앙상블 Online-LoRA 초기화
        
        Args:
            workspace: PlanWorkspace 인스턴스
        """
        self.workspace = workspace
        self.wm = workspace.wm
        # lora.ensemble_cfg 또는 lora 루트에서 읽기
        self.cfg = workspace.cfg_dict.get("lora", {}).get("ensemble_cfg", workspace.cfg_dict.get("lora", {}))
        self.device = next(self.wm.parameters()).device
        
        # 기존 OnlineLora를 내부적으로 사용
        from planning.online import OnlineLora
        self.base_online_lora = OnlineLora(workspace)
        
        # 기존 Online-LoRA의 속성들을 참조
        self.is_online_lora = self.base_online_lora.is_online_lora
        self.hybrid_enabled = self.base_online_lora.hybrid_enabled
        self.task_based_stacking = self.base_online_lora.task_based_stacking
        self.loss_based_stacking = self.base_online_lora.loss_based_stacking
        self.max_stacks_per_task = self.base_online_lora.max_stacks_per_task
        self.stacks_in_current_task = self.base_online_lora.stacks_in_current_task
        self.current_task_id = self.base_online_lora.current_task_id
        self.task_changed = self.base_online_lora.task_changed  # 태스크 전환 감지 플래그
        self.stack_history = self.base_online_lora.stack_history
        self.last_loss = self.base_online_lora.last_loss  # 초기값 복사 (update()에서 동기화)
        self.last_visual_loss = getattr(self.base_online_lora, "last_visual_loss", None)
        self.last_proprio_loss = getattr(self.base_online_lora, "last_proprio_loss", None)
        self.optimizer = self.base_online_lora.optimizer
        self.loss_fn = self.base_online_lora.loss_fn
        self.visual_loss_weight = self.base_online_lora.visual_loss_weight
        self.proprio_loss_weight = self.base_online_lora.proprio_loss_weight
        
        # 앙상블 전용 설정
        self.ensemble_evaluation_steps = self.cfg.get("evaluation_steps", 10)
        
        # 앙상블 저장 정책 (기본: 스택 직후 저장 비활성화, 태스크 종료 시 저장)
        self.save_on_stack = self.cfg.get("save_on_stack", False)
        self.save_on_task_end = self.cfg.get("save_on_task_end", True)

        # 앙상블 관리자 초기화
        self.ensemble_manager = LoRAEnsembleManager(
            base_model=self.wm,
            max_ensemble_size=self.cfg.get("max_ensemble_size", 5),
            cache_dir=self.cfg.get("cache_dir", "./lora_cache"),
            max_memory_mb=self.cfg.get("max_memory_mb", 200)
        )

        # 최근 앙상블 평가 결과 저장
        self.last_ensemble_evaluation_summary: Optional[Dict[str, Any]] = None
        self.created_task_ids = set()
        
        print(f"EnsembleOnlineLora initialized:")
        print(f"  - Base OnlineLora: {self.is_online_lora}")
        print(f"  - Ensemble enabled: {self.hybrid_enabled}")
        print(f"  - Task-based stacking: {self.task_based_stacking}")
        
        # 🔧 OnlineLora에 콜백 설정
        self.base_online_lora.on_lora_stack_callback = self._on_lora_stacked
        
        # 앙상블 전용 콜백 초기화
        self.on_lora_stack_callback = None
    
    def update(self, trans_obs_0, actions, e_obses):
        """
        하나의 학습 단계를 수행하는 메인 메소드 (시간 측정 포함)
        
        Args:
            trans_obs_0: 변환된 초기 관측
            actions: 행동 시퀀스
            e_obses: 실제 관측 시퀀스
        """
        start_time = time.time()
        # 기존 OnlineLora의 학습 로직 사용 (반환값 없음, 내부적으로 last_loss 설정)
        self.base_online_lora.update(trans_obs_0, actions, e_obses)
        adaptation_time = time.time() - start_time
        print(f"Ensemble LoRA adaptation time: {adaptation_time:.4f} seconds")
        
        # 🔧 adaptation time 저장 (base_online_lora에 이미 저장되지만, Ensemble 레벨에서도 추적)
        if not hasattr(self, 'adaptation_times'):
            self.adaptation_times = []
        self.adaptation_times.append(adaptation_time)
        
        # 🔧 모든 상태 동기화 (base_online_lora에서 설정된 값 사용)
        self.last_loss = self.base_online_lora.last_loss
        self.last_visual_loss = getattr(self.base_online_lora, "last_visual_loss", None)
        self.last_proprio_loss = getattr(self.base_online_lora, "last_proprio_loss", None)
        self.last_visual_loss = getattr(self.base_online_lora, "last_visual_loss", None)
        self.last_proprio_loss = getattr(self.base_online_lora, "last_proprio_loss", None)
        self.task_changed = self.base_online_lora.task_changed
        self.stacks_in_current_task = self.base_online_lora.stacks_in_current_task
        self.current_task_id = self.base_online_lora.current_task_id
        
        # 앙상블 기반 적층 로직 (향후 구현)
        if self.last_loss is not None and self.hybrid_enabled:
            self._manage_ensemble_stacking(self.last_loss)
    
    def compute_loss_only(self, trans_obs_0, actions, e_obses):
        """
        온라인 학습 없이 현재 모델의 손실을 평가합니다.
        """
        metrics = self.base_online_lora.compute_loss_only(trans_obs_0, actions, e_obses)
        if metrics:
            self.last_loss = self.base_online_lora.last_loss
            self.last_visual_loss = getattr(self.base_online_lora, "last_visual_loss", None)
            self.last_proprio_loss = getattr(self.base_online_lora, "last_proprio_loss", None)
        return metrics
    
    def trigger_task_based_stacking(self, task_id, reason="task_change"):
        """
        태스크 기반 적층을 트리거합니다 (앙상블 기반)
        
        Args:
            task_id: 새로운 태스크 ID
            reason: 적층 이유
            
        Returns:
            bool: 적층 성공 여부
        """
        # 🔧 앙상블 전용 모드에서는 task_based_stacking이 false여도 작동
        if not self.hybrid_enabled:
            return False
        
        # 🔧 태스크가 변경된 경우 - base_online_lora와 동기화
        if task_id != self.current_task_id:
            self.current_task_id = task_id
            self.stacks_in_current_task = 0
            # base_online_lora의 값도 동기화
            self.base_online_lora.current_task_id = task_id
            self.base_online_lora.stacks_in_current_task = 0
            print(f"🔄 Task changed to {task_id}. Resetting stack counter.")
        
        # 🔧 최대 적층 횟수 확인 - base_online_lora의 실제 값 사용
        actual_stacks = getattr(self.base_online_lora, 'stacks_in_current_task', 0)
        if actual_stacks >= self.max_stacks_per_task:
            print(f"⚠️  Max stacks per task ({self.max_stacks_per_task}) reached. Skipping ensemble-based stacking.")
            return False
        
        # 앙상블 기반 적층 결정
        should_stack, best_member = self._should_stack_lora_ensemble(task_id)
        
        if should_stack:
            print(f"🎯 Ensemble-based LoRA stacking triggered (Task {task_id}, Reason: {reason})")
            
            # 🔧 앙상블 전용 모드에서는 직접 적층 수행 (task_based_stacking이 false여도)
            if not self.task_based_stacking:
                # 직접 적층 수행
                stacking_success = self._perform_ensemble_lora_stacking(task_id, best_member, reason)
                
                if stacking_success:
                    # 🔧 카운터 동기화 - base_online_lora의 값 사용
                    self.stacks_in_current_task = getattr(self.base_online_lora, 'stacks_in_current_task', 0)
                    print(f"✅ Ensemble-based stacking successful. Total stacks in task: {self.stacks_in_current_task}")
                    print(f"🔧 Final sync: base_online_lora.stacks_in_current_task = {self.base_online_lora.stacks_in_current_task}")
                    return True
                else:
                    print(f"❌ Ensemble-based stacking failed.")
                    return False
            else:
                # 기존 OnlineLora의 태스크 기반 적층 사용
                stacking_success = self.base_online_lora.trigger_task_based_stacking(task_id, reason)
                
                if stacking_success:
                    # 🔧 카운터 동기화 - base_online_lora의 값 사용
                    self.stacks_in_current_task = getattr(self.base_online_lora, 'stacks_in_current_task', 0)
                    print(f"✅ Ensemble-based stacking successful. Total stacks in task: {self.stacks_in_current_task}")
                    print(f"🔧 Final sync: base_online_lora.stacks_in_current_task = {self.base_online_lora.stacks_in_current_task}")
                    return True
                else:
                    print(f"❌ Base OnlineLora stacking failed. Skipping ensemble management.")
                    return False
        else:
            print(f"📊 Using existing ensemble for Task {task_id} (performance sufficient)")
            return False
    
    def _perform_training_step(self, trans_obs_0, actions, e_obses):
        """실제 예측, 손실 계산, 역전파 및 업데이트를 수행합니다."""
        try:
            print("--- Starting Ensemble LoRA Online Learning ---")
            
            # 1. 예측 (그래디언트 활성화)
            step_start = time.time()
            i_z_obses_pred, _ = self.wm.rollout(obs_0=trans_obs_0, act=actions)
            rollout_time = time.time() - step_start

            # 2. 정답 준비 (그래디언트 비활성화)
            encode_start = time.time()
            with torch.no_grad():
                trans_obs_gt = self.workspace.data_preprocessor.transform_obs(e_obses)
                trans_obs_gt = move_to_device(trans_obs_gt, self.device)
                i_z_obses_gt = self.wm.encode_obs(trans_obs_gt)
            encode_time = time.time() - encode_start

            # 3. 손실 계산
            loss_start = time.time()
            print("Computing ensemble loss...")
            frameskip = self.workspace.frameskip
            gt_proprio_resampled = i_z_obses_gt["proprio"][:, ::frameskip, :].detach()
            gt_visual_resampled = i_z_obses_gt["visual"][:, ::frameskip, :, :].detach()
            
            proprio_loss = self.loss_fn(i_z_obses_pred["proprio"], gt_proprio_resampled)
            visual_loss = self.loss_fn(i_z_obses_pred["visual"], gt_visual_resampled)
            
            total_loss = self.visual_loss_weight * visual_loss + self.proprio_loss_weight * proprio_loss
            loss_time = time.time() - loss_start
            
            print(f"Visual loss: {visual_loss.item():.6f}, Proprio loss: {proprio_loss.item():.6f}")
            print(f"Total loss: {total_loss.item():.6f}")

            # 4. 역전파 및 업데이트
            backward_start = time.time()
            if self.optimizer is None:
                # 첫 번째 학습 시 옵티마이저 초기화
                params_to_train = [p for p in self.wm.parameters() if p.requires_grad]
                self.optimizer = torch.optim.Adam(params_to_train, lr=self.cfg.get("lr", 1e-4))
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            backward_time = time.time() - backward_start
            
            print(f"Ensemble LoRA step timing - Rollout: {rollout_time:.4f}s, Encode: {encode_time:.4f}s, Loss: {loss_time:.4f}s, Backward: {backward_time:.4f}s")

            return total_loss.item()

        except Exception as e:
            print(f"Error during ensemble training step: {e}")
            return None
        
        finally:
            # 5. 메모리 정리
            if 'i_z_obses_pred' in locals(): del i_z_obses_pred
            if 'i_z_obses_gt' in locals(): del i_z_obses_gt
            if 'total_loss' in locals(): del total_loss
            torch.cuda.empty_cache()
            print("--- Ensemble LoRA Online Update Complete ---")
    
    def _should_stack_lora_ensemble(self, task_id, input_data=None, target_data=None):
        """
        앙상블 기반으로 LoRA 적층 여부 결정
        
        Args:
            task_id: 현재 태스크 ID
            input_data: 입력 데이터 (선택사항)
            target_data: 타겟 데이터 (선택사항)
            
        Returns:
            Tuple[bool, Dict]: (적층 여부, 최적 멤버 정보)
        """
        # 앙상블 멤버 상태 확인
        
        # 🔧 앙상블 멤버 상세 정보 출력
        if self.ensemble_manager.ensemble_members:
            for task_id_member, member_info in self.ensemble_manager.ensemble_members.items():
                print(f"   - Member Task {task_id_member}: {member_info.get('performance', {}).get('loss', 'N/A')}")
        
        # 현재 앙상블 성능 평가
        if not self.ensemble_manager.ensemble_members:
            print("📊 No ensemble members available. Stacking new LoRA.")
            return True, None
        
        # 🔧 앙상블 멤버가 있으면 항상 실시간 성능 평가 수행
        print(f"📊 Found {len(self.ensemble_manager.ensemble_members)} ensemble members. Performing real-time evaluation...")
        
        # 실시간 평가를 위해 continual_plan.py의 evaluate_members_for_new_task 호출
        # 이 메서드는 각 멤버에 대해 실시간으로 성능을 평가함
        print("📊 Performing real-time ensemble evaluation...")
        
        # 실시간 평가는 continual_plan.py에서 수행되므로 여기서는 단순히 LoRA 적층만 수행
        print("📊 Real-time evaluation will be performed by continual_plan.py")
        print("📊 Proceeding with LoRA stacking...")
        return True, None
    
    def _perform_ensemble_lora_stacking(self, task_id, best_member, reason):
        """
        앙상블 기반 LoRA 적층 수행 (OnlineLora에 위임)
        
        Args:
            task_id: 태스크 ID
            best_member: 최적 멤버 정보 (사용하지 않음)
            reason: 적층 이유
            
        Returns:
            bool: 적층 성공 여부
        """
        try:
            # 🔧 OnlineLora의 _perform_lora_stacking 메서드 직접 호출
            stacking_success = self.base_online_lora._perform_lora_stacking("ensemble_based", task_id, reason)
            
            if stacking_success:
                # 🔧 base_online_lora의 카운터 직접 업데이트
                if hasattr(self.base_online_lora, 'stacks_in_current_task'):
                    self.base_online_lora.stacks_in_current_task += 1
                    print(f"🔧 Updated base_online_lora.stacks_in_current_task to {self.base_online_lora.stacks_in_current_task}")
                
                # 🔧 EnsembleOnlineLora의 카운터도 동기화
                self.stacks_in_current_task = self.base_online_lora.stacks_in_current_task
                
                # 적층 완료 로그
                print(f"Ensemble-based LoRA stacking completed successfully!")
                print(f"   - Task ID: {task_id}")
                print(f"   - Reason: {reason}")
                print(f"   - Stacks in current task: {self.stacks_in_current_task}/{self.max_stacks_per_task}")
                
                return True
            else:
                print(f"❌ OnlineLora stacking failed.")
                return False
            
        except Exception as e:
            print(f"❌ Error during ensemble-based LoRA stacking: {e}")
            return False
    
    def _manage_ensemble_stacking(self, current_loss_value):
        """
        앙상블 기반 적층 관리 (향후 구현)
        
        Args:
            current_loss_value: 현재 loss 값
        """
        # 향후 loss 기반 앙상블 적층 로직 구현
        pass
    
    def _extract_current_stacked_lora_weights(self):
        """
        현재 적층된 LoRA 가중치 추출 (모든 적층 효과 포함)
        
        Returns:
            Dict: 현재 적층된 LoRA 가중치 (첫 번째 스택 - 모든 적층 효과 포함)
        """
        lora_weights = {}
        
        try:
            # LoRA_ViT_spread에서 적층된 가중치 추출
            if hasattr(self.wm.predictor, 'w_As') and hasattr(self.wm.predictor, 'w_Bs'):
                w_As = self.wm.predictor.w_As
                w_Bs = self.wm.predictor.w_Bs
                
                print(f"📊 Extracting LoRA weights from {len(w_As)} total layers...")
                
                # 🔧 첫 번째 스택 추출 (모든 적층 효과가 포함된 레이어들)
                # ViT에 6개의 attention block이 있고, 각 블록마다 q, v 2개씩 = 총 12개
                layers_per_stack = 12  # 각 LoRA 스택당 레이어 수
                
                # 첫 번째 스택 (모든 적층 효과가 포함된 레이어들) 추출
                # 🔧 w + wnew를 추출하여 실제 사용되는 가중치 저장
                wnew_As = getattr(self.wm.predictor, 'wnew_As', [])
                wnew_Bs = getattr(self.wm.predictor, 'wnew_Bs', [])
                
                for i in range(min(layers_per_stack, len(w_As))):
                    layer_key = f'layer_{i}'
                    
                    # w + wnew 계산 (실제 사용되는 가중치)
                    w_A_combined = w_As[i].weight.data.clone().detach()
                    w_B_combined = w_Bs[i].weight.data.clone().detach()
                    
                    if i < len(wnew_As) and i < len(wnew_Bs):
                        w_A_combined += wnew_As[i].weight.data.clone().detach()
                        w_B_combined += wnew_Bs[i].weight.data.clone().detach()
                    
                    lora_weights[layer_key] = {
                        'w_A': w_A_combined,  # w + wnew (실제 사용되는 가중치)
                        'w_B': w_B_combined   # w + wnew (실제 사용되는 가중치)
                    }
                
                print(f"✅ Successfully extracted {len(lora_weights)} LoRA layers with all stacking effects")
                
                # 🔧 디버깅: 적층 효과 확인
                if len(w_As) > layers_per_stack:
                    print(f"📊 LoRA Stacking Info:")
                    print(f"   - Total layers: {len(w_As)}")
                    print(f"   - Layers per stack: {layers_per_stack}")
                    print(f"   - Number of stacks: {len(w_As) // layers_per_stack}")
                    print(f"   - Extracted: First stack (all stacking effects included)")
                else:
                    print(f"📊 Single LoRA stack detected (no stacking yet)")
                    
            else:
                print(f"⚠️  w_As or w_Bs not found in predictor. Available attributes:")
                if hasattr(self.wm.predictor, '__dict__'):
                    for attr in dir(self.wm.predictor):
                        if not attr.startswith('_'):
                            print(f"   - {attr}")
                
        except Exception as e:
            print(f"❌ Error extracting LoRA weights: {e}")
            import traceback
            traceback.print_exc()
        
        return lora_weights
    
    def _apply_lora_weights(self, lora_weights):
        """
        앙상블 멤버의 LoRA 가중치를 현재 모델에 적용 (첫 번째 스택에 적용)
        
        Args:
            lora_weights: Dict - {'layer_0': {'w_A': tensor, 'w_B': tensor}, ...}
            
        Returns:
            bool: 적용 성공 여부
        """
        try:
            if not hasattr(self.wm, 'predictor'):
                print(f"⚠️  World model doesn't have predictor. Cannot apply LoRA weights.")
                return False
                
            predictor = self.wm.predictor
            
            if not (hasattr(predictor, 'w_As') and hasattr(predictor, 'w_Bs')):
                print(f"⚠️  Predictor doesn't have w_As/w_Bs. Cannot apply LoRA weights.")
                return False
            
            # 🔧 앙상블 추론을 위한 LoRA 적용 방식
            # 첫 번째 스택 (모든 적층 효과가 포함된 레이어들)에 앙상블 멤버의 가중치 적용
            
            # 현재 LoRA 상태 백업
            original_w_As = [w_A.weight.data.clone() for w_A in predictor.w_As]
            original_w_Bs = [w_B.weight.data.clone() for w_B in predictor.w_Bs]
            
            # 앙상블 멤버의 LoRA 가중치 적용
            w_As = predictor.w_As
            w_Bs = predictor.w_Bs
            
            with torch.no_grad():
                # 🔧 첫 번째 스택에만 앙상블 멤버의 가중치 적용
                layers_per_stack = 12
                
                for i in range(min(layers_per_stack, len(w_As))):
                    layer_key = f'layer_{i}'
                    
                    if layer_key in lora_weights:
                        w_As[i].weight.data.copy_(lora_weights[layer_key]['w_A'])
                        w_Bs[i].weight.data.copy_(lora_weights[layer_key]['w_B'])
                    else:
                        print(f"⚠️  Layer {layer_key} not found in lora_weights")
                        # 원래 가중치 복원
                        self._restore_lora_weights(original_w_As, original_w_Bs)
                        return False
            
            print(f"✅ Successfully applied LoRA weights from ensemble member to first stack")
            return True
            
        except Exception as e:
            print(f"❌ Error applying LoRA weights: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _restore_lora_weights(self, original_w_As, original_w_Bs):
        """원래 LoRA 가중치 복원"""
        try:
            predictor = self.wm.predictor
            w_As = predictor.w_As
            w_Bs = predictor.w_Bs
            
            with torch.no_grad():
                for i, (orig_A, orig_B) in enumerate(zip(original_w_As, original_w_Bs)):
                    w_As[i].weight.data.copy_(orig_A)
                    w_Bs[i].weight.data.copy_(orig_B)
            
            print(f"✅ Restored original LoRA weights")
            
        except Exception as e:
            print(f"❌ Error restoring LoRA weights: {e}")
    
    def _on_lora_stacked(self, steps, loss, task_id, stack_type, reason):
        """
        OnlineLora에서 LoRA 적층 후 호출되는 콜백 함수
        
        Args:
            steps: 적층까지의 스텝 수
            loss: 현재 loss 값
            task_id: 태스크 ID
            stack_type: 적층 타입 ("task_based" 또는 "loss_based")
            reason: 적층 이유
        """
        print(f"🔄 LoRA stacked in base model. save_on_stack={self.save_on_stack}")
        print(f"   - Task ID: {task_id}")
        print(f"   - Stack Type: {stack_type}")
        print(f"   - Reason: {reason}")
        print(f"   - Steps: {steps}")
        loss_str = f"{loss:.6f}" if loss is not None else "N/A"
        print(f"   - Loss: {loss_str}")
        
        # 저장 정책: 기본은 스택 직후 저장하지 않고 태스크 종료 시 저장
        if not self.save_on_stack:
            # 메타만 기록해두고 저장은 연기
            self._pending_stack_info = {
                'task_id': task_id,
                'steps': steps,
                'stack_type': stack_type,
                'reason': reason,
                'loss_at_stack': float(loss) if loss is not None else None,
                'timestamp': time.time(),
            }
            print("ℹ️ Deferring ensemble save until task end (finalized state)")
            return
        
        # save_on_stack=True인 경우에만 즉시 저장
        try:
            self._save_current_lora_member_impl(task_id=task_id, reason=f"{stack_type}:{reason}", loss_value=loss, steps=steps)
        except Exception as e:
            print(f"❌ Error in immediate save during _on_lora_stacked: {e}")
            import traceback
            traceback.print_exc()

    def save_current_lora_member(self, task_id: int, reason: str = "task_end") -> bool:
        """태스크 종료/수렴 시점의 LoRA 가중치를 앙상블에 저장."""
        if not self.save_on_task_end:
            print("ℹ️ save_on_task_end is disabled; skipping final save")
            return False
        if task_id in self.created_task_ids:
            print(f"ℹ️ LoRA member for Task {task_id} already saved in this session. Skipping save.")
            return False
        
        # 🔧 stacks_in_current_task 동기화 문제 해결
        # base_online_lora의 실제 값을 직접 참조
        try:
            actual_stacks = getattr(self.base_online_lora, 'stacks_in_current_task', 0)
            print(f"🔍 Checking stacking status: actual_stacks={actual_stacks}")
            
            if actual_stacks == 0:
                print("ℹ️ No LoRA stacking in current task; skipping final save at task end")
                return False
        except Exception as e:
            print(f"⚠️  Could not determine stacking status: {e}")
            # 에러 발생 시에도 저장을 시도 (안전한 폴백)
        
        try:
            last_loss = getattr(self.base_online_lora, 'last_loss', None)
            steps = getattr(self.base_online_lora, 'steps_since_last_stack', 0)
            return self._save_current_lora_member_impl(task_id=task_id, reason=reason, loss_value=last_loss, steps=steps)
        except Exception as e:
            print(f"❌ Error in save_current_lora_member: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_current_lora_member_impl(self, task_id: int, reason: str, loss_value: float, steps: int) -> bool:
        # 🔧 적층 없이 멤버를 사용 중이면 저장하지 않음
        if getattr(self, 'using_member_without_stacking', False):
            print(f"ℹ️  Using member without stacking - skipping ensemble save for Task {task_id}")
            return False
        
        current_weights = self._extract_current_stacked_lora_weights()
        if not current_weights:
            print("⚠️ No LoRA weights extracted. Skipping save.")
            return False
        
        # 🔧 저장 시점 fingerprint 계산 및 출력
        try:
            fingerprint_parts = self._fingerprint_weights(current_weights, sample_layers=4)
            fingerprint = "|".join(fingerprint_parts) if fingerprint_parts else "EMPTY"
            print(f"🧬 Save-time fingerprint for Task {task_id}")
        except Exception as e:
            print(f"⚠️ Could not compute save-time fingerprint: {e}")
            fingerprint = None
        
        performance = {
            'loss': float(loss_value) if loss_value is not None else float('inf'),
            'steps': int(steps) if steps is not None else 0,
            'stack_type': reason,
        }
        metadata = {
            'reason': reason,
            'saved_at': time.time(),
            'fingerprint': fingerprint,  # 🔧 fingerprint 저장
        }
        saved = self.ensemble_manager.add_ensemble_member(
            task_id=task_id,
            lora_weights=current_weights,
            performance=performance,
            metadata=metadata,
        )
        if saved:
            print(f"💾 Saved finalized LoRA member for Task {task_id} (reason={reason})")
            self.created_task_ids.add(task_id)
        return saved
    
    
    
    
    
    
    
    
    def check_task_change(self, new_task_id):
        """
        태스크 전환을 감지하고 task_changed 플래그를 설정합니다.
        base_online_lora의 메서드를 호출하고 결과를 동기화합니다.
        
        Args:
            new_task_id (int): 새로운 태스크 ID
            
        Returns:
            bool: 태스크가 실제로 변경되었는지 여부
        """
        # base_online_lora의 check_task_change 호출
        task_changed = self.base_online_lora.check_task_change(new_task_id)
        
        # 🔧 모든 상태 동기화
        self.task_changed = self.base_online_lora.task_changed
        self.current_task_id = self.base_online_lora.current_task_id
        self.stacks_in_current_task = self.base_online_lora.stacks_in_current_task
        self.last_loss = self.base_online_lora.last_loss
        self.last_visual_loss = getattr(self.base_online_lora, "last_visual_loss", None)
        self.last_proprio_loss = getattr(self.base_online_lora, "last_proprio_loss", None)
        
        # 🔧 태스크가 변경되면 using_member_without_stacking 플래그 리셋
        if task_changed:
            if hasattr(self, 'using_member_without_stacking'):
                print(f"🔄 Task changed - resetting using_member_without_stacking flag")
                self.using_member_without_stacking = False
                self.base_member_task_id = None
        
        return task_changed
    
    def reset_task_changed_flag(self):
        """task_changed 플래그를 리셋합니다."""
        self.base_online_lora.reset_task_changed_flag()
        # 🔧 모든 상태 동기화
        self.task_changed = self.base_online_lora.task_changed
        self.current_task_id = self.base_online_lora.current_task_id
        self.stacks_in_current_task = self.base_online_lora.stacks_in_current_task
        self.last_loss = self.base_online_lora.last_loss

    # =====================
    # Debug/Verification Utils
    # =====================

    def _tensor_hash(self, tensor_obj) -> str:
        """Return SHA256 hash of a tensor's CPU bytes (float32 assumed)."""
        try:
            import hashlib
            arr = tensor_obj.detach().cpu().contiguous().numpy()
            return hashlib.sha256(arr.tobytes()).hexdigest()
        except Exception:
            return "NA"

    def _fingerprint_weights(self, lora_weights: dict, sample_layers: int = 4) -> list:
        """Create a simple fingerprint list [hashA0, hashB0, hashA1, hashB1, ...]."""
        if not isinstance(lora_weights, dict) or not lora_weights:
            return []
        keys = sorted(list(lora_weights.keys()))[:sample_layers]
        parts = []
        for k in keys:
            lw = lora_weights.get(k, {})
            if not isinstance(lw, dict):
                continue
            w_a = lw.get('w_A', None)
            w_b = lw.get('w_B', None)
            if w_a is not None:
                parts.append(self._tensor_hash(w_a))
            if w_b is not None:
                parts.append(self._tensor_hash(w_b))
        return parts

    def compute_member_fingerprint(self, task_id: int, sample_layers: int = 4) -> str:
        """Return a compact string fingerprint for a given member."""
        if task_id not in self.ensemble_manager.ensemble_members:
            return ""
        lora_weights = self.ensemble_manager.ensemble_members[task_id].get('lora_weights', {})
        parts = self._fingerprint_weights(lora_weights, sample_layers)
        return "|".join(parts) if parts else "EMPTY"

    def log_all_member_fingerprints(self, sample_layers: int = 4):
        """Print fingerprints for all ensemble members and detect duplicates."""
        if not self.ensemble_manager.ensemble_members:
            print("⚠️  No ensemble members available for fingerprinting")
            return
        fingerprints = {}
        for m_task_id, m_info in self.ensemble_manager.ensemble_members.items():
            fp = self.compute_member_fingerprint(m_task_id, sample_layers)
            fingerprints[m_task_id] = fp
            print(f"🔍 Ensemble fingerprint - Task {m_task_id}: {fp}")
        # duplicate groups
        fp_groups = {}
        for tid, fp in fingerprints.items():
            fp_groups.setdefault(fp, []).append(tid)
        dups = [grp for grp in fp_groups.values() if len(grp) > 1]
        if dups:
            print(f"⚠️  Detected identical fingerprints (sampled layers): {dups}")
        else:
            print("✅ All ensemble member fingerprints differ (on sampled layers)")

    def verify_ensemble_save_load_integrity(self, tmp_path: str, sample_layers: int = 4) -> bool:
        """Save ensemble to disk, reload into a fresh manager, and compare fingerprints."""
        if not self.ensemble_manager.ensemble_members:
            print("⚠️  No ensemble members to verify")
            return False
        # capture current fingerprints
        before = {tid: self.compute_member_fingerprint(tid, sample_layers)
                  for tid in self.ensemble_manager.ensemble_members.keys()}
        # save
        ok = self.ensemble_manager.save_ensemble_to_disk(tmp_path)
        if not ok:
            print("❌ Failed to save ensemble for integrity check")
            return False
        # load into a fresh manager
        temp_mgr = LoRAEnsembleManager(
            base_model=self.wm,
            max_ensemble_size=self.ensemble_manager.max_ensemble_size,
            cache_dir=self.ensemble_manager.cache_dir,
            max_memory_mb=self.ensemble_manager.max_memory_mb,
        )
        ok2 = temp_mgr.load_ensemble_from_disk(tmp_path)
        if not ok2:
            print("❌ Failed to load ensemble for integrity check")
            return False
        # compare
        after = {}
        for tid, info in temp_mgr.ensemble_members.items():
            lora_weights = info.get('lora_weights', {})
            parts = self._fingerprint_weights(lora_weights, sample_layers)
            after[tid] = "|".join(parts) if parts else "EMPTY"
        # report
        all_ok = True
        for tid in before.keys():
            b = before.get(tid, "")
            a = after.get(tid, "")
            same = (a == b)
            print(f"🧪 Integrity [{tid}]: {'OK' if same else 'MISMATCH'}")
            if not same:
                print(f"   before: {b}")
                print(f"   after : {a}")
                all_ok = False
        if all_ok:
            print("✅ Save/Load integrity verified (fingerprints match)")
        return all_ok

    def summarize_member_differences(self, sample_layers: int = 4):
        """Print simple differences between member weights (hash equality and L1 mean)."""
        import itertools
        if not self.ensemble_manager.ensemble_members:
            print("⚠️  No ensemble members available for difference summary")
            return
        members = list(self.ensemble_manager.ensemble_members.items())
        for (tid_a, a), (tid_b, b) in itertools.combinations(members, 2):
            lw_a = a.get('lora_weights', {})
            lw_b = b.get('lora_weights', {})
            keys = sorted(list(set(lw_a.keys()) & set(lw_b.keys())))[:sample_layers]
            same_hash = 0
            total = 0
            l1_accum = 0.0
            for k in keys:
                wa_a = lw_a[k]['w_A']; wb_a = lw_a[k]['w_B']
                wa_b = lw_b[k]['w_A']; wb_b = lw_b[k]['w_B']
                ha = (self._tensor_hash(wa_a), self._tensor_hash(wb_a))
                hb = (self._tensor_hash(wa_b), self._tensor_hash(wb_b))
                same_hash += int(ha == hb)
                total += 1
                try:
                    import torch
                    l1_accum += torch.mean(torch.abs(wa_a.detach().cpu() - wa_b.detach().cpu())).item()
                    l1_accum += torch.mean(torch.abs(wb_a.detach().cpu() - wb_b.detach().cpu())).item()
                except Exception:
                    pass
            hash_eq_ratio = (same_hash / total) if total else 0.0
            avg_l1 = (l1_accum / (2 * total)) if total else 0.0
            print(f"🔎 Diff Task {tid_a} vs {tid_b}: hash_eq_ratio={hash_eq_ratio:.2f}, avg_L1={avg_l1:.6f}")
    
    def _apply_lora_weights(self, lora_weights, metadata=None):
        """
        LoRA 가중치를 현재 모델에 적용합니다.
        LoRAEnsembleManager의 로직을 직접 구현합니다.
        
        Args:
            lora_weights: Dict - {'layer_0': {'w_A': tensor, 'w_B': tensor}, ...}
            metadata: 추가 메타데이터 (선택사항)
            
        Returns:
            bool: 적용 성공 여부
        """
        try:
            if not hasattr(self.wm, 'predictor'):
                print(f"⚠️  World model doesn't have predictor. Cannot apply LoRA weights.")
                return False
                
            predictor = self.wm.predictor
            
            if not (hasattr(predictor, 'w_As') and hasattr(predictor, 'w_Bs')):
                print(f"⚠️  Predictor doesn't have w_As/w_Bs. Cannot apply LoRA weights.")
                return False
            
            # 🔧 w와 wnew 초기화: 앙상블 멤버 로드 전에 이전 값 제거
            with torch.no_grad():
                # w_As와 w_Bs 초기화
                for w_A in predictor.w_As:
                    nn.init.zeros_(w_A.weight)
                for w_B in predictor.w_Bs:
                    nn.init.zeros_(w_B.weight)
                
                # wnew_As와 wnew_Bs 초기화
                if hasattr(predictor, 'wnew_As') and hasattr(predictor, 'wnew_Bs'):
                    for wnew_A in predictor.wnew_As:
                        nn.init.zeros_(wnew_A.weight)
                    for wnew_B in predictor.wnew_Bs:
                        nn.init.zeros_(wnew_B.weight)
            
            print(f"🔧 Reset all LoRA weights (w, wnew) to zeros before loading ensemble member")
            
            # 첫 번째 스택 (모든 적층 효과가 포함된 레이어들)에 앙상블 멤버의 가중치 적용
            w_As = predictor.w_As
            w_Bs = predictor.w_Bs
            
            layers_per_stack = 12
            
            with torch.no_grad():
                for i in range(min(layers_per_stack, len(w_As))):
                    layer_key = f'layer_{i}'
                    
                    if layer_key in lora_weights:
                        w_As[i].weight.data.copy_(lora_weights[layer_key]['w_A'])
                        w_Bs[i].weight.data.copy_(lora_weights[layer_key]['w_B'])
                    else:
                        print(f"⚠️  Layer {layer_key} not found in lora_weights")
                        return False
            
            return True
                
        except Exception as e:
            print(f"❌ Error applying LoRA weights: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _restore_lora_weights(self, original_w_As, original_w_Bs):
        """
        원래 LoRA 가중치를 복원합니다.
        LoRAEnsembleManager의 로직을 직접 구현합니다.
        
        Args:
            original_w_As: 백업된 w_As 리스트
            original_w_Bs: 백업된 w_Bs 리스트
        """
        try:
            predictor = self.wm.predictor
            w_As = predictor.w_As
            w_Bs = predictor.w_Bs
            
            with torch.no_grad():
                for i, (orig_A, orig_B) in enumerate(zip(original_w_As, original_w_Bs)):
                    w_As[i].weight.data.copy_(orig_A)
                    w_Bs[i].weight.data.copy_(orig_B)
            
            print(f"✅ Restored original LoRA weights")
            
        except Exception as e:
            print(f"❌ Error restoring LoRA weights: {e}")
            import traceback
            traceback.print_exc()

    # =====================
    # Ensemble Evaluation Methods (moved from continual_plan.py)
    # =====================

    def perform_task_change_ensemble_selection(self, workspace):
        """
        태스크 전환 시 새로운 태스크에 대한 실제 성능 평가를 통한 최적 멤버 선택 및 적층
        """
        print(f"🎯 Performing task change ensemble selection with task-specific evaluation...")

        # 최근 평가 요약 초기화
        self.last_ensemble_evaluation_summary = None
        
        ensemble_cfg = workspace.cfg_dict.get("lora", {}).get("ensemble_cfg", {})
        inference_cfg = ensemble_cfg.get("inference", {})
        
        # 설정 확인
        task_change_evaluation = inference_cfg.get("task_change_evaluation", True)
        task_specific_evaluation = inference_cfg.get("task_specific_evaluation", True)
        select_best_member = inference_cfg.get("select_best_member", True)
        stack_on_selected = inference_cfg.get("stack_on_selected", True)
        evaluation_loss_threshold = inference_cfg.get("evaluation_loss_threshold", 0.1)
        
        if not task_change_evaluation:
            print(f"⚠️  Task change evaluation disabled, skipping ensemble selection")
            self.last_ensemble_evaluation_summary = {
                "status": "skipped",
                "reason": "task_change_evaluation_disabled",
            }
            return
        
        if not task_specific_evaluation:
            print(f"⚠️  Task-specific evaluation disabled, using stored performance")
            self.perform_legacy_ensemble_selection(workspace)
            self.last_ensemble_evaluation_summary = {
                "status": "legacy_selection",
                "reason": "task_specific_evaluation_disabled",
            }
            return
        
        try:
            # 1. 🔧 새로운 태스크에 대한 각 멤버의 실제 성능 평가
            print(f"📊 Evaluating ensemble members for new task...")
            # 멤버가 없으면 평가를 건너뜀 (초기 태스크 등)
            if len(self.ensemble_manager.ensemble_members) == 0:
                print("ℹ️  No ensemble members available for evaluation (first task or no previous members).")
                print("ℹ️  Proceeding with normal planning without ensemble selection.")
                self.last_ensemble_evaluation_summary = {
                    "status": "skipped",
                    "reason": "no_ensemble_members",
                }
                return
            
            member_performances = self.evaluate_members_for_new_task(workspace)
            
            if not member_performances:
                print(f"⚠️  No valid member performances found")
                self.last_ensemble_evaluation_summary = {
                    "status": "skipped",
                    "reason": "no_valid_member_performance",
                }
                return
            
            # 2. 실제 성능 기반 최적 멤버 선택
            best_member_task_id, best_performance = min(member_performances, key=lambda x: x[1]['loss'])
            
            print(f"📈 Task-Specific Performance Results:")
            for task_id, performance in member_performances:
                print(f"   - Task {task_id}: Loss {performance['loss']:.6f}")
            print(f"🏆 Best member for new task: Task {best_member_task_id} (Loss: {best_performance['loss']:.6f})")
            
            stacking_triggered = False
            stacking_applied = False
            stacking_reason = "loss_within_threshold"
            
            # 3. loss 임계값 확인 후 LoRA 적층 여부 결정
            if best_performance['loss'] <= evaluation_loss_threshold:
                print(f"✅ Best member loss ({best_performance['loss']:.6f}) < threshold ({evaluation_loss_threshold})")
                print(f"🎯 No LoRA stacking needed - using best member directly")
                
                # 최적 멤버를 현재 모델에 적용하되 새로운 LoRA 적층은 하지 않음
                self.apply_best_member_without_stacking(best_member_task_id)
            else:
                print(f"⚠️  Best member loss ({best_performance['loss']:.6f}) > threshold ({evaluation_loss_threshold})")
                print(f"🔧 LoRA stacking needed - stacking on best member")
                stacking_triggered = True
                stacking_reason = "loss_above_threshold"
                
                if stack_on_selected:
                    # 선택된 멤버 위에 새로운 LoRA 적층
                    self.stack_on_selected_member(best_member_task_id, workspace)
                    stacking_applied = True
                else:
                    print(f"ℹ️  Stacking on selected member disabled")
                    stacking_reason = "stacking_disabled"
            
            # 🔒 평가 결과 저장 (부모 프로세스로 전달할 수 있도록)
            members_summary: List[Dict[str, Any]] = []
            for task_id, performance in member_performances:
                entry: Dict[str, Any] = {"task_id": int(task_id)}
                if isinstance(performance, dict):
                    for key, value in performance.items():
                        if isinstance(value, (int, float, bool)) or value is None:
                            entry[key] = value
                        else:
                            try:
                                entry[key] = float(value)
                            except Exception:
                                entry[key] = str(value)
                members_summary.append(entry)
            
            best_member_summary = {
                "task_id": int(best_member_task_id),
                "performance": {
                    key: (value if isinstance(value, (int, float, bool)) or value is None else str(value))
                    for key, value in (best_performance or {}).items()
                },
            }
            
            self.last_ensemble_evaluation_summary = {
                "status": "evaluated",
                "members": members_summary,
                "best_member": best_member_summary,
                "threshold": evaluation_loss_threshold,
                "stacking_triggered": stacking_triggered,
                "stacking_applied": stacking_applied,
                "stacking_reason": stacking_reason,
                "stack_on_selected": stack_on_selected,
                "select_best_member": select_best_member,
            }
                
        except Exception as e:
            print(f"❌ Task change ensemble selection failed: {e}")
            import traceback
            traceback.print_exc()
            self.last_ensemble_evaluation_summary = {
                "status": "error",
                "reason": str(e),
            }
    
    def evaluate_members_for_new_task(self, workspace):
        """
        새로운 태스크에 대한 각 앙상블 멤버의 실제 성능 평가
        
        Returns:
            List[Tuple[str, Dict]]: (task_id, performance) 튜플 리스트
        """
        print(f"🔍 Evaluating each member for new task...")
        # 앙상블 멤버 없으면 바로 반환
        if len(self.ensemble_manager.ensemble_members) == 0:
            print("⚠️  No ensemble members to evaluate.")
            return []

        # 디버깅: 현재 평가 대상 멤버 ID 목록 출력
        try:
            member_ids = list(self.ensemble_manager.ensemble_members.keys())
            print(f"🧩 Ensemble members to evaluate: {member_ids}")
        except Exception:
            pass

        ensemble_cfg = workspace.cfg_dict.get("ensemble_lora", {})
        inference_cfg = ensemble_cfg.get("inference", {})
        evaluation_steps = inference_cfg.get("evaluation_steps", 5)
        
        member_performances = []
        
        # 현재 LoRA 상태 백업
        original_w_As = None
        original_w_Bs = None
        
        try:
            if hasattr(self.wm.predictor, 'w_As') and hasattr(self.wm.predictor, 'w_Bs'):
                original_w_As = [w_A.weight.data.clone() for w_A in self.wm.predictor.w_As]
                original_w_Bs = [w_B.weight.data.clone() for w_B in self.wm.predictor.w_Bs]
        except Exception as e:
            print(f"⚠️  Warning: Could not backup LoRA weights: {e}")
        
        # 멤버 목록을 고정 스냅샷으로 복사하여 평가 중 변경으로 인한 스킵 방지
        members_snapshot = list(self.ensemble_manager.ensemble_members.items())
        for task_id, member_info in members_snapshot:
            try:
                print(f"📊 Evaluating member Task {task_id} for new task...")
                
                # 🔧 각 멤버 평가 전에 완전히 초기 상태로 리셋 (이전 멤버 영향 완전 제거)
                if hasattr(self.wm.predictor, 'w_As') and hasattr(self.wm.predictor, 'w_Bs'):
                    for w_A in self.wm.predictor.w_As:
                        nn.init.zeros_(w_A.weight)
                    for w_B in self.wm.predictor.w_Bs:
                        nn.init.zeros_(w_B.weight)
                
                if hasattr(self.wm.predictor, 'wnew_As') and hasattr(self.wm.predictor, 'wnew_Bs'):
                    for wnew_A in self.wm.predictor.wnew_As:
                        nn.init.zeros_(wnew_A.weight)
                    for wnew_B in self.wm.predictor.wnew_Bs:
                        nn.init.zeros_(wnew_B.weight)
                
                print(f"🔧 Reset all LoRA weights (w, wnew) to zeros before loading ensemble member {task_id}")
                
                # 해당 멤버의 LoRA 가중치를 모델에 적용
                lora_weights = member_info['lora_weights']
                # 로드 전후 지문 비교를 위한 저장 시점 fingerprint 가져오기(있다면)
                saved_fp = None
                try:
                    saved_fp = member_info.get('metadata', {}).get('fingerprint', None)
                except Exception:
                    saved_fp = None
                # 플래튼 저장본이면 wnew 이중 적용 방지를 위해 메타 전달
                meta_for_apply = member_info.get('metadata', {}) if isinstance(member_info, dict) else {}
                success = self._apply_lora_weights(lora_weights, metadata=meta_for_apply)
                
                if not success:
                    print(f"❌ Failed to apply LoRA weights for member {task_id}")
                    continue
                # 적용 후 현재 모델 첫 스택 지문 계산 및 대조 확인
                try:
                    applied_weights = self._extract_current_stacked_lora_weights()
                    applied_fp_parts = self._fingerprint_weights(applied_weights, sample_layers=4)
                    applied_fp = "|".join(applied_fp_parts) if applied_fp_parts else "EMPTY"
                    
                    if saved_fp:
                        match_status = "✅ MATCH" if applied_fp == saved_fp else "❌ MISMATCH"
                        print(f"🔍 Fingerprint comparison: {match_status}")
                        if applied_fp != saved_fp:
                            print(f"⚠️  WARNING: Task {task_id} fingerprint mismatch detected!")
                    else:
                        print(f"🧬 Load-time fingerprint (sampled): {applied_fp}")
                        print(f"⚠️  No saved fingerprint available for Task {task_id}")
                except Exception as e:
                    print(f"⚠️  Could not compute load-time fingerprint: {e}")
                
                # 새로운 태스크에 대한 실제 성능 평가
                # evaluator.py의 eval_actions를 직접 사용하여 평가
                performance = self.evaluate_member_for_current_task(
                    workspace, actions=None, evaluation_steps=evaluation_steps
                )
                
                if performance is not None:
                    member_performances.append((task_id, performance))
                    print(f"   - Task {task_id}: Loss {performance['loss']:.6f}")
                else:
                    print(f"   - Task {task_id}: Evaluation failed")
                # 각 멤버 평가 후 원래 LoRA 상태 복원 (독립성 보장)
                if original_w_As is not None and original_w_Bs is not None:
                    self._restore_lora_weights(original_w_As, original_w_Bs)
                
            except Exception as e:
                print(f"❌ Error evaluating member {task_id}: {e}")
                continue
        
        # 원래 LoRA 상태 복원
        if original_w_As is not None and original_w_Bs is not None:
            self._restore_lora_weights(original_w_As, original_w_Bs)
        
        print(f"✅ Evaluated {len(member_performances)} members for new task")
        return member_performances
    
    def evaluate_member_for_current_task(self, workspace, actions=None, evaluation_steps=5):
        """
        evaluator.py의 eval_actions를 직접 사용하여 앙상블 멤버 평가
        (망각 측정과 동일하게 단일 평가만 수행, MPC planning 없음)
        
        Args:
            workspace: PlanWorkspace 인스턴스
            actions: 사용되지 않음 (망각 측정과 동일하게 zero actions 사용)
            evaluation_steps: 사용되지 않음
            
        Returns:
            Dict: 성능 지표
        """
        try:
            # 망각 측정과 동일한 방식: MPC planning 없이 단순 평가만 수행
            # evaluate_loss_only()와 동일한 로직 사용
            if workspace.gt_actions is not None:
                actions_eval = (
                    workspace.gt_actions.to(device=workspace.device, dtype=torch.float32).detach()
                )
                action_len = np.full(actions_eval.shape[0], actions_eval.shape[1])
            else:
                batch_size = workspace.obs_0["visual"].shape[0]
                horizon = 1 if workspace.goal_H <= 0 else workspace.goal_H
                actions_eval = torch.zeros(
                    (batch_size, horizon, workspace.action_dim),
                    device=workspace.device,
                    dtype=torch.float32,
                )
                action_len = np.full(batch_size, horizon)

            workspace.evaluator.force_recenter_for_next_rollout()
            logs, successes, e_obses, _ = workspace.evaluator.eval_actions(
                actions_eval,
                action_len,
                save_video=False,
                filename="ensemble_eval",
                learning_enabled=False,
            )
            
            # logs에서 직접 loss 값 추출 (evaluator.py에서 이미 계산되어 있음)
            visual_loss = logs.get('visual_loss', None)
            proprio_loss = logs.get('proprio_loss', None)
            total_loss = logs.get('total_loss', None)
            
            # total_loss가 없으면 visual_loss와 proprio_loss 합산 시도
            if total_loss is None:
                if visual_loss is not None and proprio_loss is not None:
                    total_loss = visual_loss + proprio_loss
                elif visual_loss is not None:
                    total_loss = visual_loss
                else:
                    print("   ❌ Failed to extract loss from logs!")
                    return None
            
            visual_str = f"{visual_loss:.6f}" if visual_loss is not None else "N/A"
            proprio_str = f"{proprio_loss:.6f}" if proprio_loss is not None else "N/A"
            print(f"   📊 Ensemble evaluation loss: total={total_loss:.6f}, visual={visual_str}, proprio={proprio_str}")
            
            return {
                'loss': total_loss,
                'visual_loss': visual_loss if visual_loss is not None else (total_loss * 0.8),
                'proprio_loss': proprio_loss if proprio_loss is not None else (total_loss * 0.2),
                'success_rate': logs.get('success_rate', 0),
                'chamfer_distance': logs.get('chamfer_distance', None)
            }
            
        except Exception as e:
            print(f"❌ Error evaluating member: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_loss_from_output(self, output_text):
        """
        터미널 출력에서 loss 값을 파싱합니다.
        
        Args:
            output_text: 캡처된 터미널 출력
            
        Returns:
            float: 파싱된 loss 값 또는 None
        """
        try:
            # 방법 1: PARSED_LOSS_START 마커 사용
            if "PARSED_LOSS_START:" in output_text:
                start_marker = "PARSED_LOSS_START:"
                end_marker = ":PARSED_LOSS_END"
                
                start_idx = output_text.find(start_marker)
                if start_idx != -1:
                    start_idx += len(start_marker)
                    end_idx = output_text.find(end_marker, start_idx)
                    if end_idx != -1:
                        loss_str = output_text[start_idx:end_idx]
                        return float(loss_str)
            
            # 방법 2: "Total loss: " 패턴 사용
            if "Total loss: " in output_text:
                lines = output_text.split('\n')
                for line in lines:
                    if "Total loss: " in line:
                        # "Total loss: 0.071619" 형태에서 숫자 추출
                        parts = line.split("Total loss: ")
                        if len(parts) > 1:
                            loss_str = parts[1].strip()
                            return float(loss_str)
            
            return None
            
        except Exception as e:
            print(f"   ⚠️  Error parsing loss: {e}")
            return None
    
    def apply_best_member_without_stacking(self, best_member_task_id):
        """
        최적 멤버를 현재 모델에 적용하되 새로운 LoRA 적층은 하지 않음
        wnew는 0으로 초기화하여 온라인 학습은 가능하게 함 (단, 앙상블 멤버로 저장하지 않음)
        
        Args:
            best_member_task_id: 최적 멤버의 task_id
        """
        try:
            print(f"🎯 Applying best member {best_member_task_id} without stacking...")
            
            if best_member_task_id in self.ensemble_manager.ensemble_members:
                member_info = self.ensemble_manager.ensemble_members[best_member_task_id]
                lora_weights = member_info['lora_weights']
                
                # 최적 멤버의 가중치를 모델에 적용 (적층 없이)
                success = self._apply_lora_weights(lora_weights)
                
                if success:
                    # 🔧 wnew를 0으로 초기화하여 온라인 학습 가능하게 함
                    if hasattr(self.wm.predictor, 'wnew_As') and hasattr(self.wm.predictor, 'wnew_Bs'):
                        import torch.nn as nn
                        for wnew_A in self.wm.predictor.wnew_As:
                            nn.init.zeros_(wnew_A.weight)
                        for wnew_B in self.wm.predictor.wnew_Bs:
                            nn.init.zeros_(wnew_B.weight)
                        print(f"🔧 Initialized wnew to zeros - online learning enabled without stacking")
                    
                    # 🔧 적층 없이 사용 중임을 플래그로 표시 (앙상블 저장 방지)
                    self.using_member_without_stacking = True
                    self.base_member_task_id = best_member_task_id
                    
                    print(f"✅ Successfully applied best member's LoRA weights without stacking")
                    print(f"🎯 Using best member directly for new task (online learning enabled)")
                else:
                    print(f"❌ Failed to apply best member's LoRA weights")
            else:
                print(f"❌ Best member task {best_member_task_id} not found in ensemble members")
                
        except Exception as e:
            print(f"❌ Error applying best member without stacking: {e}")
            import traceback
            traceback.print_exc()
    
    def perform_legacy_ensemble_selection(self, workspace):
        """
        기존 방식의 앙상블 선택 (저장된 성능 기반)
        폴백용 메서드
        """
        print(f"🔄 Using legacy ensemble selection (stored performance)...")
        
        try:
            # 기존 방식: 저장된 성능으로 최적 멤버 선택
            best_member = self.ensemble_manager.get_best_member(
                input_data=workspace.obs_0,
                metric='loss'
            )
            
            if best_member is not None:
                best_task_id = best_member['task_id']
                print(f"🏆 Selected best ensemble member (legacy): Task {best_task_id}")
                
                # 선택된 멤버 위에 새로운 LoRA 적층
                self.stack_on_selected_member(best_task_id, workspace)
            else:
                print(f"⚠️  No suitable ensemble member found (legacy)")
                
        except Exception as e:
            print(f"❌ Legacy ensemble selection failed: {e}")
    
    def apply_latest_member_without_evaluation(self, workspace):
        """
        대조군 모드: 앙상블 평가 없이 가장 최근 멤버만 로드하여 사용
        
        Args:
            workspace: PlanWorkspace 인스턴스
        """
        try:
            print(f"🔬 Control Group Mode: Applying latest member without ensemble evaluation...")
            
            if not hasattr(self, 'ensemble_manager') or self.ensemble_manager is None:
                print("⚠️  No ensemble manager available")
                return
            
            if len(self.ensemble_manager.ensemble_members) == 0:
                print("ℹ️  No ensemble members available (first task). Proceeding with normal learning.")
                return
            
            # 가장 최근 멤버 찾기
            latest_task_id = max(self.ensemble_manager.ensemble_members.keys())
            latest_member = self.ensemble_manager.ensemble_members[latest_task_id]
            
            print(f"📌 Latest member: Task {latest_task_id}")
            
            # 최근 멤버 위에 새로운 LoRA 적층
            self.stack_on_selected_member(latest_task_id, workspace)
            
            print(f"✅ Control Group: Applied latest member (Task {latest_task_id}) and stacked new LoRA")
            
        except Exception as e:
            print(f"❌ Control Group: Failed to apply latest member: {e}")
            import traceback
            traceback.print_exc()
    
    def stack_on_selected_member(self, selected_task_id, workspace):
        """
        선택된 멤버 위에 새로운 LoRA 적층
        
        LoRAEnsembleManager의 기존 _apply_lora_weights 메서드를 활용
        
        Args:
            selected_task_id: 선택된 멤버의 task_id
            workspace: PlanWorkspace 인스턴스
        """
        try:
            print(f"🔧 Stacking new LoRA on selected member: Task {selected_task_id}")
            
            # 선택된 멤버의 LoRA 가중치를 현재 모델에 적용
            if selected_task_id in self.ensemble_manager.ensemble_members:
                member_info = self.ensemble_manager.ensemble_members[selected_task_id]
                lora_weights = member_info['lora_weights']
                
                # 🔧 EnsembleOnlineLora의 _apply_lora_weights 메서드 사용
                success = self._apply_lora_weights(lora_weights)
                
                if success:
                    print(f"✅ Successfully applied selected member's LoRA weights")
                    print(f"🔄 New LoRA will be stacked on top of selected member")
                    
                    # 실제 LoRA 적층 수행 (OnlineLora 경로로 위임)
                    try:
                        # 현재 태스크 ID를 명시적으로 전달
                        current_task_id = getattr(self, 'current_task_id', None)
                        if current_task_id is None:
                            current_task_id = selected_task_id
                        # 추적용: 선택된 멤버 ID 기록
                        try:
                            setattr(self, 'last_selected_member_task_id', selected_task_id)
                        except Exception:
                            pass
                        stacking_success = self._perform_ensemble_lora_stacking(
                            task_id=current_task_id,
                            best_member=None,
                            reason="task_change_eval"
                        )
                        if stacking_success:
                            print("✅ Triggered actual LoRA stacking on selected member")
                            # 스택 히스토리에 선택 멤버 ID를 명시적으로 추가(이중 보호)
                            try:
                                if hasattr(self, 'stack_history') and isinstance(self.stack_history, list):
                                    self.stack_history.append({
                                        'type': 'task_based',
                                        'reason': 'task_change_eval',
                                        'selected_member_task_id': selected_task_id,
                                        'timestamp': time.time(),
                                    })
                            except Exception:
                                pass
                        else:
                            print("❌ Failed to trigger actual LoRA stacking on selected member")
                    except Exception as e:
                        print(f"❌ Error triggering actual LoRA stacking: {e}")
                else:
                    print(f"❌ Failed to apply selected member's LoRA weights")
            else:
                print(f"❌ Selected task {selected_task_id} not found in ensemble members")
                
        except Exception as e:
            print(f"❌ Error stacking on selected member: {e}")
            import traceback
            traceback.print_exc()
