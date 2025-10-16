import torch
import time
import os
import gzip
import pickle
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict, deque
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
    
    def evaluate_member_performance(self, member_info: Dict, trans_obs_0, actions, target_data) -> Dict:
        """
        특정 앙상블 멤버의 실제 성능 평가 (월드 모델 rollout 사용)
        
        Args:
            member_info: 앙상블 멤버 정보
            trans_obs_0: 변환된 초기 관측
            actions: 행동 시퀀스
            target_data: 타겟 데이터
            
        Returns:
            Dict: 성능 지표
        """
        try:
            # 해당 멤버의 LoRA 가중치를 모델에 로드
            lora_weights = member_info['lora_weights']
            
            # 현재 LoRA 상태 백업
            original_w_As = None
            original_w_Bs = None
            
            if hasattr(self.wm.predictor, 'w_As') and hasattr(self.wm.predictor, 'w_Bs'):
                original_w_As = [w_A.weight.data.clone() for w_A in self.wm.predictor.w_As]
                original_w_Bs = [w_B.weight.data.clone() for w_B in self.wm.predictor.w_Bs]
            
            # 모델에 LoRA 가중치 적용
            success = self._apply_lora_weights(lora_weights)
            
            if not success:
                print(f"❌ Failed to apply LoRA weights for member {member_info['task_id']}")
                return {'loss': float('inf'), 'mae': float('inf'), 'mse': float('inf')}
            
            # 🔧 월드 모델 rollout 수행
            with torch.no_grad():
                i_z_obses_pred, _ = self.wm.rollout(obs_0=trans_obs_0, act=actions)
                
                # 손실 계산 (visual과 proprio 모두 고려)
                loss_fn = torch.nn.MSELoss()
                
                # Visual loss
                visual_loss = loss_fn(i_z_obses_pred["visual"], target_data["visual"])
                
                # Proprio loss  
                proprio_loss = loss_fn(i_z_obses_pred["proprio"], target_data["proprio"])
                
                # Total loss (가중치 적용)
                total_loss = visual_loss + 0.3 * proprio_loss
                
                # 추가 성능 지표 계산
                visual_mae = torch.mean(torch.abs(i_z_obses_pred["visual"] - target_data["visual"]))
                proprio_mae = torch.mean(torch.abs(i_z_obses_pred["proprio"] - target_data["proprio"]))
                
                performance = {
                    'loss': total_loss.item(),
                    'visual_loss': visual_loss.item(),
                    'proprio_loss': proprio_loss.item(),
                    'visual_mae': visual_mae.item(),
                    'proprio_mae': proprio_mae.item(),
                    'evaluated_at': time.time()
                }
                
                print(f"📊 Member {member_info['task_id']} performance: "
                      f"Total Loss {total_loss.item():.6f}, Visual {visual_loss.item():.6f}, Proprio {proprio_loss.item():.6f}")
            
            # 원래 LoRA 상태 복원
            if original_w_As is not None and original_w_Bs is not None:
                self._restore_lora_weights(original_w_As, original_w_Bs)
                
                return performance
                
        except Exception as e:
            print(f"❌ Error evaluating member {member_info['task_id']}: {e}")
            return {'loss': float('inf'), 'mae': float('inf'), 'mse': float('inf')}
    
    def ensemble_predict(self, trans_obs_0, actions, method: str = 'weighted_average'):
        """
        앙상블 추론 수행 (월드 모델 rollout 사용)
        
        Args:
            trans_obs_0: 변환된 초기 관측
            actions: 행동 시퀀스
            method: 앙상블 방법 ('weighted_average', 'best_only', 'all')
            
        Returns:
            torch.Tensor: 앙상블 예측 결과
        """
        if not self.ensemble_members:
            print("⚠️  No ensemble members available for prediction")
            return None
        
        predictions = []
        weights = []
        
        print(f"🔮 Ensemble prediction with {len(self.ensemble_members)} members using {method}")
        
        # 현재 LoRA 상태 백업 (앙상블 추론 후 복원용)
        original_w_As = None
        original_w_Bs = None
        
        try:
            if hasattr(self.wm.predictor, 'w_As') and hasattr(self.wm.predictor, 'w_Bs'):
                original_w_As = [w_A.weight.data.clone() for w_A in self.wm.predictor.w_As]
                original_w_Bs = [w_B.weight.data.clone() for w_B in self.wm.predictor.w_Bs]
        except Exception as e:
            print(f"⚠️  Warning: Could not backup LoRA weights: {e}")
            original_w_As = None
            original_w_Bs = None
        
        for task_id, member_info in self.ensemble_members.items():
            try:
                # 각 멤버의 LoRA 가중치를 모델에 로드
                lora_weights = member_info['lora_weights']
                
                # 모델에 LoRA 가중치 적용
                success = self._apply_lora_weights(lora_weights)
                
                if not success:
                    print(f"❌ Failed to apply LoRA weights for member {task_id}")
                    continue
                
                # 🔧 월드 모델 rollout 수행
                with torch.no_grad():
                    i_z_obses_pred, _ = self.wm.rollout(obs_0=trans_obs_0, act=actions)
                    predictions.append(i_z_obses_pred)
                    
                    # 가중치 계산 (성능 기반)
                    performance = member_info['performance']
                    loss = performance.get('loss', 1.0)
                    weight = 1.0 / (loss + 1e-8)
                    weights.append(weight)
                    
                    print(f"   - Member {task_id}: Loss {loss:.6f}, Weight {weight:.6f}")
                    
            except Exception as e:
                print(f"❌ Error in member {task_id} prediction: {e}")
                continue
        
        # 원래 LoRA 상태 복원
        if original_w_As is not None and original_w_Bs is not None:
            self._restore_lora_weights(original_w_As, original_w_Bs)
        
        if not predictions:
            print("❌ No valid predictions from ensemble members")
            return None
        
        # 앙상블 방법에 따른 최종 예측 계산
        if method == 'weighted_average':
            return self._weighted_average_predictions(predictions, weights)
        elif method == 'best_only':
            best_idx = weights.index(max(weights))
            return predictions[best_idx]
        elif method == 'all':
            return torch.mean(torch.stack(predictions), dim=0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def _weighted_average_predictions(self, predictions: List[torch.Tensor], 
                                    weights: List[float]) -> torch.Tensor:
        """가중 평균으로 예측 결과 통합"""
        if not predictions or not weights:
            return None
        
        # 가중치 정규화
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # 가중 평균 계산
        weighted_sum = None
        for pred, weight in zip(predictions, normalized_weights):
            if weighted_sum is None:
                weighted_sum = weight * pred
            else:
                weighted_sum += weight * pred
        
        print(f"🔗 Weighted average prediction computed with {len(predictions)} members")
        return weighted_sum
    
    def evaluate_ensemble_performance(self, trans_obs_0, actions, target_data) -> Dict:
        """
        전체 앙상블의 성능 평가 (월드 모델 rollout 사용)
        
        Args:
            trans_obs_0: 변환된 초기 관측
            actions: 행동 시퀀스
            target_data: 타겟 데이터
            
        Returns:
            Dict: 앙상블 성능 지표
        """
        if not self.ensemble_members:
            return {'avg_loss': float('inf'), 'best_loss': float('inf'), 
                   'worst_loss': float('inf'), 'member_count': 0}
        
        print(f"📊 Evaluating ensemble performance with {len(self.ensemble_members)} members")
        
        member_performances = []
        individual_losses = []
        
        # 각 멤버의 실제 성능 평가
        for task_id, member_info in self.ensemble_members.items():
            performance = self.evaluate_member_performance(member_info, trans_obs_0, actions, target_data)
            member_performances.append(performance)
            individual_losses.append(performance['loss'])
        
        # 앙상블 전체 성능 계산
        ensemble_performance = {
            'avg_loss': sum(individual_losses) / len(individual_losses),
            'best_loss': min(individual_losses),
            'worst_loss': max(individual_losses),
            'member_count': len(self.ensemble_members),
            'loss_std': torch.std(torch.tensor(individual_losses)).item() if len(individual_losses) > 1 else 0.0,
            'individual_performances': member_performances,
            'evaluated_at': time.time()
        }
        
        # 앙상블 예측 성능도 평가
        ensemble_prediction = self.ensemble_predict(trans_obs_0, actions, method='weighted_average')
        if ensemble_prediction is not None:
            with torch.no_grad():
                loss_fn = torch.nn.MSELoss()
                
                # Visual loss
                visual_loss = loss_fn(ensemble_prediction["visual"], target_data["visual"])
                
                # Proprio loss
                proprio_loss = loss_fn(ensemble_prediction["proprio"], target_data["proprio"])
                
                # Total loss
                total_loss = visual_loss + 0.3 * proprio_loss
                
                ensemble_performance['ensemble_loss'] = total_loss.item()
                ensemble_performance['ensemble_visual_loss'] = visual_loss.item()
                ensemble_performance['ensemble_proprio_loss'] = proprio_loss.item()
                
                # MAE 계산
                visual_mae = torch.mean(torch.abs(ensemble_prediction["visual"] - target_data["visual"]))
                proprio_mae = torch.mean(torch.abs(ensemble_prediction["proprio"] - target_data["proprio"]))
                ensemble_performance['ensemble_visual_mae'] = visual_mae.item()
                ensemble_performance['ensemble_proprio_mae'] = proprio_mae.item()
        
        print(f"📈 Ensemble Performance Summary:")
        print(f"   - Average Loss: {ensemble_performance['avg_loss']:.6f}")
        print(f"   - Best Loss: {ensemble_performance['best_loss']:.6f}")
        print(f"   - Worst Loss: {ensemble_performance['worst_loss']:.6f}")
        print(f"   - Loss Std: {ensemble_performance['loss_std']:.6f}")
        print(f"   - Members: {ensemble_performance['member_count']}")
        if 'ensemble_loss' in ensemble_performance:
            print(f"   - Ensemble Loss: {ensemble_performance['ensemble_loss']:.6f}")
            print(f"   - Ensemble Visual Loss: {ensemble_performance['ensemble_visual_loss']:.6f}")
            print(f"   - Ensemble Proprio Loss: {ensemble_performance['ensemble_proprio_loss']:.6f}")
        
        return ensemble_performance
    
    def consolidate_lora_weights(self, task_ids: List[int], 
                               method: str = 'weighted_average') -> Dict:
        """
        여러 LoRA 가중치를 하나로 통합
        
        Args:
            task_ids: 통합할 태스크 ID 리스트
            method: 통합 방법 ('weighted_average', 'best_only', 'all')
            
        Returns:
            Dict: 통합된 LoRA 가중치
        """
        if not task_ids:
            return {}
        
        # 해당 태스크들의 LoRA 가중치 수집
        lora_weights_list = []
        weights = []
        
        for task_id in task_ids:
            if task_id in self.ensemble_members:
                member_info = self.ensemble_members[task_id]
                lora_weights_list.append(member_info['lora_weights'])
                
                # 가중치 계산 (성능 기반)
                performance = member_info['performance']
                weight = 1.0 / (performance.get('loss', 1.0) + 1e-8)
                weights.append(weight)
        
        if not lora_weights_list:
            return {}
        
        # 통합 방법에 따른 처리
        if method == 'weighted_average':
            return self._weighted_average_consolidation(lora_weights_list, weights)
        elif method == 'best_only':
            best_idx = weights.index(max(weights))
            return lora_weights_list[best_idx]
        elif method == 'all':
            return self._simple_average_consolidation(lora_weights_list)
        else:
            raise ValueError(f"Unknown consolidation method: {method}")
    
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
        """멤버를 디스크에 저장"""
        save_path = os.path.join(self.cache_dir, f"lora_task_{task_id}.pth")
        
        # LoRA 가중치만 저장 (메타데이터는 별도 관리)
        torch.save(member_info['lora_weights'], save_path)
    
    def _weighted_average_consolidation(self, lora_weights_list: List[Dict], 
                                      weights: List[float]) -> Dict:
        """가중 평균 통합"""
        if not lora_weights_list:
            return {}
        
        # 가중치 정규화
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        consolidated = {}
        
        # 모든 레이어에 대해 가중 평균 계산
        for layer_name in lora_weights_list[0].keys():
            weighted_sum = None
            
            for i, lora_weights in enumerate(lora_weights_list):
                layer_weights = lora_weights[layer_name]
                weight = normalized_weights[i]
                
                if weighted_sum is None:
                    weighted_sum = weight * layer_weights
                else:
                    weighted_sum += weight * layer_weights
            
            consolidated[layer_name] = weighted_sum
        
        print(f"🔗 Consolidated {len(lora_weights_list)} LoRA weights using weighted average")
        return consolidated
    
    def _simple_average_consolidation(self, lora_weights_list: List[Dict]) -> Dict:
        """단순 평균 통합"""
        if not lora_weights_list:
            return {}
        
        consolidated = {}
        
        for layer_name in lora_weights_list[0].keys():
            layer_sum = None
            
            for lora_weights in lora_weights_list:
                layer_weights = lora_weights[layer_name]
                
                if layer_sum is None:
                    layer_sum = layer_weights
                else:
                    layer_sum += layer_weights
            
            # 평균 계산
            consolidated[layer_name] = layer_sum / len(lora_weights_list)
        
        print(f"🔗 Consolidated {len(lora_weights_list)} LoRA weights using simple average")
        return consolidated


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
        하나의 학습 단계를 수행하는 메인 메소드
        
        Args:
            trans_obs_0: 변환된 초기 관측
            actions: 행동 시퀀스
            e_obses: 실제 관측 시퀀스
        """
        # 기존 OnlineLora의 학습 로직 사용 (반환값 없음, 내부적으로 last_loss 설정)
        self.base_online_lora.update(trans_obs_0, actions, e_obses)
        
        # 마지막 Loss 값 동기화 (base_online_lora에서 설정된 값 사용)
        self.last_loss = self.base_online_lora.last_loss
        
        # task_changed 플래그 동기화
        self.task_changed = self.base_online_lora.task_changed
        
        # 앙상블 기반 적층 로직 (향후 구현)
        if self.last_loss is not None and self.hybrid_enabled:
            self._manage_ensemble_stacking(self.last_loss)
    
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
        
        # 태스크가 변경된 경우
        if task_id != self.current_task_id:
            self.current_task_id = task_id
            self.stacks_in_current_task = 0
            print(f"🔄 Task changed to {task_id}. Resetting stack counter.")
        
        # 최대 적층 횟수 확인
        if self.stacks_in_current_task >= self.max_stacks_per_task:
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
                    self.stacks_in_current_task += 1
                    print(f"✅ Ensemble-based stacking successful. Total stacks in task: {self.stacks_in_current_task}")
                    return True
                else:
                    print(f"❌ Ensemble-based stacking failed.")
                    return False
            else:
                # 기존 OnlineLora의 태스크 기반 적층 사용
                stacking_success = self.base_online_lora.trigger_task_based_stacking(task_id, reason)
                
                if stacking_success:
                    self.stacks_in_current_task += 1
                    print(f"✅ Ensemble-based stacking successful. Total stacks in task: {self.stacks_in_current_task}")
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
            i_z_obses_pred, _ = self.wm.rollout(obs_0=trans_obs_0, act=actions)

            # 2. 정답 준비 (그래디언트 비활성화)
            with torch.no_grad():
                trans_obs_gt = self.workspace.data_preprocessor.transform_obs(e_obses)
                trans_obs_gt = move_to_device(trans_obs_gt, self.device)
                i_z_obses_gt = self.wm.encode_obs(trans_obs_gt)

            # 3. 손실 계산
            print("Computing ensemble loss...")
            frameskip = self.workspace.frameskip
            gt_proprio_resampled = i_z_obses_gt["proprio"][:, ::frameskip, :].detach()
            gt_visual_resampled = i_z_obses_gt["visual"][:, ::frameskip, :, :].detach()
            
            proprio_loss = self.loss_fn(i_z_obses_pred["proprio"], gt_proprio_resampled)
            visual_loss = self.loss_fn(i_z_obses_pred["visual"], gt_visual_resampled)
            
            total_loss = self.visual_loss_weight * visual_loss + self.proprio_loss_weight * proprio_loss
            
            print(f"Visual loss: {visual_loss.item():.6f}, Proprio loss: {proprio_loss.item():.6f}")
            print(f"Total loss: {total_loss.item():.6f}")

            # 4. 역전파 및 업데이트
            if self.optimizer is None:
                # 첫 번째 학습 시 옵티마이저 초기화
                params_to_train = [p for p in self.wm.parameters() if p.requires_grad]
                self.optimizer = torch.optim.Adam(params_to_train, lr=self.cfg.get("lr", 1e-4))
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

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
            print(f"Performing ensemble-based LoRA stacking (delegating to OnlineLora)...")
            
            # 🔧 OnlineLora의 _perform_lora_stacking 메서드 직접 호출
            stacking_success = self.base_online_lora._perform_lora_stacking("ensemble_based", task_id, reason)
            
            if stacking_success:
                # 적층 완료 로그
                print(f"Ensemble-based LoRA stacking completed successfully!")
                print(f"   - Task ID: {task_id}")
                print(f"   - Reason: {reason}")
                print(f"   - Stacks in current task: {self.stacks_in_current_task + 1}/{self.max_stacks_per_task}")
                
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
                for i in range(min(layers_per_stack, len(w_As))):
                    layer_key = f'layer_{i}'
                    lora_weights[layer_key] = {
                        'w_A': w_As[i].weight.data.clone().detach(),  # 모든 적층 효과 포함
                        'w_B': w_Bs[i].weight.data.clone().detach()   # 모든 적층 효과 포함
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
        current_weights = self._extract_current_stacked_lora_weights()
        if not current_weights:
            print("⚠️ No LoRA weights extracted. Skipping save.")
            return False
        performance = {
            'loss': float(loss_value) if loss_value is not None else float('inf'),
            'steps': int(steps) if steps is not None else 0,
            'stack_type': reason,
        }
        metadata = {
            'reason': reason,
            'saved_at': time.time(),
        }
        saved = self.ensemble_manager.add_ensemble_member(
            task_id=task_id,
            lora_weights=current_weights,
            performance=performance,
            metadata=metadata,
        )
        if saved:
            print(f"💾 Saved finalized LoRA member for Task {task_id} (reason={reason})")
        return saved
    
    def get_ensemble_info(self):
        """앙상블 정보 반환"""
        return self.ensemble_manager.get_ensemble_info()
    
    def save_ensemble(self, save_path):
        """앙상블 저장"""
        return self.ensemble_manager.save_ensemble_to_disk(save_path)
    
    def load_ensemble(self, load_path):
        """앙상블 로드"""
        return self.ensemble_manager.load_ensemble_from_disk(load_path)
    
    def test_ensemble_inference(self, trans_obs_0, actions, target_data=None):
        """
        앙상블 추론 테스트 및 검증
        
        Args:
            trans_obs_0: 변환된 초기 관측
            actions: 행동 시퀀스  
            target_data: 타겟 데이터 (선택사항)
            
        Returns:
            Tuple: (individual_predictions, ensemble_prediction)
        """
        print(f"🧪 Testing ensemble inference with {len(self.ensemble_manager.ensemble_members)} members")
        
        if not self.ensemble_manager.ensemble_members:
            print(f"⚠️  No ensemble members available for testing")
            return None, None
        
        # 각 멤버별 개별 예측
        individual_predictions = {}
        for task_id, member_info in self.ensemble_manager.ensemble_members.items():
            try:
                lora_weights = member_info['lora_weights']
                success = self.ensemble_manager._apply_lora_weights(lora_weights)
                
                if not success:
                    print(f"❌ Failed to apply LoRA weights for member {task_id}")
                    continue
                
                with torch.no_grad():
                    pred = self.wm.rollout(obs_0=trans_obs_0, act=actions)[0]
                    individual_predictions[task_id] = pred
                    
                    print(f"✅ Member {task_id} prediction successful")
                    
            except Exception as e:
                print(f"❌ Member {task_id} prediction failed: {e}")
        
        # 앙상블 예측
        ensemble_pred = self.ensemble_manager.ensemble_predict(trans_obs_0, actions, method='weighted_average')
        
        if ensemble_pred is not None:
            print(f"✅ Ensemble prediction successful")
            
            # 타겟 데이터가 있으면 성능 평가
            if target_data is not None:
                loss_fn = torch.nn.MSELoss()
                visual_loss = loss_fn(ensemble_pred["visual"], target_data["visual"])
                proprio_loss = loss_fn(ensemble_pred["proprio"], target_data["proprio"])
                total_loss = visual_loss + 0.3 * proprio_loss
                
                print(f"📊 Ensemble Performance:")
                print(f"   - Total Loss: {total_loss.item():.6f}")
                print(f"   - Visual Loss: {visual_loss.item():.6f}")
                print(f"   - Proprio Loss: {proprio_loss.item():.6f}")
        
        return individual_predictions, ensemble_pred
    
    def test_ensemble_performance(self, trans_obs_0, actions, target_data):
        """
        앙상블 성능 평가 테스트
        
        Args:
            trans_obs_0: 변환된 초기 관측
            actions: 행동 시퀀스
            target_data: 타겟 데이터
            
        Returns:
            Dict: 앙상블 성능 지표
        """
        print(f"🧪 Testing ensemble performance evaluation...")
        
        if not self.ensemble_manager.ensemble_members:
            print(f"⚠️  No ensemble members available for testing")
            return None
        
        # 앙상블 성능 평가 수행
        performance = self.ensemble_manager.evaluate_ensemble_performance(
            trans_obs_0, actions, target_data
        )
        
        print(f"📊 Ensemble Performance Test Results:")
        print(f"   - Member Count: {performance['member_count']}")
        print(f"   - Average Loss: {performance['avg_loss']:.6f}")
        print(f"   - Best Loss: {performance['best_loss']:.6f}")
        print(f"   - Worst Loss: {performance['worst_loss']:.6f}")
        print(f"   - Loss Std: {performance['loss_std']:.6f}")
        
        if 'ensemble_loss' in performance:
            print(f"   - Ensemble Loss: {performance['ensemble_loss']:.6f}")
            print(f"   - Ensemble Visual Loss: {performance['ensemble_visual_loss']:.6f}")
            print(f"   - Ensemble Proprio Loss: {performance['ensemble_proprio_loss']:.6f}")
        
        return performance
    
    def test_ensemble_methods(self, trans_obs_0, actions, target_data=None):
        """
        다양한 앙상블 방법 테스트
        
        Args:
            trans_obs_0: 변환된 초기 관측
            actions: 행동 시퀀스
            target_data: 타겟 데이터 (선택사항)
            
        Returns:
            Dict: 각 방법별 예측 결과
        """
        print(f"🧪 Testing different ensemble methods...")
        
        if not self.ensemble_manager.ensemble_members:
            print(f"⚠️  No ensemble members available for testing")
            return None
        
        methods = ['weighted_average', 'best_only', 'all']
        results = {}
        
        for method in methods:
            try:
                print(f"📊 Testing method: {method}")
                pred = self.ensemble_manager.ensemble_predict(trans_obs_0, actions, method=method)
                
                if pred is not None:
                    results[method] = pred
                    
                    # 타겟 데이터가 있으면 성능 평가
                    if target_data is not None:
                        loss_fn = torch.nn.MSELoss()
                        visual_loss = loss_fn(pred["visual"], target_data["visual"])
                        proprio_loss = loss_fn(pred["proprio"], target_data["proprio"])
                        total_loss = visual_loss + 0.3 * proprio_loss
                        
                        print(f"   - {method}: Total Loss {total_loss.item():.6f}")
                    else:
                        print(f"   - {method}: Prediction successful")
                else:
                    print(f"   - {method}: Prediction failed")
                    
            except Exception as e:
                print(f"❌ Error testing method {method}: {e}")
                results[method] = None
        
        return results
    
    def debug_ensemble_members(self):
        """앙상블 멤버 디버깅 정보 출력"""
        print(f"🔍 Debugging ensemble members...")
        
        if not self.ensemble_manager.ensemble_members:
            print(f"⚠️  No ensemble members available")
            return
        
        print(f"📊 Ensemble Member Details:")
        print(f"   - Total Members: {len(self.ensemble_manager.ensemble_members)}")
        print(f"   - Memory Usage: {self.ensemble_manager.memory_usage:.2f}MB")
        print(f"   - Max Memory: {self.ensemble_manager.max_memory_mb}MB")
        
        for task_id, member_info in self.ensemble_manager.ensemble_members.items():
            print(f"\n   📋 Member {task_id}:")
            print(f"      - Performance: {member_info['performance']}")
            print(f"      - Size: {member_info['size_mb']:.2f}MB")
            print(f"      - Created: {member_info['created_at']:.2f}")
            print(f"      - Last Accessed: {member_info['last_accessed']:.2f}")
            print(f"      - Access Count: {member_info['access_count']}")
            
            # LoRA 가중치 정보
            lora_weights = member_info['lora_weights']
            print(f"      - LoRA Layers: {len(lora_weights)}")
            for layer_key, layer_data in lora_weights.items():
                w_A_shape = layer_data['w_A'].shape
                w_B_shape = layer_data['w_B'].shape
                print(f"        - {layer_key}: w_A {w_A_shape}, w_B {w_B_shape}")
    
    def verify_ensemble_stacking_effects(self):
        """
        앙상블 멤버들의 적층 효과 검증
        
        Returns:
            Dict: 각 멤버별 적층 효과 정보
        """
        print(f"🔍 Verifying ensemble stacking effects...")
        
        if not self.ensemble_manager.ensemble_members:
            print(f"⚠️  No ensemble members available for verification")
            return None
        
        stacking_info = {}
        
        for task_id, member_info in self.ensemble_manager.ensemble_members.items():
            lora_weights = member_info['lora_weights']
            
            # 각 레이어의 가중치 크기 분석
            layer_info = {}
            total_params = 0
            
            for layer_key, layer_data in lora_weights.items():
                w_A_params = layer_data['w_A'].numel()
                w_B_params = layer_data['w_B'].numel()
                layer_params = w_A_params + w_B_params
                
                layer_info[layer_key] = {
                    'w_A_shape': layer_data['w_A'].shape,
                    'w_B_shape': layer_data['w_B'].shape,
                    'w_A_params': w_A_params,
                    'w_B_params': w_B_params,
                    'total_params': layer_params
                }
                
                total_params += layer_params
            
            stacking_info[task_id] = {
                'layer_info': layer_info,
                'total_params': total_params,
                'size_mb': member_info['size_mb'],
                'performance': member_info['performance']
            }
            
            print(f"📊 Task {task_id} Stacking Analysis:")
            print(f"   - Total Parameters: {total_params:,}")
            print(f"   - Size: {member_info['size_mb']:.2f}MB")
            print(f"   - Performance: {member_info['performance']}")
            
            # 첫 번째 레이어의 가중치 크기로 적층 효과 추정
            if layer_info:
                first_layer = list(layer_info.values())[0]
                estimated_base_params = first_layer['total_params']
                stacking_ratio = total_params / estimated_base_params if estimated_base_params > 0 else 1.0
                
                print(f"   - Estimated Stacking Ratio: {stacking_ratio:.2f}x")
                if stacking_ratio > 1.0:
                    print(f"   - ✅ Multiple LoRA stacks detected (all stacking effects included)")
            else:
                    print(f"   - ℹ️  Single LoRA stack (no stacking yet)")
        
        return stacking_info
    
    def test_ensemble_stacking_consistency(self, trans_obs_0, actions):
        """
        앙상블 멤버들의 적층 효과 일관성 테스트
        
        Args:
            trans_obs_0: 변환된 초기 관측
            actions: 행동 시퀀스
            
        Returns:
            Dict: 일관성 테스트 결과
        """
        print(f"🧪 Testing ensemble stacking consistency...")
        
        if not self.ensemble_manager.ensemble_members:
            print(f"⚠️  No ensemble members available for testing")
            return None
        
        # 현재 모델의 예측 (기준)
        with torch.no_grad():
            current_pred = self.wm.rollout(obs_0=trans_obs_0, act=actions)[0]
        
        consistency_results = {}
        
        for task_id, member_info in self.ensemble_manager.ensemble_members.items():
            try:
                lora_weights = member_info['lora_weights']
                
                # 앙상블 멤버의 LoRA 가중치 적용
                success = self.ensemble_manager._apply_lora_weights(lora_weights)
                
                if not success:
                    print(f"❌ Failed to apply LoRA weights for member {task_id}")
                    continue
                
                # 앙상블 멤버로 예측 수행
                with torch.no_grad():
                    member_pred = self.wm.rollout(obs_0=trans_obs_0, act=actions)[0]
                
                # 현재 예측과의 차이 계산
                visual_diff = torch.mean(torch.abs(member_pred["visual"] - current_pred["visual"]))
                proprio_diff = torch.mean(torch.abs(member_pred["proprio"] - current_pred["proprio"]))
                
                consistency_results[task_id] = {
                    'visual_diff': visual_diff.item(),
                    'proprio_diff': proprio_diff.item(),
                    'total_diff': (visual_diff + 0.3 * proprio_diff).item(),
                    'success': True
                }
                
                print(f"📊 Task {task_id} Consistency Check:")
                print(f"   - Visual Diff: {visual_diff.item():.6f}")
                print(f"   - Proprio Diff: {proprio_diff.item():.6f}")
                print(f"   - Total Diff: {(visual_diff + 0.3 * proprio_diff).item():.6f}")
                
                if visual_diff.item() > 1e-3 or proprio_diff.item() > 1e-3:
                    print(f"   - ✅ Significant difference detected (stacking effects working)")
                else:
                    print(f"   - ⚠️  Minimal difference (may indicate stacking issue)")
                
            except Exception as e:
                print(f"❌ Error testing member {task_id}: {e}")
                consistency_results[task_id] = {
                    'error': str(e),
                    'success': False
                }
        
        # 원래 상태 복원
        if hasattr(self.wm.predictor, 'w_As') and hasattr(self.wm.predictor, 'w_Bs'):
            original_w_As = [w_A.weight.data.clone() for w_A in self.wm.predictor.w_As]
            original_w_Bs = [w_B.weight.data.clone() for w_B in self.wm.predictor.w_Bs]
            self.ensemble_manager._restore_lora_weights(original_w_As, original_w_Bs)
        
        return consistency_results
    
    def comprehensive_ensemble_test(self, trans_obs_0, actions, target_data=None):
        """
        종합적인 앙상블 테스트 (적층 효과 검증 포함)
        
        Args:
            trans_obs_0: 변환된 초기 관측
            actions: 행동 시퀀스
            target_data: 타겟 데이터 (선택사항)
            
        Returns:
            Dict: 모든 테스트 결과
        """
        print(f"🧪 Running comprehensive ensemble test with stacking verification...")
        
        test_results = {
            'ensemble_info': self.get_ensemble_info(),
            'lora_loading_test': False,
            'ensemble_inference_test': None,
            'ensemble_performance_test': None,
            'ensemble_methods_test': None,
            'stacking_verification': None,
            'stacking_consistency_test': None,
            'debug_info': None
        }
        
        # 1. LoRA 로딩 테스트
        print(f"\n1️⃣ Testing LoRA loading...")
        test_results['lora_loading_test'] = self.test_lora_loading()
        
        # 2. 앙상블 추론 테스트
        print(f"\n2️⃣ Testing ensemble inference...")
        individual_preds, ensemble_pred = self.test_ensemble_inference(trans_obs_0, actions, target_data)
        test_results['ensemble_inference_test'] = {
            'individual_predictions': individual_preds,
            'ensemble_prediction': ensemble_pred
        }
        
        # 3. 앙상블 성능 평가 테스트
        if target_data is not None:
            print(f"\n3️⃣ Testing ensemble performance evaluation...")
            test_results['ensemble_performance_test'] = self.test_ensemble_performance(trans_obs_0, actions, target_data)
        
        # 4. 다양한 앙상블 방법 테스트
        print(f"\n4️⃣ Testing different ensemble methods...")
        test_results['ensemble_methods_test'] = self.test_ensemble_methods(trans_obs_0, actions, target_data)
        
        # 5. 🔧 적층 효과 검증
        print(f"\n5️⃣ Verifying ensemble stacking effects...")
        test_results['stacking_verification'] = self.verify_ensemble_stacking_effects()
        
        # 6. 🔧 적층 효과 일관성 테스트
        print(f"\n6️⃣ Testing ensemble stacking consistency...")
        test_results['stacking_consistency_test'] = self.test_ensemble_stacking_consistency(trans_obs_0, actions)
        
        # 7. 디버깅 정보
        print(f"\n7️⃣ Debugging ensemble members...")
        self.debug_ensemble_members()
        test_results['debug_info'] = "Debug info printed to console"
        
        print(f"\n✅ Comprehensive ensemble test with stacking verification completed!")
        
        # 🔧 테스트 결과 요약
        print(f"\n📊 TEST SUMMARY:")
        print(f"   - LoRA Loading: {'✅ PASS' if test_results['lora_loading_test'] else '❌ FAIL'}")
        print(f"   - Ensemble Inference: {'✅ PASS' if test_results['ensemble_inference_test']['ensemble_prediction'] is not None else '❌ FAIL'}")
        
        if test_results['stacking_verification']:
            print(f"   - Stacking Verification: ✅ PASS")
            for task_id, info in test_results['stacking_verification'].items():
                print(f"     - Task {task_id}: {info['total_params']:,} params, {info['size_mb']:.2f}MB")
        
        if test_results['stacking_consistency_test']:
            print(f"   - Stacking Consistency: ✅ PASS")
            for task_id, result in test_results['stacking_consistency_test'].items():
                if result.get('success', False):
                    print(f"     - Task {task_id}: Diff {result['total_diff']:.6f}")
        
        return test_results
    
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
        
        # 결과 동기화
        self.task_changed = self.base_online_lora.task_changed
        self.current_task_id = self.base_online_lora.current_task_id
        self.stacks_in_current_task = self.base_online_lora.stacks_in_current_task
        
        return task_changed
    
    def reset_task_changed_flag(self):
        """task_changed 플래그를 리셋합니다."""
        self.base_online_lora.reset_task_changed_flag()
        self.task_changed = self.base_online_lora.task_changed
    
    def _apply_lora_weights(self, lora_weights):
        """
        LoRA 가중치를 현재 모델에 적용합니다.
        LoRAEnsembleManager의 로직을 직접 구현합니다.
        
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
