import torch
import time
from collections import deque
from utils import move_to_device # spread_wm의 유틸리티 함수

class OnlineLora:
    """
    Online-LoRA 학습과 관련된 모든 로직을 전담하는 클래스.
    - 옵티마이저, 손실 함수 등 학습 관련 객체를 내부적으로 생성하고 관리합니다.
    - Loss Window를 통해 학습 안정성을 높이고, LoRA 적층 시점을 판단합니다.
    """
    def __init__(self, workspace):
        """
        OnlineLora 모듈을 초기화합니다.
        
        Args:
            workspace (PlanWorkspace): 필요한 모든 객체(wm, cfg 등)에 접근할 수 있는 상위 워크스페이스.
        """
        self.workspace = workspace
        self.wm = workspace.wm
        self.cfg = workspace.cfg_dict.get("lora", {}) # lora 관련 설정만 가져옴
        self.device = next(self.wm.parameters()).device

        # --- 1. 학습에 필요한 객체들 초기화 ---
        self.is_online_lora = self.cfg.get("online", False) # lora 적층 등을 포함하는 전체 온라인 기능 활성화 여부
        
        # Loss 가중치 설정 (설정 파일에 없으므로 기본값 사용)
        self.visual_loss_weight = self.cfg.get("visual_loss_weight", 1.0)
        self.proprio_loss_weight = self.cfg.get("proprio_loss_weight", 0.3)

        # 기존 모델 파라미터 고정
        print("INFO: Freezing all non-LoRA parameters for online learning.")
        for name, param in self.wm.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
        
        # 학습 대상 LoRA 파라미터 필터링 및 옵티마이저 생성
        params_to_train = [p for p in self.wm.parameters() if p.requires_grad]
        if not params_to_train:
            raise ValueError("No trainable LoRA parameters found. Check if LoRA wrappers are correctly applied.")
        
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.cfg.get("lr", 1e-4))
        self.loss_fn = torch.nn.MSELoss()

        # --- 2. Online-LoRA (적층) 관련 변수 초기화 ---
        # 하이브리드 적층 설정 (online 모드 여부와 관계없이 항상 초기화)
        self.hybrid_config = self.cfg.get("hybrid_stacking", {})
        self.hybrid_enabled = self.hybrid_config.get("enabled", False)
        self.task_based_stacking = self.hybrid_config.get("task_based_stacking", False)
        self.loss_based_stacking = self.hybrid_config.get("loss_based_stacking", False)
        self.max_stacks_per_task = self.hybrid_config.get("max_stacks_per_task", 3)
        self.stack_type_tracking = self.hybrid_config.get("stack_type_tracking", True)
        
        if self.is_online_lora:
            print("INFO: Initializing Online LoRA module for dynamic stacking.")
            self.loss_window = deque(maxlen=self.cfg.get("loss_window_length", 10))
            self.mean_threshold = self.cfg.get("loss_window_mean_threshold", 0.01)
            self.variance_threshold = self.cfg.get("loss_window_variance_threshold", 1e-5)
            self.min_steps_for_stack = self.cfg.get("min_steps_for_stack", 50)
            
            self.steps_since_last_stack = 0
            self.new_peak_detected = True
            self.last_loss_mean = float('inf')
            self.last_loss_var = float('inf')
            
            # 태스크별 적층 추적
            self.stacks_in_current_task = 0
            self.current_task_id = 0
            self.task_changed = False  # 태스크 전환 감지 플래그
            self.stack_history = []  # 적층 히스토리 (타입, 시점, Loss 등)
            
            if self.hybrid_enabled:
                print(f"INFO: Hybrid stacking enabled - Task-based: {self.task_based_stacking}, Loss-based: {self.loss_based_stacking}")
                print(f"INFO: Max stacks per task: {self.max_stacks_per_task}")
        else:
            # online 모드가 아닌 경우에도 기본값으로 초기화
            self.stacks_in_current_task = 0
            self.current_task_id = 0
            self.stack_history = []
            
        # 마지막 Loss 값 저장용 (태스크 전환을 위해)
        self.last_loss = None
        self.last_visual_loss = None
        self.last_proprio_loss = None
        
        # LoRA 적층 콜백 함수 (태스크 추적을 위해)
        self.on_lora_stack_callback = None
        
        # 🔧 LoRA adaptation time 추적 (최종 결과 출력용)
        self.adaptation_times = []


    def update(self, trans_obs_0, actions, e_obses):
        """
        하나의 학습 단계를 수행하는 메인 메소드. PlanEvaluator로부터 호출됩니다.
        """
        # 학습 단계 수행 (예측, 손실 계산, 역전파, 업데이트)
        start_time = time.time()
        total_loss_value = self._perform_training_step(trans_obs_0, actions, e_obses)
        adaptation_time = time.time() - start_time
        print(f"LoRA adaptation time: {adaptation_time:.4f} seconds")
        
        # 🔧 adaptation time 저장 (최종 결과 출력용)
        self.adaptation_times.append(adaptation_time)
        
        # 마지막 Loss 값 저장 (태스크 전환을 위해)
        if total_loss_value is not None:
            self.last_loss = total_loss_value
        
        # 학습이 성공적으로 이루어졌고, online_lora 기능이 활성화되었다면
        if total_loss_value is not None and self.is_online_lora:
            # 하이브리드 적층 로직
            if self.hybrid_enabled:
                self._manage_hybrid_stacking(total_loss_value)
            else:
                # 기존 Loss 기반 적층 로직
                self._manage_loss_window_and_stacking(total_loss_value)


    def _perform_training_step(self, trans_obs_0, actions, e_obses):
        """실제 예측, 손실 계산, 역전파 및 업데이트를 수행합니다."""
        try:
            print("--- Starting LoRA Online Learning ---")
            
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
            print("Computing loss...")
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
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            backward_time = time.time() - backward_start
            
            print(f"LoRA step timing - Rollout: {rollout_time:.4f}s, Encode: {encode_time:.4f}s, Loss: {loss_time:.4f}s, Backward: {backward_time:.4f}s")

            loss_value = total_loss.item()
            self.last_visual_loss = visual_loss.item()
            self.last_proprio_loss = proprio_loss.item()

            planner_iter = None
            if getattr(self.workspace, "planner", None) is not None:
                planner_iter = getattr(self.workspace.planner, "iter", None)

            if (
                getattr(self.workspace, "is_training_mode", True)
                and planner_iter == 1
                and getattr(self.workspace, "forward_transfer_metrics", None) is None
            ):
                self.workspace.forward_transfer_metrics = {
                    "total_loss": loss_value,
                    "visual_loss": self.last_visual_loss,
                    "proprio_loss": self.last_proprio_loss,
                    "iteration": planner_iter,
                }

            return loss_value

        except Exception as e:
            print(f"Error during training step: {e}")
            return None
        
        finally:
            # 5. 메모리 정리 (오류 발생 여부와 관계없이 실행)
            # del을 위해 변수가 선언되었는지 확인
            if 'i_z_obses_pred' in locals(): del i_z_obses_pred
            if 'i_z_obses_gt' in locals(): del i_z_obses_gt
            if 'total_loss' in locals(): del total_loss
            torch.cuda.empty_cache()
            print("--- LoRA Online Update Complete ---")
    # LoRA 파라미터 변화량 확인 코드 추가 가능

    def _manage_loss_window_and_stacking(self, current_loss_value):
        """Loss Window를 업데이트하고, LoRA를 추가할지 여부를 결정합니다."""
        self.loss_window.append(current_loss_value)
        self.steps_since_last_stack += 1
        
        if len(self.loss_window) < self.loss_window.maxlen:
            return

        loss_tensor = torch.tensor(list(self.loss_window))
        mean = torch.mean(loss_tensor)
        var = torch.var(loss_tensor)
        
        print(f"Loss Window >> Mean: {mean:.6f}, Variance: {var:.6f}")
        print(f"Steps since last stack: {self.steps_since_last_stack}/{self.min_steps_for_stack}")
        print(f"New peak detected: {self.new_peak_detected}")

        # `engine.py`의 Loss Peak 감지 로직
        if not self.new_peak_detected and mean > self.last_loss_mean + torch.sqrt(self.last_loss_var):
            self.new_peak_detected = True
            print("Loss peak detected after a plateau period.")

        # LoRA 적층 트리거 조건
        if (self.new_peak_detected and
            mean < self.mean_threshold and
            var < self.variance_threshold and
            self.steps_since_last_stack > self.min_steps_for_stack):
            
            # 최대 적층 횟수 확인
            if self.stacks_in_current_task >= self.max_stacks_per_task:
                print(f"⚠️  Max stacks per task ({self.max_stacks_per_task}) reached. Skipping loss-based stacking.")
                return
            
            print("! Loss plateau detected. Triggering LoRA stacking process !")
            
            # Loss 기반 적층 수행
            success = self._perform_lora_stacking("loss_based", self.current_task_id, "loss_plateau")
            
            if success:
                # 적층 횟수 증가
                self.stacks_in_current_task += 1
                
                # 적층 히스토리 기록
                if self.hybrid_enabled and self.stack_type_tracking:
                    self.stack_history.append({
                        'type': 'loss_based',
                        'task_id': self.current_task_id,
                        'reason': 'loss_plateau',
                        'step': self.steps_since_last_stack,
                        'loss_mean': mean,
                        'loss_var': var,
                        'timestamp': time.time()
                    })
                
                print(f"   - Loss mean: {mean:.6f} (threshold: {self.mean_threshold})")
                print(f"   - Loss variance: {var:.6f} (threshold: {self.variance_threshold})")
                print(f"   - Steps since last stack: {self.steps_since_last_stack}")
                print(f"   - Stacks in current task: {self.stacks_in_current_task}/{self.max_stacks_per_task}")
            
            # 상태 변수 초기화
            self.new_peak_detected = False
            self.last_loss_mean = mean
            self.last_loss_var = var
            self.steps_since_last_stack = 0
            self.loss_window.clear()

    def _manage_hybrid_stacking(self, current_loss_value):
        """
        하이브리드 적층 로직: 태스크 기반과 Loss 기반 적층을 모두 고려합니다.
        """
        # Loss 기반 적층 (기존 로직)
        if self.loss_based_stacking:
            self._manage_loss_window_and_stacking(current_loss_value)
    
    def trigger_task_based_stacking(self, task_id, reason="task_change"):
        """
        태스크 기반 적층을 트리거합니다.
        
        Args:
            task_id: 새로운 태스크 ID
            reason: 적층 이유 ("task_change", "manual", etc.)
        """
        # hybrid_enabled 속성이 없거나 False인 경우 적층하지 않음
        if not hasattr(self, 'hybrid_enabled') or not self.hybrid_enabled or not self.task_based_stacking:
            return False
            
        # 태스크가 변경된 경우
        if task_id != self.current_task_id:
            self.current_task_id = task_id
            self.stacks_in_current_task = 0
            print(f"🔄 Task changed to {task_id}. Resetting stack counter.")
        
        # 최대 적층 횟수 확인
        if self.stacks_in_current_task >= self.max_stacks_per_task:
            print(f"⚠️  Max stacks per task ({self.max_stacks_per_task}) reached. Skipping task-based stacking.")
            return False
        
        # 태스크 기반 적층 수행
        print(f"🎯 Task-based LoRA stacking triggered (Task {task_id}, Reason: {reason})")
        success = self._perform_lora_stacking("task_based", task_id, reason)
        
        if success:
            self.stacks_in_current_task += 1
            self.steps_since_last_stack = 0
            
            # 적층 히스토리 기록
            if self.stack_type_tracking:
                self.stack_history.append({
                    'type': 'task_based',
                    'task_id': task_id,
                    'reason': reason,
                    'step': self.steps_since_last_stack,
                    'timestamp': time.time()
                })
        
        return success
    
    def compute_loss_only(self, trans_obs_0, actions, e_obses):
        """
        온라인 학습을 수행하지 않고 현재 모델의 손실을 계산합니다.
        """
        try:
            with torch.no_grad():
                i_z_obses_pred, _ = self.wm.rollout(obs_0=trans_obs_0, act=actions)
                trans_obs_gt = self.workspace.data_preprocessor.transform_obs(e_obses)
                trans_obs_gt = move_to_device(trans_obs_gt, self.device)
                i_z_obses_gt = self.wm.encode_obs(trans_obs_gt)

                frameskip = self.workspace.frameskip
                gt_proprio_resampled = i_z_obses_gt["proprio"][:, ::frameskip, :]
                gt_visual_resampled = i_z_obses_gt["visual"][:, ::frameskip, :, :]

                proprio_loss = self.loss_fn(i_z_obses_pred["proprio"], gt_proprio_resampled)
                visual_loss = self.loss_fn(i_z_obses_pred["visual"], gt_visual_resampled)
                total_loss = self.visual_loss_weight * visual_loss + self.proprio_loss_weight * proprio_loss

                metrics = {
                    "visual_loss": float(visual_loss.item()),
                    "proprio_loss": float(proprio_loss.item()),
                    "total_loss": float(total_loss.item()),
                }

                self.last_visual_loss = metrics["visual_loss"]
                self.last_proprio_loss = metrics["proprio_loss"]
                self.last_loss = metrics["total_loss"]

                return metrics
        except Exception as e:
            print(f"Error computing evaluation loss: {e}")
            return None
        finally:
            if 'i_z_obses_pred' in locals():
                del i_z_obses_pred
            if 'i_z_obses_gt' in locals():
                del i_z_obses_gt
    
    def _perform_lora_stacking(self, stack_type, task_id, reason):
        """
        실제 LoRA 적층을 수행합니다.
        
        Args:
            stack_type: "task_based" 또는 "loss_based"
            task_id: 태스크 ID
            reason: 적층 이유
        
        Returns:
            bool: 적층 성공 여부
        """
        try:
            print(f"Performing {stack_type} LoRA stacking...")
            
            # --- LoRA 적층 수행 ---
            self.wm.predictor.update_and_reset_lora_parameters()
            
            # --- 옵티마이저 재설정 ---
            self._reset_optimizer_for_new_lora()
            
            # 적층 완료 로그
            print(f"{stack_type.title()} LoRA stacking completed successfully!")
            print(f"   - Task ID: {task_id}")
            print(f"   - Reason: {reason}")
            print(f"   - Stacks in current task: {self.stacks_in_current_task + 1}/{self.max_stacks_per_task}")
            
            # 🔧 LoRA 적층 콜백 호출 (앙상블 저장을 위해 확장된 정보 전달)
            if self.on_lora_stack_callback:
                self.on_lora_stack_callback(
                    steps=self.steps_since_last_stack, 
                    loss=self.last_loss,
                    task_id=task_id,
                    stack_type=stack_type,
                    reason=reason
                )
            
            return True
            
        except Exception as e:
            print(f"❌ Error during {stack_type} LoRA stacking: {e}")
            return False

    def _reset_optimizer_for_new_lora(self):
        """적층 후 새로운 wnew 파라미터들만 학습하도록 옵티마이저를 재설정합니다."""
        # 현재 학습 가능한 모든 파라미터를 가져옴
        params_to_train = [p for p in self.wm.parameters() if p.requires_grad]
        
        if not params_to_train:
            print("Warning: No trainable parameters found after LoRA stacking.")
            return
            
        # 새로운 옵티마이저 생성 (기존 옵티마이저의 상태는 유지하지 않음)
        old_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer = torch.optim.Adam(params_to_train, lr=old_lr)
        
        print(f"Optimizer reset: {len(params_to_train)} trainable parameters, lr={old_lr}")
    
    def check_task_change(self, new_task_id):
        """
        태스크 전환을 감지하고 task_changed 플래그를 설정합니다.
        
        Args:
            new_task_id (int): 새로운 태스크 ID
            
        Returns:
            bool: 태스크가 실제로 변경되었는지 여부
        """
        if not self.is_online_lora:
            return False
            
        if new_task_id != self.current_task_id:
            self.current_task_id = new_task_id
            self.stacks_in_current_task = 0
            self.task_changed = True
            return True
        else:
            self.task_changed = False
            return False
    
    def reset_task_changed_flag(self):
        """task_changed 플래그를 리셋합니다."""
        self.task_changed = False