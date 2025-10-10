import torch
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
        print("INFO: Freezing all non-LoRA parameters for online learning...")
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


    def update(self, trans_obs_0, actions, e_obses):
        """
        하나의 학습 단계를 수행하는 메인 메소드. PlanEvaluator로부터 호출됩니다.
        """
        # 학습 단계 수행 (예측, 손실 계산, 역전파, 업데이트)
        total_loss_value = self._perform_training_step(trans_obs_0, actions, e_obses)
        
        # 학습이 성공적으로 이루어졌고, online_lora 기능이 활성화되었다면
        if total_loss_value is not None and self.is_online_lora:
            # Loss Window를 관리하고 LoRA 적층 여부를 판단
            self._manage_loss_window_and_stacking(total_loss_value)


    def _perform_training_step(self, trans_obs_0, actions, e_obses):
        """실제 예측, 손실 계산, 역전파 및 업데이트를 수행합니다."""
        try:
            print("--- Starting LoRA Online Learning ---")
            
            # 1. 예측 (그래디언트 활성화)
            i_z_obses_pred, _ = self.wm.rollout(obs_0=trans_obs_0, act=actions)

            # 2. 정답 준비 (그래디언트 비활성화)
            with torch.no_grad():
                trans_obs_gt = self.workspace.preprocessor.transform_obs(e_obses)
                trans_obs_gt = move_to_device(trans_obs_gt, self.device)
                i_z_obses_gt = self.wm.encode_obs(trans_obs_gt)

            # 3. 손실 계산
            print("Computing loss...")
            frameskip = self.workspace.frameskip
            gt_proprio_resampled = i_z_obses_gt["proprio"][:, ::frameskip, :].detach()
            gt_visual_resampled = i_z_obses_gt["visual"][:, ::frameskip, :, :].detach()
            
            proprio_loss = self.loss_fn(i_z_obses_pred["proprio"], gt_proprio_resampled)
            visual_loss = self.loss_fn(i_z_obses_pred["visual"], gt_visual_resampled)
            
            total_loss = self.visual_loss_weight * visual_loss + self.proprio_loss_weight * proprio_loss
            
            print(f"Visual loss: {visual_loss.item():.6f}, Proprio loss: {proprio_loss.item():.6f}")
            print(f"Total loss: {total_loss.item():.6f}")

            # 4. 역전파 및 업데이트
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            return total_loss.item()

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
            
            print("! Loss plateau detected. Triggering LoRA stacking process !")
            
            # --- LoRA 적층 수행 ---
            print("Performing LoRA stacking...")
            self.wm.predictor.update_and_reset_lora_parameters()
            
            # --- 옵티마이저 재설정 (새로운 wnew 파라미터들만 학습) ---
            print("Resetting optimizer for new LoRA parameters...")
            self._reset_optimizer_for_new_lora()
            
            # 적층 완료 로그
            print("LoRA stacking completed successfully!")
            print(f"   - Loss mean: {mean:.6f} (threshold: {self.mean_threshold})")
            print(f"   - Loss variance: {var:.6f} (threshold: {self.variance_threshold})")
            print(f"   - Steps since last stack: {self.steps_since_last_stack}")
            
            # 상태 변수 초기화
            self.new_peak_detected = False
            self.last_loss_mean = mean
            self.last_loss_var = var
            self.steps_since_last_stack = 0
            self.loss_window.clear()

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