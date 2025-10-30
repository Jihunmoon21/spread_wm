import os
import torch
import imageio
import numpy as np
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils


class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
        # lora 관련 인자 추가 (기본값으로 호환성 유지)
        is_lora_enabled=False,
        is_online_lora=False,
        workspace=None,
        obs_g_traj=None,  # 추가: 목표 궤적 (다중 이미지 골용)
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.obs_g_traj = obs_g_traj  # 추가
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device

        self.plot_full = False  # plot all frames or frames after frameskip

        # lora 학습을 위한 관련 설정 초기화
        self.workspace = workspace  # PlanWorkspace 인스턴스 참조
        self.is_lora_enabled = self.workspace.is_lora_enabled if self.workspace is not None else is_lora_enabled
        self.is_online_lora = self.workspace.is_online_lora if self.workspace is not None else is_online_lora
        # workspace로부터 optimizer와 loss_fn 가져오기
        if self.is_lora_enabled:
            # 앙상블 사용 여부 확인 (online_learner가 있고 ensemble_manager가 있으면 True)
            try:
                lora_ensemble_enabled = (
                    hasattr(self.workspace, 'online_learner') and
                    hasattr(self.workspace.online_learner, 'ensemble_manager') and
                    self.workspace.online_learner.ensemble_manager is not None
                )
            except Exception:
                lora_ensemble_enabled = False
            print(
                f"LoRA enabled: {self.is_lora_enabled}, Online LoRA: {self.is_online_lora}, LoRA Ensemble: {lora_ensemble_enabled}"
            )
            
            # LoRA 파라미터 상태 확인
            if hasattr(self.wm, 'predictor') and hasattr(self.wm.predictor, 'lora_vit'):
                total_params = sum(p.numel() for p in self.wm.parameters())
                trainable_params = sum(p.numel() for p in self.wm.parameters() if p.requires_grad)
                print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,} ({(trainable_params / total_params) * 100 if total_params > 0 else 0:.4f}%)")
                
            else:
                print("Warning: LoRA predictor not found in world model")
            
        #     # LoRA 파라미터 변화량 추적을 위한 변수 초기화
        #     self._prev_lora_params = None

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
        # rollout in wm
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(self.obs_g), self.device
        )
        with torch.no_grad():
            i_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # rollout in env
        # 각 action을 frameskip번 반복 (planner는 원본 action_dim으로 생성)
        exec_actions = repeat(
            actions.cpu(), "b t d -> b (t f) d", f=self.frameskip
        )
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()

        # state_0가 None이면 각 환경에 None 전달 (환경 기본 reset 사용)
        if self.state_0 is None:
            init_states = [None] * len(exec_actions)
        else:
            init_states = self.state_0
        
        rollout_result = self.env.rollout(self.seed, init_states, exec_actions)
        
        # rollout이 infos도 반환하는지 확인 (LIBERO task completion check 포함)
        if len(rollout_result) == 3:
            e_obses, e_states, e_infos = rollout_result
        else:
            e_obses, e_states = rollout_result
            e_infos = None
        # # ======================================================= #
        # LoRA 학습이 활성화된 경우, 학습 책임을 OnlineLora 객체에 위임합니다.
        if self.is_lora_enabled and self.workspace.online_learner is not None:
            self.workspace.online_learner.update(trans_obs_0, actions, e_obses)
        # if self.is_lora_enabled:
        #     print("--- Starting LoRA Online Learning ---")
            
        #     # 1. 예측: 월드 모델을 사용하여 동일한 행동으로 미래를 '예측'합니다.
        #     #    (그래디언트 계산이 활성화된 상태에서 실행)
        #     # print("Step 1: Running world model rollout with gradients enabled...")
        #     torch.cuda.empty_cache()
        #     i_z_obses_pred, _ = self.wm.rollout(
        #         obs_0=trans_obs_0,
        #         act=actions,
        #     )

        #     # 2. 정답 준비: 실제 환경 결과(e_obses)를 인코딩하여 '정답' 잠재 상태를 만듭니다.
        #     # print("Step 2: Encoding ground truth observations...")
        #     with torch.no_grad():
        #         trans_obs_gt = self.preprocessor.transform_obs(e_obses)
        #         trans_obs_gt = move_to_device(trans_obs_gt, self.device)
        #         i_z_obses_gt = self.wm.encode_obs(trans_obs_gt)

        #     # 3. 손실 계산: 예측과 정답 사이의 오차(MSE Loss)를 계산합니다.
        #     # .detach()를 사용하여 정답값으로부터는 그래디언트가 흐르지 않도록 함)
        #     print("Computing loss...")
        #     # 실제 궤적을 self.frameskip 간격으로 샘플링(slicing)하여 시점을 통일합니다.
        #     gt_proprio_resampled = i_z_obses_gt["proprio"][:, ::self.frameskip, :].detach()
        #     gt_visual_resampled = i_z_obses_gt["visual"][:, ::self.frameskip, :, :].detach()
            
        #     # 시각과 proprioceptive 손실을 각각 계산
        #     proprio_loss = self.workspace.loss_fn(i_z_obses_pred["proprio"], gt_proprio_resampled)
        #     visual_loss = self.workspace.loss_fn(i_z_obses_pred["visual"], gt_visual_resampled)
 
        #     # 가중합으로 전체 손실 계산 (설정 가능한 가중치 사용)
        #     visual_weight = self.workspace.visual_loss_weight
        #     proprio_weight = self.workspace.proprio_loss_weight
        #     loss = visual_weight * visual_loss + proprio_weight * proprio_loss
            
        #     print(f"Visual loss: {visual_loss.item():.6f}, Proprio loss: {proprio_loss.item():.6f}")
        #     print(f"Total loss: {loss.item():.6f}")

        #     # 4. 역전파 및 업데이트: 계산된 손실을 바탕으로 LoRA 가중치를 업데이트합니다.
        #     # print("Step 4: Backpropagation and parameter update...")
        #     self.lora_optimizer.zero_grad()
            
        #     # 그래디언트 계산
        #     loss.backward()
            
        #     # 그래디언트 크기 체크
        #     total_grad_norm = 0
        #     trainable_params = 0
        #     for param in self.wm.parameters():
        #         if param.requires_grad and param.grad is not None:
        #             total_grad_norm += param.grad.data.norm(2).item() ** 2
        #             trainable_params += 1
            
        #     total_grad_norm = total_grad_norm ** 0.5
        #     print(f"Gradient norm: {total_grad_norm:.6f}, Trainable params: {trainable_params}")
            
        #     # 파라미터 업데이트
        #     self.lora_optimizer.step()
            
        #     # 메모리 정리
        #     del i_z_obses_pred, i_z_obses_gt, trans_obs_gt
        #     torch.cuda.empty_cache()
        #     print(f"--- LoRA Online Update Complete ---")
            
            # 5. 추가 검증: LoRA 파라미터 변화량 확인
            # if hasattr(self, '_prev_lora_params') and self._prev_lora_params is not None:
            #     # 현재 학습 가능한(requires_grad=True) 모든 파라미터를 가져옴
            #     current_lora_params = [
            #         p.clone() for p in self.wm.predictor.parameters() if p.requires_grad
            #     ]
                
            #     if len(self._prev_lora_params) != len(current_lora_params):
            #         print("Warning: Number of trainable parameters changed. Skipping parameter change check for this step.")
                
            #     else:
            #         param_changes = []
            #         all_shapes_match = True
            #         for prev, curr in zip(self._prev_lora_params, current_lora_params):
            #             if prev.shape != curr.shape:
            #                 print(f"Warning: Parameter shape mismatch. Prev: {prev.shape}, Curr: {curr.shape}. Skipping check.")
            #                 all_shapes_match = False
            #                 break # 모양이 하나라도 다르면 비교 중단
                        
            #             change = torch.norm(curr - prev).item()
            #             param_changes.append(change)

            #         if all_shapes_match and param_changes:
            #             avg_param_change = sum(param_changes) / len(param_changes)
            #             print(f"Average LoRA parameter change: {avg_param_change:.8f}")
            
            # params_to_save = [
            #     p.clone() for p in self.wm.predictor.parameters() if p.requires_grad
            # ]
            # self._prev_lora_params = params_to_save
            
            # # 메모리 정리
            # del params_to_save
            # if 'current_lora_params' in locals():
            #     del current_lora_params
            
        # LoRA 학습 완료 후 최종 메모리 정리
        # torch.cuda.empty_cache()
        # ======================================================= #
        e_visuals = e_obses["visual"]
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        # 월드 모델 예측 오류 계산: 예측 임베딩 vs 실제 관측 임베딩
        # 실제 환경 관측을 인코딩
        trans_e_obses = move_to_device(self.preprocessor.transform_obs(e_obses), self.device)
        e_z_obses = self.wm.encode_obs(trans_e_obses)
        
        # Frameskip 적용: 환경에서는 frameskip번 실행했지만, 월드 모델은 1번만 예측
        # 따라서 환경 관측을 frameskip 간격으로 리샘플링
        e_z_obses_resampled = {
            "visual": e_z_obses["visual"][:, ::self.frameskip, :, :],
            "proprio": e_z_obses["proprio"][:, ::self.frameskip, :],
        }
        
        # 월드 모델 예측 오류 계산
        wm_visual_error = torch.nn.functional.mse_loss(
            i_z_obses["visual"][:, -1:, :, :], 
            e_z_obses_resampled["visual"][:, -1:, :, :]
        ).item()
        
        wm_proprio_error = torch.nn.functional.mse_loss(
            i_z_obses["proprio"][:, -1:, :], 
            e_z_obses_resampled["proprio"][:, -1:, :]
        ).item() if i_z_obses.get("proprio") is not None else 0
        
        print(f"World Model Prediction Error: visual={wm_visual_error:.6f}, proprio={wm_proprio_error:.6f}")
        
        # compute eval metrics
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
            e_infos=e_infos,  # LIBERO task completion check 포함
            wm_visual_error=wm_visual_error,  # 월드 모델 예측 오류 추가
            wm_proprio_error=wm_proprio_error,
        )
        

        # plot trajs
        if self.wm.decoder is not None:
            i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
            i_visuals = self._mask_traj(
                i_visuals, action_len + 1
            )  # we have action_len + 1 states
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )

        return logs, successes, e_obses, e_states

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs, e_infos=None, wm_visual_error=None, wm_proprio_error=None):
        """
        Args
            e_state
            e_obs
            i_z_obs
            e_infos: rollout infos (LIBERO task completion check 포함)
        Return
            logs
            successes
        """
        # 우선순위: infos의 success > state 기반 평가 > fallback
        if e_infos is not None and 'success' in e_infos:
            # ✅ BEST: LIBERO의 task completion check 사용
            # infos['success']는 (n_evals, T) 형태, 마지막 step의 success 사용
            successes = e_infos['success'][:, -1]  # (n_evals,)
            
            logs = {
                'success_rate': np.mean(successes.astype(float))
            }
            print(f"Success rate (LIBERO task check): {logs['success_rate']:.3f}")
            print(f"Successes: {successes}")
            
        elif self.state_g is not None:
            # State 기반 평가
            eval_results = self.env.eval_state(self.state_g, e_state)
            successes = eval_results['success']

            logs = {
                f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
                for key, value in eval_results.items()
            }

            print("Success rate (state-based): ", logs['success_rate'])
            print(eval_results)
        else:
            # Fallback
            print("Warning: No goal state or infos available.")
            successes = np.zeros(len(e_state), dtype=bool)
            logs = {}

        # Visual distance 계산 (크기가 다를 수 있으므로 리사이즈)
        e_visual = e_obs["visual"]  # (B, T, H1, W1, C)
        g_visual = self.obs_g["visual"]  # (B, T, H2, W2, C)
        
        # 크기가 다르면 goal 크기에 맞춤
        if e_visual.shape[2:4] != g_visual.shape[2:4]:
            from skimage.transform import resize
            b, t = e_visual.shape[:2]
            target_h, target_w = g_visual.shape[2:4]
            
            # Reshape to (B*T, H, W, C) for resize
            e_visual_flat = e_visual.reshape(-1, *e_visual.shape[2:])
            e_visual_resized = []
            for img in e_visual_flat:
                img_resized = resize(img, (target_h, target_w, 3), preserve_range=True, anti_aliasing=True)
                e_visual_resized.append(img_resized)
            e_visual = np.array(e_visual_resized).reshape(b, t, target_h, target_w, 3)
        
        # (B, T, H, W, C) 차이를 flatten해서 L2 norm 계산
        diff = e_visual - g_visual  # (B, T, H, W, C)
        b, t = diff.shape[:2]
        diff_flat = diff.reshape(b, t, -1)  # (B, T, H*W*C)
        visual_dists = np.linalg.norm(diff_flat, axis=2)  # (B, T)
        visual_dists = np.mean(visual_dists, axis=1)  # (B,) - average over time
        mean_visual_dist = np.mean(visual_dists)
        
        # Proprio distance 계산 (joint_pos + gripper_qpos: 9차원)
        if e_obs["proprio"].shape == self.obs_g["proprio"].shape:
            proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=2)  # (B, T)
            proprio_dists = np.mean(proprio_dists, axis=1)  # (B,)
            mean_proprio_dist = np.mean(proprio_dists)
        else:
            # 차원 불일치 - 디버그용 경고
            print(f"Warning: Proprio dimension mismatch - env: {e_obs['proprio'].shape}, goal: {self.obs_g['proprio'].shape}. Skipping proprio distance.")
            mean_proprio_dist = 0.0

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        e_z_obs = self.wm.encode_obs(e_obs)
        div_visual_emb = torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()
        div_proprio_emb = torch.norm(e_z_obs["proprio"] - i_z_obs["proprio"]).item()

        logs.update({
            "mean_visual_dist": mean_visual_dist,
            "mean_proprio_dist": mean_proprio_dist,
            "mean_div_visual_emb": div_visual_emb,
            "mean_div_proprio_emb": div_proprio_emb,
        })
        
        # 월드 모델 예측 오류 추가
        if wm_visual_error is not None:
            logs["wm_visual_error"] = wm_visual_error
            logs["wm_proprio_error"] = wm_proprio_error

        return logs, successes

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename=""
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, h, w, c)
        i_visuals: (b, t, h, w, c)
        goal: (b, h, w, c)
        """
        # 데이터 원본 프레임 수를 plan_output과 무관하게 표기
        if self.workspace is not None and hasattr(self.workspace, "data_traj_lens"):
            try:
                print(f"[Data] dataset traj_lens (env frames): {self.workspace.data_traj_lens}")
            except Exception:
                pass
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]

        # i_visuals의 frameskip padding 먼저 처리
        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )  # pad i_visuals (due to frameskip)
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        correction = 0.3  # to distinguish env visuals and imagined visuals
        
        # 최종 프레임 수 결정 (모든 subsample/마스킹 처리 후)
        n_columns = e_visuals.shape[1]
        # 디버그: 실제 데이터 프레임 수 로깅
        try:
            print(f"[EvalPlot] frames -> env:{e_visuals.shape[1]}, imag:{i_visuals.shape[1]}, n_columns:{n_columns}")
        except Exception:
            pass
        
        # 목표 이미지 준비: obs_g_traj가 있으면 다중 목표, 없으면 단일 목표
        if self.obs_g_traj is not None and self.obs_g_traj["visual"].shape[1] > 1:
            # 다중 목표: 각 프레임 위치에 해당하는 목표 이미지 매핑
            goal_visual_traj = self.obs_g_traj["visual"][: self.n_plot_samples]
            goal_visual_traj = self.preprocessor.transform_obs_visual(goal_visual_traj)  # (B, N_goals, C, H, W)
            
            # goal_frame_indices 또는 goal_indices를 workspace에서 가져오기
            goal_indices = None
            if self.workspace is not None:
                goal_indices = self.workspace.cfg_dict.get('objective', {}).get('goal_indices', None)
                if goal_indices is None:
                    goal_indices = self.workspace.cfg_dict.get('goal_frame_indices', None)
            
            n_goals = goal_visual_traj.shape[1]
            try:
                print(f"[EvalPlot] goals -> n_goals:{n_goals}, goal_indices:{goal_indices}")
            except Exception:
                pass
            
            # 최종 프레임 수(n_columns)를 기준으로 목표 이미지 매핑
            goal_visual_per_frame = []
            goal_mapping_debug = []  # 디버깅용
            for frame_idx in range(n_columns):
                # action step 계산: goal_indices는 action step 기준
                # plot_full=False: subsample 후 frame_idx = action_step (1:1 매핑)
                # plot_full=True: padding만 하고 subsample 안 함, frame_idx = action_step * frameskip
                if self.plot_full and self.frameskip > 1:
                    # plot_full=True일 때는 environment frame이므로 action step으로 변환
                    action_step = frame_idx // self.frameskip
                else:
                    # plot_full=False일 때는 이미 action step과 1:1 매핑
                    action_step = frame_idx
                
                if goal_indices is not None and len(goal_indices) == n_goals:
                    # goal_indices를 기반으로 구간별 매핑 (action step 기준)
                    # 음수 인덱스(-1)는 예측 궤적의 마지막(action step = n_columns - 1)을 의미
                    # 구간별 매핑: 각 목표가 특정 구간 동안 유지되도록
                    goal_idx_mapped = 0
                    
                    # 실제 goal_idx 값 계산 (음수 인덱스 처리)
                    actual_goal_indices = []
                    for goal_idx in goal_indices:
                        if goal_idx < 0:
                            actual_goal_indices.append(n_columns - 1)
                        else:
                            actual_goal_indices.append(goal_idx)
                    
                    # action_step에 따라 적절한 목표 선택
                    if action_step < actual_goal_indices[0]:
                        # 첫 번째 목표 이전: 첫 번째 목표 사용
                        goal_idx_mapped = 0
                    elif action_step >= actual_goal_indices[-1]:
                        # 마지막 목표 이후: 마지막 목표 사용
                        goal_idx_mapped = n_goals - 1
                    else:
                        # 중간 구간: 해당 구간의 목표 선택
                        for i in range(len(actual_goal_indices) - 1):
                            if actual_goal_indices[i] <= action_step < actual_goal_indices[i + 1]:
                                goal_idx_mapped = i + 1
                                break
                        else:
                            # 예외 케이스: 마지막 구간
                            goal_idx_mapped = n_goals - 1
                    
                    closest_goal_idx = goal_idx_mapped
                else:
                    # goal_indices가 없거나 길이가 다르면 등간격으로 매핑
                    closest_goal_idx = min(frame_idx * n_goals // n_columns, n_goals - 1)
                
                goal_visual_per_frame.append(goal_visual_traj[:, closest_goal_idx:closest_goal_idx+1])
                goal_mapping_debug.append((frame_idx, action_step, closest_goal_idx))
            
            # (B, T, C, H, W) 형태로 변환
            goal_visual = torch.cat(goal_visual_per_frame, dim=1)
            try:
                print(f"[EvalPlot] goal_visual_len:{goal_visual.shape[1]}")
                # 목표 이미지 매핑 정보 출력
                mapping_summary = {}
                for frame_idx, action_step, goal_idx in goal_mapping_debug:
                    if goal_idx not in mapping_summary:
                        mapping_summary[goal_idx] = []
                    mapping_summary[goal_idx].append(f"frame{frame_idx}(step{action_step})")
                mapping_str = ', '.join([f'goal{i}: [{" ".join(mapping_summary[i])}]' for i in sorted(mapping_summary.keys())])
                print(f"[EvalPlot] Goal mapping: {mapping_str}")
            except Exception:
                pass
        else:
            # 단일 목표 (기존 로직)
            goal_visual = self.obs_g["visual"][: self.n_plot_samples]
            goal_visual = self.preprocessor.transform_obs_visual(goal_visual)  # (B, 1, C, H, W)
            try:
                print(f"[EvalPlot] goal_visual_len(single):{goal_visual.shape[1]}")
            except Exception:
                pass

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    i_obs = i_visuals[idx, i, ...]
                    # goal_visual이 (B, T, C, H, W) 형태면 해당 프레임 사용, (B, 1, C, H, W)면 첫 번째 사용
                    goal_frame = goal_visual[idx, min(i, goal_visual.shape[1]-1)] if goal_visual.ndim == 5 else goal_visual[idx, 0]
                    e_obs = torch.cat(
                        [e_obs.cpu(), goal_frame - correction], dim=2
                    )
                    i_obs = torch.cat(
                        [i_obs.cpu(), goal_frame - correction], dim=2
                    )
                    frame = torch.cat([e_obs - correction, i_obs], dim=1)
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    frames.append(frame)
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    video_writer.append_data(
                        (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    )
                video_writer.close()

        # n_columns는 이미 위에서 계산됨
        assert (
            i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"
        assert (
            goal_visual.shape[1] == n_columns or goal_visual.shape[1] == 1
        ), f"Goal visual length {goal_visual.shape[1]} does not match n_columns {n_columns}"

        # add goal columns: 각 프레임에 해당하는 목표 이미지를 디코더 자리(i_visuals)에 표시
        if goal_visual.ndim == 5 and goal_visual.shape[1] == n_columns:
            # 다중 목표: 각 프레임마다 해당하는 목표 이미지를 디코더 자리(i_visuals)에 표시
            goal_visual_transformed = goal_visual.cpu() - correction  # (B, T, C, H, W)
            
            # 실제 환경 이미지는 그대로, 디코더 자리에는 목표 이미지 표시
            e_visuals = e_visuals.cpu()
            i_visuals = goal_visual_transformed  # 디코더 자리를 목표 이미지로 대체
        elif goal_visual.ndim == 5 and goal_visual.shape[1] == 1:
            # 단일 목표 (B, 1, C, H, W) 형태
            goal_visual_transformed = goal_visual.cpu() - correction
            # 모든 프레임에 동일한 목표 이미지 표시
            goal_visual_repeated = goal_visual_transformed.repeat(1, n_columns, 1, 1, 1)
            e_visuals = e_visuals.cpu()
            i_visuals = goal_visual_repeated  # 디코더 자리를 목표 이미지로 대체
        else:
            # 단일 목표 (기존 로직): (B, 1, C, H, W) 또는 (B, C, H, W)
            goal_visual_expanded = goal_visual if goal_visual.ndim == 5 else goal_visual.unsqueeze(1)
            if goal_visual_expanded.shape[1] == 1:
                goal_visual_repeated = goal_visual_expanded.repeat(1, n_columns, 1, 1, 1)
                e_visuals = e_visuals.cpu()
                i_visuals = goal_visual_repeated[:, :, :, :, :] - correction
            else:
                e_visuals = torch.cat([e_visuals.cpu(), goal_visual_expanded[:, 0:1] - correction], dim=1)
                i_visuals = torch.cat([i_visuals.cpu(), goal_visual_expanded[:, 0:1] - correction], dim=1)
                n_columns += 1
        
        rollout = torch.cat([e_visuals - correction, i_visuals], dim=1)

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}.png",
            nrow=n_columns,  # nrow is the number of columns
            normalize=True,
            value_range=(-1, 1),
        )