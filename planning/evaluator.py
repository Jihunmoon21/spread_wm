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
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
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
        self._initial_cd_measured = False  # Flag to track if initial CD has been measured

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
        
        # 🔧 last_index를 실제 trajectory 길이로 클램핑
        if isinstance(traj_data, torch.Tensor):
            max_index = traj_data.shape[1] - 1
            last_index = np.clip(last_index, 0, max_index)
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            max_index = traj_data.shape[1] - 1
            last_index = np.clip(last_index, 0, max_index)
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
        self,
        actions,
        action_len=None,
        filename="output",
        save_video=False,
        learning_enabled=True,
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
        exec_actions = rearrange(
            actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()

        # 🔧 final output 계산 시 재센터링 강제 적용 (태스크별 파일명 포함, failure 포함)
        if filename.startswith("output_final"):
            print("[EVAL] Setting force_recenter flag for final output - will recenter in reset() and set_states()")
            # 🔧 force_recenter_for_next_rollout()를 호출하여 카운터 리셋 + _force_recenter_unlimited 플래그 설정
            self.force_recenter_for_next_rollout()

        # Set flag before rollout to measure Initial CD after first set_states in prepare()
        if not hasattr(self, '_initial_cd_measured') or not self._initial_cd_measured:
            # Set flags in environment to measure CD after set_states
            if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
                # For vectorized environment, set flag in first worker
                first_env = self.env.envs[0].env
                if hasattr(first_env, 'set_measure_initial_cd'):
                    first_env.set_measure_initial_cd(True, self.state_g)
            elif hasattr(self.env, 'env'):
                # For single environment
                if hasattr(self.env.env, 'set_measure_initial_cd'):
                    self.env.env.set_measure_initial_cd(True, self.state_g)
            self._initial_cd_measured = True

        # 🔧 final output인 경우 플래그를 rollout 직전에 다시 확인 및 설정 (태스크별 파일명 포함)
        force_recenter_for_rollout = False
        if filename.startswith("output_final"):
            print("[EVAL] Re-checking force_recenter flag before rollout")
            force_recenter_for_rollout = True
            if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
                for i, env_wrapper in enumerate(self.env.envs):
                    # FlexEnvWrapper에 직접 설정
                    env_wrapper._force_recenter_after_set_states = True
                    # 확인용 로그
                    flag_after = getattr(env_wrapper, '_force_recenter_after_set_states', None)
                    print(f"[EVAL] Re-set _force_recenter_after_set_states=True for env[{i}], verified={flag_after}")
        
        # 🔧 final output인 경우 force_recenter 파라미터를 직접 전달
        e_obses, e_states = self.env.rollout(
            self.seed,
            self.state_0,
            exec_actions,
            force_recenter=force_recenter_for_rollout,
        )
        # # ======================================================= #
        # LoRA 학습이 활성화된 경우, 학습 책임을 OnlineLora 객체에 위임합니다.
        if (
            learning_enabled
            and self.is_lora_enabled
            and self.workspace.online_learner is not None
        ):
            self.workspace.online_learner.update(trans_obs_0, actions, e_obses)
            # online_learner에서 계산한 loss를 logs에 추가하기 위해 임시로 저장
            if hasattr(self.workspace.online_learner, 'last_visual_loss'):
                self._computed_visual_loss = self.workspace.online_learner.last_visual_loss
            if hasattr(self.workspace.online_learner, 'last_proprio_loss'):
                self._computed_proprio_loss = self.workspace.online_learner.last_proprio_loss
            if hasattr(self.workspace.online_learner, 'last_loss'):
                self._computed_total_loss = self.workspace.online_learner.last_loss
        else:
            # LoRA가 비활성화되어 있어도 loss를 계산하고 출력 (로그 확인용)
            if self.workspace is not None:
                # 위에서 계산한 i_z_obses를 재사용
                i_z_obses_pred = i_z_obses
                
                # 2. 정답 준비: 실제 환경 결과(e_obses)를 인코딩하여 '정답' 잠재 상태를 만듭니다.
                with torch.no_grad():
                    trans_obs_gt = self.preprocessor.transform_obs(e_obses)
                    trans_obs_gt = move_to_device(trans_obs_gt, self.device)
                    i_z_obses_gt = self.wm.encode_obs(trans_obs_gt)

                    # 3. 손실 계산: 예측과 정답 사이의 오차(MSE Loss)를 계산합니다.
                    print("Computing loss...")
                    frameskip = self.frameskip
                    gt_proprio_resampled = i_z_obses_gt["proprio"][:, ::frameskip, :]
                    gt_visual_resampled = i_z_obses_gt["visual"][:, ::frameskip, :, :]
                    
                    # Loss function 가져오기
                    if hasattr(self.workspace, 'loss_fn') and self.workspace.loss_fn is not None:
                        loss_fn = self.workspace.loss_fn
                    else:
                        import torch.nn as nn
                        loss_fn = nn.MSELoss()
                    
                    # 시각과 proprioceptive 손실을 각각 계산
                    proprio_loss = loss_fn(i_z_obses_pred["proprio"], gt_proprio_resampled)
                    visual_loss = loss_fn(i_z_obses_pred["visual"], gt_visual_resampled)
     
                    # 가중합으로 전체 손실 계산 (기본값 사용)
                    visual_weight = getattr(self.workspace, 'visual_loss_weight', 1.0)
                    proprio_weight = getattr(self.workspace, 'proprio_loss_weight', 0.3)
                    total_loss = visual_weight * visual_loss + proprio_weight * proprio_loss
                    
                    print(f"Visual loss: {visual_loss.item():.6f}, Proprio loss: {proprio_loss.item():.6f}")
                    print(f"Total loss: {total_loss.item():.6f}")
                    
                    # logs에 추가하기 위해 임시로 저장
                    self._computed_visual_loss = float(visual_loss.item())
                    self._computed_proprio_loss = float(proprio_loss.item())
                    self._computed_total_loss = float(total_loss.item())
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

        # compute eval metrics
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
        )
        
        # 계산한 visual_loss를 logs에 추가 (learning_enabled 여부와 관계없이)
        if self.workspace is not None and hasattr(self, '_computed_visual_loss'):
            logs_to_add = {}
            if self._computed_visual_loss is not None:
                logs_to_add["visual_loss"] = self._computed_visual_loss
            if hasattr(self, '_computed_proprio_loss') and self._computed_proprio_loss is not None:
                logs_to_add["proprio_loss"] = self._computed_proprio_loss
            if hasattr(self, '_computed_total_loss') and self._computed_total_loss is not None:
                logs_to_add["total_loss"] = self._computed_total_loss
            if logs_to_add:
                logs.update(logs_to_add)
            # 임시 변수 정리
            delattr(self, '_computed_visual_loss')
            if hasattr(self, '_computed_proprio_loss'):
                delattr(self, '_computed_proprio_loss')
            if hasattr(self, '_computed_total_loss'):
                delattr(self, '_computed_total_loss')

        # plot/save rollouts
        if save_video:
            if self.wm.decoder is not None:
                i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
                i_visuals = self._mask_traj(
                    i_visuals, action_len + 1
                )  # we have action_len + 1 states
                e_visuals_t = self.preprocessor.transform_obs_visual(e_visuals)
                e_visuals_t = self._mask_traj(
                    e_visuals_t, action_len * self.frameskip + 1
                )
                self._plot_rollout_compare(
                    e_visuals=e_visuals_t,
                    i_visuals=i_visuals,
                    successes=successes,
                    save_video=True,
                    filename=filename,
                )
            else:
                # decoder가 없으면 환경 영상만 저장
                self._plot_rollout_env_only(
                    e_visuals=e_visuals,
                    successes=successes,
                    save_video=True,
                    filename=filename,
                )

        return logs, successes, e_obses, e_states

    def force_recenter_for_next_rollout(self):
        """Force environments to recenter on the very next rollout."""
        def _force_wrapper(wrapper):
            if hasattr(wrapper, "force_recenter_next_rollout"):
                wrapper.force_recenter_next_rollout()
            else:
                if hasattr(wrapper, "_force_recenter_after_set_states"):
                    wrapper._force_recenter_after_set_states = True
                inner_env = getattr(wrapper, "env", None)
                if inner_env is not None:
                    if hasattr(inner_env, "reset_recentering_counters"):
                        inner_env.reset_recentering_counters()
                    if hasattr(inner_env, "_force_recenter_after_set_states"):
                        inner_env._force_recenter_after_set_states = True

        if hasattr(self.env, "envs"):
            for env_wrapper in self.env.envs:
                _force_wrapper(env_wrapper)
        elif hasattr(self.env, "workers"):
            for worker in self.env.workers:
                env_wrapper = getattr(worker, "env", None)
                if env_wrapper is not None:
                    _force_wrapper(env_wrapper)
        else:
            _force_wrapper(self.env)

        if hasattr(self, "_initial_cd_measured"):
            self._initial_cd_measured = False

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        """
        Args
            e_state
            e_obs
            i_z_obs
        Return
            logs
            successes
        """
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
            for key, value in eval_results.items()
        }

        print("Success rate: ", logs['success_rate'])
        print(eval_results)

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)
        proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=1)
        mean_proprio_dist = np.mean(proprio_dists)

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
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )  # pad i_visuals (due to frameskip)
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    i_obs = i_visuals[idx, i, ...]
                    e_obs = torch.cat(
                        [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    i_obs = torch.cat(
                        [i_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
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

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        n_columns = e_visuals.shape[1]
        assert (
            i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

        # add a goal column
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        i_visuals = torch.cat([i_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)
        n_columns += 1

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

    def _plot_rollout_env_only(self, e_visuals, successes, save_video=False, filename=""):
        """
        디코더가 없을 때, 환경에서 관측된 프레임(e_visuals)과 목표 이미지를 이용해
        PNG/MP4를 저장합니다.
        """
        # 시각 데이터 정규화 및 샘플 수 제한
        e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
        e_visuals = e_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        correction = 0.3

        # 비디오 저장: 각 샘플별로 실행 프레임과 목표 한 장을 좌우로 붙여 저장
        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    frame = torch.cat(
                        [e_obs.cpu() - correction, goal_visual[idx, 0] - correction],
                        dim=2,
                    )
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
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

        # 이미지 그리드 저장: 마지막 프레임과 목표 이미지를 좌우로 붙여 한 장으로 저장
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
        last_frames = e_visuals[:, -1]  # (b, c, h, w)
        goal_frames = goal_visual[:, 0]  # (b, c, h, w)
        rollout = torch.cat([last_frames.cpu() - correction, goal_frames - correction], dim=2)
        utils.save_image(
            rollout,
            f"{filename}.png",
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )