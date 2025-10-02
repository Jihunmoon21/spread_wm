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
    stack_trajdict,
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
        # lora 관련 인자 추가
        is_lora_enabled,
        is_online_lora,
        workspace,
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

        # lora 학습을 위한 관련 설정 초기화
        self.is_lora_enabled = is_lora_enabled
        self.is_online_lora = is_online_lora
        self.workspace = workspace  # PlanWorkspace 인스턴스 참조

        # action_dim을 workspace에서 안전하게 가져오기 (fallback 포함)
        if self.workspace is not None and hasattr(self.workspace, "action_dim"):
            self.action_dim = self.workspace.action_dim
        else:
            # 최후 수단: wm이나 preprocessor에서 유추 (환경별로 다를 수 있음)
            self.action_dim = getattr(getattr(self.wm, "action_dim", None), "__int__", lambda: None)() or 0
            
        # workspace로부터 optimizer와 loss_fn 가져오기
        if self.is_lora_enabled:
            self.lora_optimizer = self.workspace.lora_optimizer
            self.loss_fn = self.workspace.loss_fn

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
        actions: detached torch tensors on cuda, shape [B, T, F*D]
        Returns
            metrics, and feedback from env
        """
        # ------------------------------ 기본 준비 ------------------------------
        B, T, FD = actions.shape
        F = self.frameskip
        
        assert FD % F == 0, f"Action dimension {FD} is not divisible by frameskip {F}"
        D = FD // F
        
        if action_len is None:
            action_len = np.full(B, np.inf)

        # 월드모델 비교용 rollout (추론 전용)
        self.wm.eval()
        with torch.no_grad():
            trans_obs_0 = move_to_device(self.preprocessor.transform_obs(self.obs_0), self.device)
            trans_obs_g = move_to_device(self.preprocessor.transform_obs(self.obs_g), self.device)
            i_z_obses, _ = self.wm.rollout(obs_0=trans_obs_0, act=actions)  # [B, T+1, ...] 가정
        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # 환경 실행용 액션 전개
        # - Env에는 비정규화된 D차원 액션을 frameskip만큼 순차 공급
        exec_actions = rearrange(actions.cpu(), "b t (f d) -> b t f d", f=F)  # [B, T, F, D]
        exec_actions = self.preprocessor.denormalize_actions(exec_actions.reshape(B, T * F, D))
        exec_actions = exec_actions.numpy().reshape(B, T, F, D)

        # ------------------------------ 헬퍼 ------------------------------
        def _env_step_one(current_state, act_np):
            """환경 한 스텝 실행. 다양한 API를 보수적으로 지원."""
            # 선호: Gym 스타일 VectorEnv
            if hasattr(self.env, "step"):
                # act_np: [B, D]
                obs, states = self.env.step(act_np)
                return obs, states
            # 커스텀: rollout_one_step(current_state, act)
            if hasattr(self.env, "rollout_one_step"):
                return self.env.rollout_one_step(current_state, act_np)
            # 레거시: rollout(seed, state, actions_seq)
            if hasattr(self.env, "rollout"):
                # 1-step용으로 감싸서 호출 (seed는 가능한 외부에서 설정되어 있어야 함)
                seed = getattr(self, "seed", None)
                return self.env.rollout(seed, current_state, act_np[None, ...])
            raise RuntimeError("Unsupported env API: need .step(...) or .rollout_one_step(...) or .rollout(...)")

        def _wm_predict_one(prev_obs_tensor, action_step_tensor):
            """WM 단일 스텝 예측: predict_one_step 우선, 없으면 rollout(H=1)로 대체."""
            # action_step_tensor: [B, D] on device
            if hasattr(self.wm, "predict_one_step"):
                return self.wm.predict_one_step(prev_obs_tensor, action_step_tensor)  # dict of features
            # fallback: rollout with H=1 (act needs shape [B, 1, D])
            pred_seq, _ = self.wm.rollout(obs_0=prev_obs_tensor, act=action_step_tensor.unsqueeze(1))
            # pred_seq가 dict(traj)라고 가정: 각 키에 대해 마지막 시점 추출
            return {k: v[:, -1] for k, v in pred_seq.items()}

        # ------------------------------ 온라인 루프 ------------------------------
        all_e_obses_list, all_e_states_list = [], []
        current_state = self.state_0
        prev_obs = self.obs_0  # 다음 스텝 타깃 계산에 사용

        for t in range(T):
            # 현재 계획 스텝의 액션 분리
            # - WM 업데이트용: 정규화 D차원 1-step
            a_t_for_wm = actions[:, t, :].reshape(B, F, D)[:, 0, :]             # [B, D], 정규화 상태 그대로
            # - Env 실행용: 비정규화 D차원 F-step
            a_tf_for_env = exec_actions[:, t, :, :]                              # [B, F, D] (numpy, denorm)

            # ----------- Env: frameskip만큼 실제 실행 -----------
            obs_seq, state_seq = [], []
            for f in range(F):
                obs_f, state_f = _env_step_one(current_state, a_tf_for_env[:, f, :])
                obs_seq.append(obs_f)
                state_seq.append(state_f)
                current_state = state_f  # 다음 프레임의 초기 상태

            # [B, F, ...] 형태로 정리
            obs = stack_trajdict(obs_seq)                 # 구현체에 맞춘 유틸: e.g., dict of arrays [B, F, ...]
            states = np.stack(state_seq, axis=1)          # [B, F, state_dim]

            # ----------- LoRA 온라인 업데이트 -----------
            if getattr(self, "is_lora_enabled", False):
                self.wm.train()
                self.lora_optimizer.zero_grad()

                # (a) 입력/타깃 준비
                prev_obs_tensor = move_to_device(self.preprocessor.transform_obs(prev_obs), self.device)
                true_next_obs_tensor = move_to_device(self.preprocessor.transform_obs(obs[:, -1]), self.device)  # F번째 관측

                # (b) 예측/타깃 특징 추출 (타깃 인코딩은 no_grad + eval)
                pred_feat = _wm_predict_one(prev_obs_tensor, a_t_for_wm.to(self.device))

                with torch.no_grad():
                    self.wm.eval()
                    true_feat = self.wm.encode_obs(true_next_obs_tensor)

                # (c) 손실 계산
                loss_v = self.loss_fn(pred_feat["visual"],  true_feat["visual"])
                loss_p = self.loss_fn(pred_feat["proprio"], true_feat["proprio"])
                loss = loss_v + loss_p

                # (d) 역전파/업데이트
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.wm.parameters() if p.requires_grad], max_norm=1.0
                )
                self.wm.train()
                self.lora_optimizer.step()
                self.wm.eval()  # 이후 추론 일관성
                # 매 스텝(t)마다 LoRA 학습 손실 값을 출력합니다.
                
                print(f"[LoRA Training] Step: {t}, Loss: {loss.item():.4f}")
                # ==========================================================

                self.wm.train()
                self.lora_optimizer.step()
                self.wm.eval()

                # (e) Online LoRA plateau 감지 (컨트롤러 우선)
                if getattr(self, "is_online_lora", False):
                    if hasattr(self, "online_lora"):
                        # 권장: 컨트롤러 객체에 위임
                        self.online_lora.on_step(loss.item(), getattr(self.wm, "predictor", self.wm))
                    elif hasattr(self, "workspace"):
                        # 레거시: workspace의 loss_window 사용
                        ws = self.workspace
                        if not hasattr(ws, "loss_window"):
                            from collections import deque
                            ws.loss_window = deque(maxlen=getattr(ws, "loss_window_length", 5))
                        ws.loss_window.append(loss.item())
                        mean, var = np.mean(ws.loss_window), np.var(ws.loss_window)
                        if (not getattr(ws, "new_peak_detected", True)) and mean > getattr(ws, "last_loss_window_mean", 1e9) + np.sqrt(getattr(ws, "last_loss_window_variance", 1e9)):
                            ws.new_peak_detected = True
                        if mean < getattr(ws, "mean_threshold", 5.6) and var < getattr(ws, "variance_threshold", 0.08) and getattr(ws, "new_peak_detected", True):
                            print("INFO: Loss plateau detected. Updating and resetting Online LoRA weights.")
                            predictor = getattr(self.wm, "predictor", self.wm)
                            if hasattr(predictor, "update_and_reset_lora_parameters"):
                                predictor.update_and_reset_lora_parameters()
                            ws.last_loss_window_mean, ws.last_loss_window_variance = mean, var
                            ws.new_peak_detected = False

            # ----------- 버퍼 업데이트 -----------
            all_e_obses_list.append(obs)         # [B, F, ...]
            all_e_states_list.append(states)     # [B, F, ...]
            prev_obs = obs[:, -1]                # 다음 스텝의 prev로 사용

        # ------------------------------ 결과 정리 ------------------------------
        e_obses = concat_trajdict(all_e_obses_list)             # [B, T*F, ...] 형태로 합치도록 구현
        e_states = np.concatenate(all_e_states_list, axis=1)    # [B, T*F, state_dim]
        e_visuals = e_obses["visual"]

        e_final_obs   = self._get_trajdict_last(e_obses, action_len * F + 1)
        e_final_state = self._get_traj_last(e_states, action_len * F + 1)[:, 0]

        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
        )

        # ------------------------------ 시각화 ------------------------------
        if getattr(self.wm, "decoder", None) is not None:
            self.wm.eval()
            with torch.no_grad():
                i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
            i_visuals = self._mask_traj(i_visuals, action_len + 1)

            e_visuals_transformed = self.preprocessor.transform_obs_visual(e_visuals)
            e_visuals_transformed = self._mask_traj(
                e_visuals_transformed, action_len * F + 1
            )
            self._plot_rollout_compare(
                e_visuals=e_visuals_transformed,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )

        return logs, successes, e_obses, e_states


        # e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        # e_visuals = e_obses["visual"]
        # e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        # e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
        #     :, 0
        # ]  # reduce dim back

        # # compute eval metrics
        # logs, successes = self._compute_rollout_metrics(
        #     e_state=e_final_state,
        #     e_obs=e_final_obs,
        #     i_z_obs=i_final_z_obs,
        # )

        # # plot trajs
        # if self.wm.decoder is not None:
        #     i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
        #     i_visuals = self._mask_traj(
        #         i_visuals, action_len + 1
        #     )  # we have action_len + 1 states
        #     e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
        #     e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
        #     self._plot_rollout_compare(
        #         e_visuals=e_visuals,
        #         i_visuals=i_visuals,
        #         successes=successes,
        #         save_video=save_video,
        #         filename=filename,
        #     )

        # return logs, successes, e_obses, e_states

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
