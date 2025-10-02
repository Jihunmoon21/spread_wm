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


    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def get_init_cond(self):
        """MPC 플래너에서 호출하는 초기 조건 반환 메서드"""
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, trajdict, action_len):
        """
        get the last observation from a trajectory dict
        """
        last_trajdict = {}
        for k in trajdict.keys():
            # 인덱스 범위 검사 및 조정
            max_len = trajdict[k].shape[1]
            safe_idx = min(action_len - 1, max_len - 1)
            last_trajdict[k] = trajdict[k][:, safe_idx]
        return last_trajdict

    def _get_traj_last(self, traj, action_len):
        """
        get the last observation from a trajectory
        """
        # 인덱스 범위 검사 및 조정
        max_len = traj.shape[1]
        safe_idx = min(action_len - 1, max_len - 1)
        return traj[:, safe_idx]

    def _mask_traj(self, traj, action_len):
        """
        mask the trajectory to only include the first action_len steps
        """
        return traj[:, :action_len]

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        """
        compute rollout metrics
        """
        logs = {}
        successes = []
        for i in range(e_state.shape[0]):
            # compute success
            success = np.linalg.norm(e_state[i] - self.state_g[i]) < 0.5
            successes.append(success)
        successes = np.array(successes)
        logs["success"] = successes
        logs["state_dist"] = np.linalg.norm(e_state - self.state_g, axis=1)
        return logs, successes


    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        
        # 액션 차원 확인 및 조정
        B, T, D = actions.shape
        F = self.frameskip
        
        # 실행용 액션 준비: [B, T, D] -> [B, T*F, D//F]
        exec_actions = rearrange(actions, "b t (f d) -> b (t f) d", f=F)
        # 정규화된 액션을 비정규화 (환경 실행용)
        print(f"[ACTION DEBUG] Before denormalize: {exec_actions[0, 0].cpu().numpy()}")
        exec_actions = self.preprocessor.denormalize_actions(exec_actions)
        print(f"[ACTION DEBUG] After denormalize: {exec_actions[0, 0].cpu().numpy()}")
        print(f"[ACTION DEBUG] Action mean: {self.preprocessor.action_mean.cpu().numpy()}")
        print(f"[ACTION DEBUG] Action std: {self.preprocessor.action_std.cpu().numpy()}")
        
        # 환경 실행
        def _env_step_one(current_state, act_np):
            """환경 한 스텝 실행. SerialVectorEnv의 개별 환경에 직접 접근."""
            if hasattr(self.env, "envs"):
                obs_list = []
                states_list = []
                for i, env in enumerate(self.env.envs):
                    if hasattr(env, "rollout"):
                        # act_np[i]는 [F, D//F] 형태
                        # env.rollout은 (T, action_dim) 형태를 기대하므로 그대로 전달
                        act_for_env = act_np[i]  # [F, D//F] 형태
                        obs_i, state_i = env.rollout(self.seed[i], current_state[i], act_for_env)
                        obs_i = {k: v[0] for k, v in obs_i.items()}  # 첫 번째 step
                        state_i = state_i[0]  # 첫 번째 step
                    elif hasattr(env, "step"):
                        try:
                            # step의 경우 첫 번째 액션만 사용
                            obs_i, reward, done, info = env.step(act_np[i][0])
                            state_i = info.get('state', current_state[i]) if isinstance(info, dict) else current_state[i]
                        except AssertionError as e:
                            if "reset" in str(e).lower():
                                env.reset()
                                obs_i, reward, done, info = env.step(act_np[i][0])
                                state_i = info.get('state', current_state[i]) if isinstance(info, dict) else current_state[i]
                            else:
                                raise
                    else:
                        raise RuntimeError(f"Environment {i} has no step or rollout method")
                    
                    obs_list.append(obs_i)
                    states_list.append(state_i)
                
                from utils import aggregate_dct
                obs = aggregate_dct(obs_list)
                states = np.stack(states_list)
                return obs, states
            
            if hasattr(self.env, "step"):
                try:
                    obs, states = self.env.step(act_np)
                    return obs, states
                except EOFError as e:
                    print(f"EOFError in env.step: {e}")
                    print(f"Action shape: {act_np.shape}, Action values: {act_np}")
                    print(f"Current state shape: {current_state.shape if hasattr(current_state, 'shape') else type(current_state)}")
                    raise
            
            raise RuntimeError("Unsupported env API: need .step(...) or individual env access")

        def _wm_predict_one(prev_obs_tensor, action_step_tensor):
            """World Model을 사용하여 다음 관측 예측."""
            # 원시 이미지인지 feature인지 확인
            visual_data = prev_obs_tensor["visual"]
            
            # 5D 형태 [B, T, H, W, C]이고 C=3이면 원시 이미지
            if len(visual_data.shape) == 5 and visual_data.shape[-1] == 3:
                # 전체 World Model 파이프라인 사용
                # World Model은 tuple을 반환: (z_pred, visual_pred, visual_reconstructed, loss, loss_components)
                z_pred, visual_pred, visual_reconstructed, loss, loss_components = self.wm(prev_obs_tensor, action_step_tensor)
                
                # z_pred에서 visual과 proprio 분리
                z_obs, z_act = self.wm.separate_emb(z_pred)
                
                return {
                    "visual": z_obs["visual"],
                    "proprio": z_obs["proprio"]
                }
            
            # 4D 형태 [B, T, H, C]이고 마지막 차원이 3이면 원시 이미지 (W 차원 누락)
            elif len(visual_data.shape) == 4 and visual_data.shape[-1] == 3:
                B, T, H, C = visual_data.shape
                # numpy 배열을 torch tensor로 변환
                if not isinstance(visual_data, torch.Tensor):
                    visual_data = torch.tensor(visual_data)
                # [B, T, H, C] -> [B, T, H, H, C] (정사각형 이미지로 가정)
                visual_data_5d = visual_data.unsqueeze(-2)  # [B, T, H, 1, C]
                visual_data_5d = visual_data_5d.expand(B, T, H, H, C)  # [B, T, H, H, C]
                
                prev_obs_5d = {
                    "visual": visual_data_5d,
                    "proprio": prev_obs_tensor["proprio"]
                }
                
                # 전체 World Model 파이프라인 사용
                z_pred, visual_pred, visual_reconstructed, loss, loss_components = self.wm(prev_obs_5d, action_step_tensor)
                
                # z_pred에서 visual과 proprio 분리
                z_obs, z_act = self.wm.separate_emb(z_pred)
                
                return {
                    "visual": z_obs["visual"],
                    "proprio": z_obs["proprio"]
                }
            
            # 4D 형태 [B, T, H, W]이고 마지막 차원이 3이 아니면 이미 feature
            elif len(visual_data.shape) == 4 and visual_data.shape[-1] != 3:
                B, T, P, D = visual_data.shape
                visual_input = visual_data.reshape(B, T * P, D)
                
                # ViT predictor로 예측
                predicted_visual = self.wm.predictor(visual_input)  # [B, (T*P), D]
                predicted_visual = predicted_visual.reshape(B, T, P, D)
                
                return {
                    "visual": predicted_visual,
                    "proprio": prev_obs_tensor["proprio"]
                }
            
            # 3D 형태 [B, P, D] - feature
            elif len(visual_data.shape) == 3:
                B, P, D = visual_data.shape
                visual_input = visual_data  # 이미 [B, P, D] 형태
                
                # ViT predictor로 예측
                predicted_visual = self.wm.predictor(visual_input)  # [B, P, D]
                
                return {
                    "visual": predicted_visual,
                    "proprio": prev_obs_tensor["proprio"]
                }
            
            # 4D 형태 [B, T, H, W]이고 두 번째 차원이 3이면 원시 이미지 (채널이 앞에 있는 경우)
            elif len(visual_data.shape) == 4 and visual_data.shape[1] == 3:
                # 전체 World Model 파이프라인 사용
                z_pred, visual_pred, visual_reconstructed, loss, loss_components = self.wm(prev_obs_tensor, action_step_tensor)
                
                # z_pred에서 visual과 proprio 분리
                z_obs, z_act = self.wm.separate_emb(z_pred)
                
                return {
                    "visual": z_obs["visual"],
                    "proprio": z_obs["proprio"]
                }
            
            else:
                raise ValueError(f"Unexpected visual data shape: {visual_data.shape}")
            

        # ------------------------------ 온라인 루프 ------------------------------
        all_e_obses_list, all_e_states_list = [], []
        current_state = self.state_0
        prev_obs = self.obs_0  # 다음 스텝 타깃 계산에 사용

        for t in range(T):
            # 현재 계획 스텝의 액션 분리
            # - WM 업데이트용: 정규화 D차원 1-step
            a_t_for_wm = actions[:, t:t+1, :]  # [B, 1, D]
            # - 환경 실행용: 비정규화 D//F차원 F-step
            a_t_for_env = exec_actions[:, t*F:(t+1)*F, :]  # [B, F, D//F]

            # 환경에서 F스텝 실행
            obs, states = _env_step_one(current_state, a_t_for_env.cpu().numpy())
            all_e_obses_list.append(obs)
            all_e_states_list.append(states)

            # LoRA 온라인 학습 (활성화된 경우)
            if self.is_lora_enabled:
                # 현재 관측을 타깃으로 사용
                true_next_obs = {k: v[:, -1:] for k, v in obs.items()}  # F번째 관측, time 차원 유지 [B, 1, ...]
                # DINOv2 임베딩을 직접 tensor로 변환 (전처리 없이)
                true_next_obs_tensor = {
                    'visual': torch.tensor(true_next_obs['visual']).to(self.device),
                    'proprio': self.preprocessor.normalize_proprios(torch.tensor(true_next_obs['proprio']).to(self.device))
                }

                # 예측 수행
                pred_next_obs = _wm_predict_one(prev_obs, a_t_for_wm)

                # 타깃도 같은 방식으로 feature space로 변환
                dummy_action = torch.zeros_like(a_t_for_wm).to(self.device)
                true_feat = _wm_predict_one(true_next_obs_tensor, dummy_action)

                # LoRA 학습 스텝
                self.wm.train()
                self.lora_optimizer.zero_grad()

                # feature space에서 loss 계산
                loss = self.workspace.loss_fn(pred_next_obs['visual'], true_feat['visual'])
                loss.backward()
                self.lora_optimizer.step()

                print(f"[LoRA Training] Step: {t+1}, Loss: {loss.item():.4f}")

            # 다음 스텝을 위한 상태 업데이트
            if len(states.shape) == 3:
                current_state = states[:, -1, :]  # F번째 상태 [B, state_dim]
            else:
                current_state = states  # 이미 [B, state_dim] 형태
            prev_obs = {k: v[:, -1:] for k, v in obs.items()}  # 다음 스텝의 prev로 사용 (time 차원 유지)

        # 전체 궤적 결합
        e_obses = aggregate_dct(all_e_obses_list)
        e_states = np.concatenate(all_e_states_list, axis=1)  # [B, T*F, state_dim]

        # 최종 상태/관측 추출
        # action_len이 None인 경우 actions의 길이로 설정
        if action_len is None:
            action_len = actions.shape[1] if len(actions.shape) > 1 else 1
        
        e_final_obs = self._get_trajdict_last(e_obses, action_len * F + 1)
        e_final_state_raw = self._get_traj_last(e_states, action_len * F + 1)
        
        # 차원 검사 및 적절한 인덱싱
        if len(e_final_state_raw.shape) == 2:
            # [B, state_dim] 형태인 경우 그대로 사용
            e_final_state = e_final_state_raw
        elif len(e_final_state_raw.shape) == 3:
            # [B, 1, state_dim] 형태인 경우 중간 차원 제거
            e_final_state = e_final_state_raw[:, 0]
        else:
            # 1차원인 경우 배치 차원 추가
            e_final_state = e_final_state_raw.reshape(1, -1) if len(e_final_state_raw.shape) == 1 else e_final_state_raw

        # 상상 궤적 생성 (World Model 예측)
        self.wm.eval()
        with torch.no_grad():
            i_z_obses, _ = self.wm.rollout(self.obs_0, actions)
            i_final_z_obs = slice_trajdict_with_t(i_z_obses, start_idx=-1, end_idx=None)

        # 평가 지표 계산
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
        )

        # ------------------------------ 시각화 ------------------------------
        if getattr(self.wm, "decoder", None) is not None:
            try:
                self.wm.eval()
                with torch.no_grad():
                    i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
                # i_visuals와 e_visuals의 길이를 맞춤
                i_visuals = self._mask_traj(i_visuals, action_len * F + 1)

                # e_visuals 처리 - 시각화용 함수 사용 (메모리 효율적)
                e_visuals = e_obses["visual"]
                if hasattr(self.preprocessor, 'transform_obs_visual_for_visualization'):
                    e_visuals_transformed = self.preprocessor.transform_obs_visual_for_visualization(e_visuals)
                else:
                    # fallback to regular transform
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
            except Exception as e:
                print(f"[WARNING] Visualization failed: {e}. Continuing without visualization.")

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
        compute rollout metrics using environment-specific eval_state functions
        """
        logs = {}
        successes = []
        distances = []
        
        for i in range(e_state.shape[0]):
            # 환경별 eval_state 함수 사용 (원본 DINO-WM 방식)
            try:
                if hasattr(self.env.envs[i], 'eval_state'):
                    # 환경의 eval_state 함수 사용
                    result = self.env.envs[i].eval_state(self.state_g[i], e_state[i])
                    successes.append(result['success'])
                    distances.append(result['state_dist'])
                    print(f"[ENV EVAL] Env {i}: success={result['success']}, dist={result['state_dist']:.4f}")
                else:
                    # fallback: 기존 방식 (위치 정보만 사용)
                    print(f"[ENV EVAL] Env {i}: No eval_state method, using fallback")
                    e_pos = e_state[i][:2] if e_state[i].shape[0] >= 2 else e_state[i]
                    g_pos = self.state_g[i][:2] if self.state_g[i].shape[0] >= 2 else self.state_g[i]
                    success = np.linalg.norm(e_pos - g_pos) < 0.5
                    distance = np.linalg.norm(e_state[i] - self.state_g[i])
                    successes.append(success)
                    distances.append(distance)
            except Exception as e:
                print(f"[ENV EVAL] Error in env {i}: {e}, using fallback")
                # fallback: 안전한 차원 처리
                e_state_safe = np.array(e_state[i])
                g_state_safe = np.array(self.state_g[i])
                
                # 최소 공통 차원으로 맞춤
                min_dim = min(len(e_state_safe), len(g_state_safe))
                e_state_trunc = e_state_safe[:min_dim]
                g_state_trunc = g_state_safe[:min_dim]
                
                # 위치 기반 성공 판단 (처음 2차원 또는 사용 가능한 차원)
                pos_dim = min(2, min_dim)
                e_pos = e_state_trunc[:pos_dim]
                g_pos = g_state_trunc[:pos_dim]
                success = np.linalg.norm(e_pos - g_pos) < 0.5
                
                # 전체 거리 계산 (안전한 차원으로)
                distance = np.linalg.norm(e_state_trunc - g_state_trunc)
                
                print(f"[ENV EVAL] Fallback - e_dim: {len(e_state_safe)}, g_dim: {len(g_state_safe)}, min_dim: {min_dim}")
                print(f"[ENV EVAL] Fallback - success: {success}, distance: {distance:.4f}")
                
                successes.append(success)
                distances.append(distance)
        
        successes = np.array(successes)
        distances = np.array(distances)
        logs["success"] = successes
        logs["state_dist"] = distances
        
        # 결과 요약 출력
        success_rate = np.mean(successes)
        avg_distance = np.mean(logs["state_dist"])
        print(f"[PLANNING RESULTS] Success Rate: {success_rate:.2%} ({np.sum(successes)}/{len(successes)})")
        print(f"[PLANNING RESULTS] Average Distance to Goal: {avg_distance:.4f}")
        print(f"[PLANNING RESULTS] Individual Successes: {successes}")
        
        return logs, successes

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename=""
    ):
        """
        plot rollout comparison
        """
        # subsample for plotting
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        successes = successes[: self.n_plot_samples]

        # get the number of columns
        n_columns = e_visuals.shape[1]

        # get the number of rows
        n_rows = e_visuals.shape[0] * 2  # 2 rows per sample (e and i)

        # create a figure
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 2, n_rows * 2))

        # plot the rollouts
        for i in range(e_visuals.shape[0]):
            for j in range(n_columns):
                # plot the environment rollout
                axes[i * 2, j].imshow(e_visuals[i, j].permute(1, 2, 0))
                axes[i * 2, j].set_title(f"Env {i} Step {j}")
                axes[i * 2, j].axis("off")

                # plot the imagination rollout
                axes[i * 2 + 1, j].imshow(i_visuals[i, j].permute(1, 2, 0))
                axes[i * 2 + 1, j].set_title(f"Imag {i} Step {j}")
                axes[i * 2 + 1, j].axis("off")

        # save the figure
        plt.tight_layout()
        plt.savefig(f"{filename}.png")
        plt.close()

        # save video if requested
        if save_video:
            # create a video
            video_frames = []
            for j in range(n_columns):
                frame = torch.cat(
                    [
                        torch.cat([e_visuals[i, j] for i in range(e_visuals.shape[0])], dim=1)
                        for i in range(e_visuals.shape[0])
                    ],
                    dim=2,
                )
                video_frames.append(frame.permute(1, 2, 0).numpy())

            # save the video
            imageio.mimsave(f"{filename}.gif", video_frames, fps=5)

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename=""
    ):
        """
        plot rollout comparison
        """
        # subsample for plotting
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        successes = successes[: self.n_plot_samples]

        # get the number of columns
        n_columns = e_visuals.shape[1]

        # 길이가 다른 경우 처리 (정상적인 동작)
        if i_visuals.shape[1] != n_columns:
            min_length = min(i_visuals.shape[1], n_columns)
            print(f"[INFO] Aligning rollout lengths for visualization: {e_visuals.shape[1]} (env) and {i_visuals.shape[1]} (model). Using min length: {min_length}")
            e_visuals = e_visuals[:, :min_length]
            i_visuals = i_visuals[:, :min_length]
            n_columns = min_length

        # get the number of rows
        n_rows = e_visuals.shape[0] * 2  # 2 rows per sample (e and i)

        # plot the rollouts
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]
            n_columns = e_visuals.shape[1]

        # add goal visual - 차원에 맞게 repeat
        goal_visual_shape = self.obs_g["visual"].shape
        e_visuals_shape = e_visuals.shape
        
        # goal_visual을 e_visuals의 배치 크기에 맞게 repeat
        if len(goal_visual_shape) == 5:  # [1, 1, C, H, W]
            goal_visual = self.obs_g["visual"].repeat(e_visuals_shape[0], 1, 1, 1, 1)
        elif len(goal_visual_shape) == 4:  # [1, C, H, W]
            goal_visual = self.obs_g["visual"].unsqueeze(1).repeat(e_visuals_shape[0], 1, 1, 1, 1)
        else:
            # 다른 차원의 경우 안전하게 처리
            goal_visual = self.obs_g["visual"].expand(e_visuals_shape[0], 1, *goal_visual_shape[1:])
        correction = torch.zeros_like(goal_visual)
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        # i_visuals = torch.cat([i_visuals.cpu(), goal_visual], dim=1)
        rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)

        rollout = rollout.permute(0, 1, 3, 4, 2)  # (b, t, h, w, c)
        rollout = rearrange(rollout, "b t h w c -> (b h) (t w) c")

        utils.save_image(
            rollout.permute(2, 0, 1),
            f"{filename}.png",
            nrow=rollout.shape[1] // (n_columns + 1),
            normalize=True,
            value_range=(0, 1),
        )