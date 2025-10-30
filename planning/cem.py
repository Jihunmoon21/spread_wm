import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def plan(self, obs_0, obs_g, obs_g_traj=None, actions=None):
        """
        Args:
            obs_g: This is the obs_g_traj mentioned in the prompt.
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        # Evaluator에서 LoRA 파라미터 변화량 추적을 위한 변수 초기화
        if hasattr(self.evaluator, '_prev_lora_params'):
            self.evaluator._prev_lora_params = None

        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        # 목표는 obs_g_traj가 제공되면 그것을 사용하고, 아니면 obs_g 사용
        target_obs = obs_g_traj if obs_g_traj is not None else obs_g
        trans_target = move_to_device(
            self.preprocessor.transform_obs(target_obs), self.device
        )
        # obs_g_traj(다중 프레임) 또는 obs_g(단일 프레임)를 인코딩
        z_obs_g_traj = self.wm.encode_obs(trans_target)
        # ------------------------------------------

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            print(f"CEM Step {i+1}/{self.opt_steps}")
            # optimize individual instances
            losses = []
            
            # 예측 오류 추적용 변수 추가
            pred_errors_visual = []
            pred_errors_proprio = []
            
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                
                # --- 수정된 부분 2: 미리 인코딩된 z_obs_g_traj 사용 ---
                cur_z_obs_g_traj = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g_traj.items() # 기존 z_obs_g 대신 z_obs_g_traj 사용
                }
                # -----------------------------------------------
                
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )

                # --- 수정된 부분 3: objective_fn에 cur_z_obs_g_traj 전달 ---
                loss = self.objective_fn(i_z_obses, cur_z_obs_g_traj) # 기존 cur_z_obs_g 대신 cur_z_obs_g_traj 사용
                # ----------------------------------------------------
                
                # 예측 오류 계산 (첫 번째 샘플만, mu[traj]를 사용)
                if traj == 0:  # 첫 번째 trajectory만 계산 (모두 동일)
                    pred_visual = i_z_obses["visual"][0, -1:]  # (1, 1, P, D)
                    goal_visual = cur_z_obs_g_traj["visual"][0, -1:]  # (1, 1, P, D)
                    # MSE와 L2 distance 둘 다 계산
                    visual_error_mse = torch.nn.functional.mse_loss(pred_visual, goal_visual).item()
                    visual_error_l2 = torch.norm(pred_visual - goal_visual).item()
                    pred_errors_visual.append((visual_error_mse, visual_error_l2))
                    
                    if 'proprio' in i_z_obses and i_z_obses['proprio'] is not None and 'proprio' in cur_z_obs_g_traj:
                        pred_proprio = i_z_obses["proprio"][0, -1:]
                        goal_proprio = cur_z_obs_g_traj["proprio"][0, -1:]
                        proprio_error_mse = torch.nn.functional.mse_loss(pred_proprio, goal_proprio).item()
                        proprio_error_l2 = torch.norm(pred_proprio - goal_proprio).item()
                        pred_errors_proprio.append((proprio_error_mse, proprio_error_l2))

                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                # 아래 두 줄 삭제하면 cem 작동 안 함
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            mean_loss = np.mean(losses)
            min_loss = np.min(losses)
            
            # 예측 오류 로깅 추가 (MSE + L2)
            if pred_errors_visual:
                # MSE와 L2 값 분리
                visual_mse_values = [x[0] for x in pred_errors_visual]
                visual_l2_values = [x[1] for x in pred_errors_visual]
                mean_visual_error_mse = np.mean(visual_mse_values)
                mean_visual_error_l2 = np.mean(visual_l2_values)
                
                print(f"  → Loss: mean={mean_loss:.4f}, min={min_loss:.4f}")
                print(f"  → Visual Distance (L2): {mean_visual_error_l2:.4f}")
                print(f"  → Pred Error: visual={mean_visual_error_mse:.4f}", end="")
                if pred_errors_proprio:
                    proprio_mse_values = [x[0] for x in pred_errors_proprio]
                    proprio_l2_values = [x[1] for x in pred_errors_proprio]
                    mean_proprio_error_mse = np.mean(proprio_mse_values)
                    mean_proprio_error_l2 = np.mean(proprio_l2_values)
                    print(f", proprio={mean_proprio_error_mse:.4f}")
                else:
                    print()
            else:
                print(f"  → Loss: mean={mean_loss:.4f}, min={min_loss:.4f}")
            
            # wandb 로깅에도 추가
            log_dict = {f"{self.logging_prefix}/loss": mean_loss, "step": i + 1}
            if pred_errors_visual:
                log_dict[f"{self.logging_prefix}/pred_error_visual_mse"] = mean_visual_error_mse
                log_dict[f"{self.logging_prefix}/pred_error_visual_l2"] = mean_visual_error_l2
                if pred_errors_proprio:
                    log_dict[f"{self.logging_prefix}/pred_error_proprio_mse"] = mean_proprio_error_mse
                    log_dict[f"{self.logging_prefix}/pred_error_proprio_l2"] = mean_proprio_error_l2
            self.wandb_run.log(log_dict)
            # 마지막 step에만 이미지 저장
            if self.evaluator is not None and i == self.opt_steps - 1:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_final"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        return mu, np.full(n_evals, np.inf)  # all actions are valid