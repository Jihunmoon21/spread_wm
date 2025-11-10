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

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
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
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]
        
        # 각 trajectory에 대한 best loss 추적 (CD가 증가하는 행동 제외용)
        best_losses = [float('inf')] * n_evals

        for i in range(self.opt_steps):
            print(f"CEM Step {i+1}/{self.opt_steps}")
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
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

                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                
                # CD가 증가하는 행동 제외: 현재 best보다 나쁜 행동 필터링
                if best_losses[traj] < float('inf'):
                    # 현재 best보다 좋거나 같은 행동만 선택
                    valid_mask = loss <= best_losses[traj]
                    if valid_mask.sum() > 0:
                        # 유효한 행동들 중에서만 topk 선택
                        valid_loss = loss[valid_mask]
                        valid_action = action[valid_mask]
                        valid_indices = torch.where(valid_mask)[0]
                        
                        # 유효한 행동이 topk보다 적으면 모두 사용, 많으면 topk만 선택
                        k = min(self.topk, valid_loss.shape[0])
                        topk_idx_in_valid = torch.argsort(valid_loss)[:k]
                        topk_idx = valid_indices[topk_idx_in_valid]
                        topk_action = valid_action[topk_idx_in_valid]
                    else:
                        # 모든 행동이 나쁘면 기존 best 유지 (mu, sigma 업데이트 안 함)
                        print(f"[CEM] Traj {traj}: All actions worse than best (best_loss={best_losses[traj]:.4f}), keeping previous best")
                        losses.append(best_losses[traj])
                        continue
                else:
                    # 첫 iteration: 기존 로직 사용
                    topk_idx = torch.argsort(loss)[: self.topk]
                    topk_action = action[topk_idx]
                
                # Best loss 업데이트
                current_best_loss = loss[topk_idx[0]].item()
                if current_best_loss < best_losses[traj]:
                    best_losses[traj] = current_best_loss
                
                losses.append(current_best_loss)
                # 아래 두 줄 삭제하면 cem 작동 안 함
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        return mu, np.full(n_evals, np.inf)  # all actions are valid
