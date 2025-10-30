import torch
import hydra
import copy
import numpy as np
from einops import rearrange, repeat
from utils import slice_trajdict_with_t
from .base_planner import BasePlanner


class MPCPlanner(BasePlanner):
    """
    an online planner so feedback from env is allowed
    """

    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        env,  # for online exec
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="mpc",
        log_filename="logs.json",
        ensemble_manager=None,  # 🔧 앙상블 매니저 추가
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
        self.env = env
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions
        self.logging_prefix = logging_prefix
        sub_planner["_target_"] = sub_planner["target"]
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,  # evaluator is shared for mpc and sub_planner
            wandb_run=self.wandb_run,
            log_filename=None,
        )
        self.is_success = None
        self.action_len = None  # keep track of the step each traj reaches success
        self.iter = 0
        self.planned_actions = []
        self.ensemble_manager = ensemble_manager  # 🔧 앙상블 매니저 저장

    def _apply_success_mask(self, actions):
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        # 성공한 trajectory는 0으로 설정 (이미 normalized 공간)
        # Planner는 원본 action_dim으로 동작하므로 frameskip rearrange 불필요
        actions[mask] = 0
        return actions

    def plan(self, obs_0, obs_g, obs_g_traj=None, actions=None):
        """
        actions is NOT used
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """

        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        cur_obs_0 = obs_0
        cur_state_0 = None  # 이전 iteration의 마지막 상태 저장
        memo_actions = None
        while not np.all(self.is_success) and self.iter < self.max_iter:
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            
            # 🔧 업데이트된 상태 설정 (평가 전에)
            if self.iter == 0:
                print(f"[MPC FIX] Setting initial conditions for iter {self.iter}")
                self.evaluator.assign_init_cond(
                    obs_0=init_obs_0,
                    state_0=init_state_0,
                )
                cur_state_0 = init_state_0
            else:
                # 이전 iteration의 마지막 상태에서 시작
                print(f"[MPC FIX] Using updated conditions from previous iter for iter {self.iter}")
                self.evaluator.assign_init_cond(
                    obs_0=cur_obs_0,
                    state_0=cur_state_0,
                )
            
            # 🔧 MPC에서는 일반 플래닝 사용 (앙상블은 태스크 전환 시에만 사용)
            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0,
                obs_g=obs_g,
                obs_g_traj=obs_g_traj,
                actions=memo_actions,
            )  # (b, t, act_dim)
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions :]
            self.planned_actions.append(taken_actions)

            print(f"MPC iter {self.iter} Eval ------- ")
            action_so_far = torch.cat(self.planned_actions, dim=1)
            
            # 🔧 새로 추가된 action만 평가
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                taken_actions,  # action_so_far 대신 새로 추가된 action만
                self.action_len,
                filename=f"{self.logging_prefix}_plan{self.iter}",
                save_video=True,
            )
            new_successes = successes & ~self.is_success  # Identify new successes
            self.is_success = (
                self.is_success | successes
            )  # Update overall success status
            self.action_len[new_successes] = (
                (self.iter + 1) * self.n_taken_actions
            )  # Update only for the newly successful trajectories

            print("self.is_success: ", self.is_success)
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

            # update evaluator's init conditions with new env feedback
            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            e_final_state = e_states[:, -1]
            
            # 다음 iteration을 위한 상태 저장 (중요: cur_state_0도 업데이트)
            cur_obs_0 = e_final_obs
            cur_state_0 = e_final_state
            
            print(f"[MPC FIX] Updating conditions for next iter: final_state shape {e_final_state.shape}")
            self.iter += 1
            self.sub_planner.logging_prefix = f"plan_{self.iter}"

        # 최종 결과 반환
        planned_actions = torch.cat(self.planned_actions, dim=1)
        
        
        # 평가자를 원래 상태로 복원
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )

        return planned_actions, self.action_len

