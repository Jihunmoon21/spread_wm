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
        actions[mask] = 0
        masked_actions = rearrange(
            actions[mask], "... (f d) -> ... f d", f=self.evaluator.frameskip
        )
        masked_actions = self.preprocessor.normalize_actions(masked_actions.cpu())
        masked_actions = rearrange(masked_actions, "... f d -> ... (f d)")
        actions[mask] = masked_actions.to(device)
        return actions

    def plan(self, obs_0, obs_g, actions=None):
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
        memo_actions = None
        # 실제 환경 기준 CD 추적 (CD 증가 행동 거르기)
        prev_cd = None  # 이전 iteration의 CD 저장
        # 🔧 임시: 성공 여부와 관계없이 max_iter까지 수행
        while self.iter < self.max_iter:
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            
            # 🔧 MPC에서는 일반 플래닝 사용 (앙상블은 태스크 전환 시에만 사용)
            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0,
                obs_g=obs_g,
                actions=memo_actions,
            )  # (b, t, act_dim)
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions :]
            self.planned_actions.append(taken_actions)

            print(f"MPC iter {self.iter} Eval ------- ")
            # 🔧 각 iteration에서는 새로 추가된 action만 평가 (현재 상태에서 시작)
            # 첫 번째 iteration에서만 초기 조건 설정
            if self.iter == 0:
                print(f"[MPC FIX] Setting initial conditions for iter {self.iter}")
                # Reset the flag for initial CD measurement
                self.evaluator._initial_cd_measured = False
                self.evaluator.assign_init_cond(
                    obs_0=init_obs_0,
                    state_0=init_state_0,
                )
            else:
                print(f"[MPC FIX] Using updated conditions from previous iter for iter {self.iter}")
            
            # 새로 추가된 action만 평가 (현재 환경 상태에서 시작)
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                taken_actions,  # 전체 action_so_far가 아닌 새로 추가된 action만
                self.action_len,
                filename=f"{self.logging_prefix}_plan{self.iter}",
                save_video=True,
            )
            # ---- CD 증가 행동 차단 로직 ----
            # evaluator 로그에서 mean_chamfer_distance 추출
            cur_cd = logs.get("mean_chamfer_distance", None)
            if cur_cd is None:
                # fallback: chamfer_distance 직접 확인
                cur_cd = logs.get("chamfer_distance", None)
            if cur_cd is not None:
                # 스칼라 값이면 배열로 변환
                if not isinstance(cur_cd, (list, tuple, np.ndarray)):
                    cur_cd = np.array([cur_cd])
                elif isinstance(cur_cd, (list, tuple)):
                    cur_cd = np.array(cur_cd)
                # 첫 번째 iteration이 아니고, 이전 CD가 있는 경우에만 비교
                if prev_cd is not None:
                    # 현재 CD가 이전 CD보다 나쁜 경우 (증가)
                    if np.any(cur_cd > prev_cd):
                        print(
                            f"[MPC] CD increased: prev={prev_cd}, cur={cur_cd}. Reverting this step."
                        )
                        # 방금 추가한 행동 취소
                        if len(self.planned_actions) > 0:
                            self.planned_actions.pop()
                        # memo_actions도 이전 상태로 되돌리기 (다음 계획을 위해)
                        memo_actions = None
                        # iter는 증가시키되, 상태 업데이트 없이 다음 반복으로 진행
                        self.iter += 1
                        print(f"[MPC] Iter {self.iter} (reverted due to CD increase)\n")
                        continue
                    else:
                        # 개선 또는 동일: prev_cd 업데이트
                        prev_cd = cur_cd.copy()
                        print(f"[MPC] CD improved or same: {cur_cd[0]:.6f}")
                else:
                    # 첫 번째 iteration: prev_cd 초기화
                    prev_cd = cur_cd.copy()
                    print(f"[MPC] Initial CD: {cur_cd[0]:.6f}")
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
            cur_obs_0 = e_final_obs
            e_final_state = e_states[:, -1]
            print(f"[MPC FIX] Updating conditions for next iter: final_state shape {e_final_state.shape}")
            self.evaluator.assign_init_cond(
                obs_0=e_final_obs,
                state_0=e_final_state,
            )
            self.iter += 1
            self.sub_planner.logging_prefix = f"plan_{self.iter}"

        # 최종 결과 반환
        planned_actions = torch.cat(self.planned_actions, dim=1)
        
        # 🔧 final output: 초기 상태에서 전체 궤적(action_so_far) 평가
        print("[MPC] Evaluating final output from initial state with full trajectory")
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )
        # 전체 궤적을 초기 상태에서 평가
        final_logs, final_successes, final_e_obses, final_e_states = self.evaluator.eval_actions(
            planned_actions,  # 전체 궤적
            self.action_len,
            filename="output_final",
            save_video=True,
        )
        print(f"[MPC] Final output CD: {final_logs.get('mean_chamfer_distance', final_logs.get('chamfer_distance', 'N/A'))}")
        
        # 🔧 최종 결과 요약 출력
        print("\n" + "="*80)
        print("[MPC] Final Results Summary")
        print("="*80)
        
        # Steps to Success 계산
        success_mask = self.is_success
        if np.any(success_mask):
            steps_to_success = self.action_len[success_mask]
            print(f"Steps to Success: {steps_to_success.tolist()}")
            print(f"  - Mean: {np.mean(steps_to_success):.2f}")
            print(f"  - Min: {np.min(steps_to_success):.2f}")
            print(f"  - Max: {np.max(steps_to_success):.2f}")
        else:
            print("Steps to Success: N/A (no successful trajectories)")
        
        # LoRA Adaptation Time 통계
        if hasattr(self.evaluator, 'workspace') and self.evaluator.workspace is not None:
            if hasattr(self.evaluator.workspace, 'online_learner') and self.evaluator.workspace.online_learner is not None:
                online_learner = self.evaluator.workspace.online_learner
                # EnsembleOnlineLora인 경우 base_online_lora에서 가져오기 (더 정확한 값)
                if hasattr(online_learner, 'base_online_lora'):
                    adaptation_times = online_learner.base_online_lora.adaptation_times
                elif hasattr(online_learner, 'adaptation_times'):
                    adaptation_times = online_learner.adaptation_times
                else:
                    adaptation_times = []
                
                if len(adaptation_times) > 0:
                    print(f"\nLoRA Adaptation Time (total {len(adaptation_times)} updates):")
                    print(f"  - Min: {min(adaptation_times):.4f} seconds")
                    print(f"  - Max: {max(adaptation_times):.4f} seconds")
                    print(f"  - Mean: {np.mean(adaptation_times):.4f} seconds")
                    print(f"  - Total: {sum(adaptation_times):.4f} seconds")
                else:
                    print("\nLoRA Adaptation Time: N/A (no LoRA updates performed)")
            else:
                print("\nLoRA Adaptation Time: N/A (LoRA not enabled)")
        else:
            print("\nLoRA Adaptation Time: N/A (workspace not available)")
        
        print("="*80 + "\n")

        return planned_actions, self.action_len

