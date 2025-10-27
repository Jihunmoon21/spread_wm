import os
import gym
import json
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange, repeat # repeat 추가
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed, select_data_from_indices # select_data_from_indices 추가
from collections import deque

from models.vit import ViTPredictor
from models.lora import LoRA_ViT_spread
from planning.online import OnlineLora

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ... (기존 코드: ALL_MODEL_KEYS, planning_main_in_dir, launch_plan_jobs, build_plan_cfg_dicts) ...

class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        # --- 1. 기본 속성 초기화 ---
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        # goal_H는 dset, random_action에서만 주로 사용되지만, 일단 유지
        self.goal_H = cfg_dict.get("goal_H", 1) # 기본값 추가
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        # --- 목적 함수 로드 ---
        # create_trajectory_objective_fn 사용 시 objective 설정이 달라지므로 유의
        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        # --- 데이터 전처리기 ---
        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        # --- 목표 데이터 준비 (prepare_targets 호출) ---
        # obs_0, obs_g, state_0, state_g, obs_g_traj 등이 여기서 설정됨
        if self.cfg_dict.get("goal_source") == "file": # .get() 사용으로 안정성 확보
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets() # 수정된 함수 호출

        # --- LoRA 학습 관련 설정 및 객체 생성 ---
        self.online_learner = None # OnlineLora 객체를 담을 변수
        self.is_lora_enabled = self.cfg_dict.get("lora", {}).get("enabled", False)
        self.is_online_lora = self.cfg_dict.get("lora", {}).get("online", False)

        if self.is_lora_enabled:
            print("INFO: LoRA training enabled. Initializing OnlineLora module.")
            self.online_learner = OnlineLora(workspace=self)

        # --- 평가기 초기화 ---
        # evaluator는 최종 목표(obs_g)만 필요할 수 있음
        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g, # 최종 목표만 전달
            state_0=self.state_0,
            state_g=self.state_g, # 최종 목표 상태만 전달
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
            workspace=self,
            is_lora_enabled=self.is_lora_enabled,
            is_online_lora=self.is_online_lora,
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"

        # --- 플래너 초기화 ---
        # planner는 전체 목표 궤적(obs_g_traj)이 필요할 수 있으므로,
        # 생성자에 obs_g_traj를 전달하거나 plan 메소드에서 받도록 수정 필요
        # 여기서는 plan 메소드에서 받는다고 가정
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # MPC용
            action_dim=self.action_dim,
            objective_fn=objective_fn, # 수정된 목적 함수 사용
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        # 플래닝 호라이즌 설정 (기존 로직 유지, 필요시 goal_H 대신 planner horizon 직접 설정)
        from planning.mpc import MPCPlanner
        # MPC의 sub_planner horizon 설정 시 주의 (goal_H가 아닐 수 있음)
        # plan_libero.yaml에서 planner.horizon을 직접 설정하는 것을 권장
        planner_horizon = self.cfg_dict.get("planner", {}).get("horizon", self.goal_H)
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = planner_horizon
            # MPC의 n_taken_actions도 goal_H 대신 별도 설정 권장
            self.planner.n_taken_actions = self.cfg_dict.get("planner", {}).get("n_taken_actions", 1)
        else:
            self.planner.horizon = planner_horizon

        self.dump_targets() # 디버깅용으로 유지

    # 여기가 수정된 prepare_targets 메소드
    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        env_info_list = [] # 여러 환경 정보를 받을 리스트

        # obs_g_traj 초기화 (trajectory 모드 아닐 때도 정의)
        self.obs_g_traj = None

        if self.goal_source == "random_state":
            log.info("Preparing targets using 'random_state' source.")
            # 기존 random_state 로직 (Deformable Env 특수 처리 포함)
            # update env config from val trajs (하나의 traj 정보만 사용)
            _, states_sample, _, env_info_sample = (
                self.sample_traj_segment_from_dset(traj_len=2, n_samples=1) # 단일 샘플만 필요
            )
            # update_env가 리스트를 받을 경우를 대비해 리스트로 전달
            # 하지만 random_state는 보통 단일 환경 설정 기준이므로 첫 번째 것만 사용
            self.env.update_env([env_info_sample[0]])

            # sample random states (n_evals 만큼)
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                 # deformable은 데이터셋의 첫 상태를 가져오는 로직 유지
                 # 이 부분은 deformable 환경 전용 로직이므로, 필요시 환경별 분기 강화
                 _, states_deform, _, _ = self.sample_traj_segment_from_dset(traj_len=2, n_samples=self.n_evals)
                 rand_init_state = np.array([x[0] for x in states_deform])


            # prepare는 각 eval seed 별로 호출해야 함 (병렬 환경 고려)
            # SubprocVectorEnv는 prepare 같은 사용자 정의 메소드 직접 지원 안 함
            # -> 각 환경에 개별적으로 상태 설정 필요 (evaluator 내부 reset 로직과 유사하게)
            # 여기서는 우선 단일 환경처럼 처리 (추후 병렬 환경 지원 시 수정 필요)
            # 일단 state_0, state_g만 저장하고 obs는 evaluator에서 생성하도록 위임 가능
            # 여기서는 기존 구조 유지를 위해 임시 obs 생성
            obs_0_list, obs_g_list = [], []
            temp_env = self.env.envs[0] # 임시로 첫번째 환경 사용
            for i in range(self.n_evals):
                 obs_0_i, _ = temp_env.prepare(self.eval_seed[i], rand_init_state[i])
                 obs_g_i, _ = temp_env.prepare(self.eval_seed[i], rand_goal_state[i])
                 obs_0_list.append(obs_0_i)
                 obs_g_list.append(obs_g_i)

            # 리스트들을 batch 형태로 변환
            self.obs_0 = stack_obs_list(obs_0_list)
            self.obs_g = stack_obs_list(obs_g_list)
            self.state_0 = rand_init_state  # (n_evals, d)
            self.state_g = rand_goal_state
            self.gt_actions = None

        # --- ✨ 여기가 추가/수정된 부분 ✨ ---
        elif self.goal_source == 'trajectory':
            log.info("Preparing targets using 'trajectory' source.")
            goal_traj_indices = self.cfg_dict.get('goal_traj_index', [0]) # 단일 인덱스 또는 리스트
            if not isinstance(goal_traj_indices, list):
                goal_traj_indices = [goal_traj_indices] * self.n_evals # n_evals 만큼 복제

            if len(goal_traj_indices) != self.n_evals:
                 log.warning(f"Number of goal_traj_indices ({len(goal_traj_indices)}) does not match n_evals ({self.n_evals}). Using the first index for all.")
                 goal_traj_indices = [goal_traj_indices[0]] * self.n_evals

            goal_frame_indices = self.cfg_dict.get('goal_frame_indices', [-1])
            if not isinstance(goal_frame_indices, list):
                raise ValueError("goal_frame_indices must be a list.")

            obs_0_list = []
            state_0_list = []
            obs_g_traj_list = [] # 목표 궤적 관측 저장
            state_g_list = [] # 최종 목표 상태만 저장 (evaluator용)
            env_info_list = [] # 각 데모별 환경 정보 저장

            for i in range(self.n_evals):
                traj_idx = goal_traj_indices[i]
                try:
                    obs_traj, _, state_traj, e_info = self.dset[traj_idx]
                    traj_len = obs_traj["visual"].shape[0]
                except IndexError:
                    raise IndexError(f"goal_traj_index {traj_idx} is out of bounds for dataset of size {len(self.dset)}.")

                # 1. 초기 상태 설정
                obs_0_i = {k: v[0] for k, v in obs_traj.items()} # 첫 프레임
                state_0_i = state_traj[0].numpy() if isinstance(state_traj, torch.Tensor) else state_traj[0] # 첫 상태

                obs_0_list.append(obs_0_i)
                state_0_list.append(state_0_i)
                env_info_list.append(e_info) # 환경 정보 저장

                # 2. 목표 프레임(들) 설정
                obs_g_frames = select_data_from_indices(obs_traj, goal_frame_indices, traj_len) # 목표 인덱스 프레임 추출
                obs_g_traj_list.append(obs_g_frames)

                # 3. 최종 목표 상태 설정 (Evaluator용) - goal_frame_indices의 마지막 인덱스 사용
                final_goal_idx = goal_frame_indices[-1]
                actual_final_goal_idx = final_goal_idx if final_goal_idx >= 0 else traj_len + final_goal_idx
                if 0 <= actual_final_goal_idx < traj_len:
                     state_g_i = state_traj[actual_final_goal_idx].numpy() if isinstance(state_traj, torch.Tensor) else state_traj[actual_final_goal_idx]
                     state_g_list.append(state_g_i)
                else:
                     log.error(f"Final goal index {actual_final_goal_idx} is out of bounds for trajectory {traj_idx} of length {traj_len}.")
                     # 오류 처리 또는 기본값 설정 (예: 마지막 프레임 상태)
                     state_g_i = state_traj[-1].numpy() if isinstance(state_traj, torch.Tensor) else state_traj[-1]
                     state_g_list.append(state_g_i)


            # 리스트들을 batch 형태로 변환 (시간 차원 추가)
            # obs_0: (n_evals, 1, C, H, W) 등
            # obs_g_traj: (n_evals, N_goals, C, H, W) 등
            self.obs_0 = stack_obs_list(obs_0_list, add_time_dim=True)
            self.obs_g_traj = stack_obs_list(obs_g_traj_list, add_time_dim=False) # 이미 시간(N_goals) 차원 있음

            # evaluator를 위해 obs_g도 설정 (obs_g_traj의 마지막 프레임 사용)
            self.obs_g = {k: v[:, -1:] for k, v in self.obs_g_traj.items()} # (n_evals, 1, ...)

            self.state_0 = np.array(state_0_list)  # (n_evals, d)
            self.state_g = np.array(state_g_list)  # (n_evals, d) - 최종 목표 상태
            self.gt_actions = None # 플래닝 모드에서는 정답 액션 없음

            # 환경 설정 업데이트 (모든 환경에 적용)
            # 주의: SubprocVecEnv가 각기 다른 env_info를 지원하는지 확인 필요
            #      일반적으로는 동일한 환경 설정을 사용하므로 첫 번째 env_info 사용 가능성 높음
            if env_info_list:
                log.info("Updating environment settings based on the first loaded trajectory.")
                # update_env가 리스트 형태를 기대할 수 있으므로 리스트로 전달
                self.env.update_env([env_info_list[0]]) # 일단 첫번째 정보만 사용


        # --- 기존 dset / random_action 로직 ---
        # (이 부분은 trajectory 모드와 약간의 중복 로직 발생 가능성 -> 리팩토링 고려)
        elif self.goal_source in ["dset", "random_action"]:
            log.info(f"Preparing targets using '{self.goal_source}' source with goal_H={self.goal_H}.")
            traj_len_needed = self.frameskip * self.goal_H + 1

            # update env config from val trajs
            observations, states, actions, env_info_list = (
                self.sample_traj_segment_from_dset(traj_len=traj_len_needed, n_samples=self.n_evals)
            )
            # 환경 설정 업데이트 (위 trajectory 로직과 동일한 방식 적용)
            if env_info_list:
                log.info("Updating environment settings based on the first sampled trajectory.")
                self.env.update_env([env_info_list[0]])

            # get states from val trajs
            init_state = np.array([x[0] for x in states])
            actions_tensor = torch.stack(actions) # 액션은 tensor로 처리

            # random_action인 경우 액션 랜덤화
            if self.goal_source == "random_action":
                log.info("Using random actions to determine goal state.")
                actions_tensor = torch.randn_like(actions_tensor)

            # 액션 정규화 해제 및 실행 가능한 형태로 변환
            # 주의: preprocessor.denormalize_actions는 (..., F, D) 형태를 기대할 수 있음
            # actions_tensor 형태: (B, T_raw, D_raw) T_raw = frameskip * goal_H
            # rollout 함수는 (T, action_dim_env) 형태의 numpy 배열 기대
            # wm_actions는 (B, T_wm, D_wm) 형태. T_wm = goal_H, D_wm = frameskip * D_raw
            wm_actions = rearrange(actions_tensor, "b (t f) d -> b t (f d)", f=self.frameskip)
            # denormalize를 위해 (B, T_wm, F, D_raw) 형태로 변경
            denorm_actions_tensor = rearrange(actions_tensor, "b (t f) d -> b t f d", f=self.frameskip)
            exec_actions_tensor = self.data_preprocessor.denormalize_actions(denorm_actions_tensor) # (B, T_wm, F, D_raw)
            # 실행을 위해 (B, T_raw, D_raw) numpy 형태로 변경
            exec_actions_np = rearrange(exec_actions_tensor, "b t f d -> b (t f) d").numpy() # (B, T_raw, D_raw)

            # 환경에서 액션 롤아웃 (병렬 실행)
            # rollout 함수가 seed 리스트를 받도록 수정되었는지 확인 필요
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions_np # (B, T_raw, D_raw) 전달
            ) # rollout_obses: dict of (B, T_raw+1, ...), rollout_states: (B, T_raw+1, d)

            # obs_0, obs_g 설정
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1) # 첫 프레임, 시간 차원 추가
                for key, arr in rollout_obses.items()
            }
            # 마지막 프레임 (T_raw 시점 후의 상태)
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1) # 마지막 프레임, 시간 차원 추가
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (B, d)
            self.state_g = rollout_states[:, -1]  # (B, d)
            self.gt_actions = wm_actions # 월드 모델용 액션 (B, T_wm, D_wm)

        else:
             raise ValueError(f"Unknown goal_source: {self.goal_source}")


    # sample_traj_segment_from_dset: n_samples 인자 추가
    def sample_traj_segment_from_dset(self, traj_len, n_samples):
        states_batch = []
        actions_batch = []
        observations_batch = []
        env_info_batch = []

        # Check if any trajectory is long enough
        valid_indices = [
            i for i, (obs, _, _, _) in enumerate(self.dset)
            if obs["visual"].shape[0] >= traj_len
        ]

        if len(valid_indices) == 0:
            raise ValueError(f"No trajectory in the dataset is long enough (>= {traj_len} frames).")

        # sample init_states from dset
        sampled_indices = random.choices(valid_indices, k=n_samples) # n_samples 만큼 샘플링

        for traj_id in sampled_indices:
            obs, act, state, e_info = self.dset[traj_id]
            max_offset = obs["visual"].shape[0] - traj_len
            state_np = state.numpy() if isinstance(state, torch.Tensor) else state

            offset = random.randint(0, max_offset)
            # Extract observation segment
            obs_segment = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            # Extract state segment
            state_segment = state_np[offset : offset + traj_len]
            # Extract action segment (only up to needed horizon for dset/random_action)
            # For trajectory mode, this action segment isn't directly used for goals
            act_segment_len = min(act.shape[0] - offset, self.frameskip * self.goal_H)
            act_segment = act[offset : offset + act_segment_len]

            actions_batch.append(act_segment)
            states_batch.append(state_segment)
            observations_batch.append(obs_segment)
            env_info_batch.append(e_info)

        return observations_batch, states_batch, actions_batch, env_info_batch


    def prepare_targets_from_file(self, file_path):
        log.info(f"Loading targets from file: {file_path}")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data.get("gt_actions") # gt_actions 없을 수 있음
        self.goal_H = data.get("goal_H", self.goal_H) # 파일에 없으면 기본값 사용
        # obs_g_traj도 파일에 저장/로드 필요 시 추가
        self.obs_g_traj = data.get("obs_g_traj", None)
        if self.obs_g_traj is None:
             log.warning("obs_g_traj not found in file, initializing as None.")


    def dump_targets(self):
        # dump_targets에도 obs_g_traj 추가
        try:
            with open("plan_targets.pkl", "wb") as f:
                pickle.dump(
                    {
                        "obs_0": self.obs_0,
                        "obs_g": self.obs_g,
                        "state_0": self.state_0,
                        "state_g": self.state_g,
                        "gt_actions": self.gt_actions,
                        "goal_H": self.goal_H,
                        "obs_g_traj": self.obs_g_traj # 추가
                    },
                    f,
                )
            file_path = os.path.abspath("plan_targets.pkl")
            log.info(f"Dumped plan targets to {file_path}")
        except Exception as e:
            log.error(f"Failed to dump plan targets: {e}")


    def perform_planning(self):
        if self.debug_dset_init and self.gt_actions is not None:
             # gt_actions가 None일 수 있으므로 체크
             log.info("Using ground truth actions for initialization (debug mode).")
             actions_init = self.gt_actions
        else:
             actions_init = None

        # --- planner.plan 호출 시 obs_g_traj 전달 ---
        # planner 인터페이스가 obs_g_traj를 받도록 수정 필요 가정
        # 예: plan(self, obs_0, obs_g, obs_g_traj=None, actions=None)
        if hasattr(self.planner, 'plan') and 'obs_g_traj' in self.planner.plan.__code__.co_varnames:
             actions, action_len = self.planner.plan(
                 obs_0=self.obs_0,
                 obs_g=self.obs_g, # 최종 목표 전달 (기존 인터페이스 유지 위함)
                 obs_g_traj=self.obs_g_traj, # 목표 궤적 전달
                 actions=actions_init,
             )
        else:
             # 기존 인터페이스 유지 (obs_g_traj 사용 못함)
             log.warning("Planner does not support 'obs_g_traj'. Falling back to using 'obs_g' only.")
             actions, action_len = self.planner.plan(
                 obs_0=self.obs_0,
                 obs_g=self.obs_g, # 최종 목표만 전달
                 actions=actions_init,
             )


        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.float64, np.int32, np.int64)) # np.float64 추가
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs

# --- Helper function to stack list of obs dicts ---
def stack_obs_list(obs_list, add_time_dim=True):
    """Stacks a list of observation dictionaries into a single batched dictionary."""
    if not obs_list:
        return {}
    keys = obs_list[0].keys()
    stacked_obs = {}
    for key in keys:
        # 각 obs 딕셔너리에서 key에 해당하는 데이터를 모음 (리스트 형태)
        # 예: [[C,H,W], [C,H,W], ...] 또는 [[N,C,H,W], [N,C,H,W], ...] (trajectory)
        data_list = [obs[key] for obs in obs_list]

        # numpy 배열로 변환하고 배치 차원 생성
        # data_list[0]의 shape을 기준으로 stack
        try:
             stacked_data = np.stack(data_list, axis=0) # (B, ...) 또는 (B, N, ...)
        except ValueError as e:
             print(f"Error stacking key '{key}': {e}")
             # 데이터 형태가 일정하지 않을 경우 처리 (예: 패딩 또는 오류 발생)
             # 여기서는 간단히 오류 출력하고 None 반환
             # shapes = [d.shape for d in data_list]
             # print(f"Shapes for key '{key}': {shapes}")
             stacked_data = None # 혹은 다른 오류 처리

        if stacked_data is not None and add_time_dim:
             # 시간 차원 추가 (B, 1, ...)
             stacked_data = np.expand_dims(stacked_data, axis=1)

        stacked_obs[key] = stacked_data
    return stacked_obs


# ... (기존 코드: load_ckpt, load_model, DummyWandbRun, planning_main, main) ...

# planning_main 내부 env 생성 부분 수정 (환경 설정 전달)
def planning_main(cfg_dict):
    # ... (기존 코드: output_dir, device, wandb_run, 모델 로딩, dataset 로딩) ...

    env_config = {}
    # 환경별 설정 로드 (예: libero의 task_name 등)
    if "environment" in cfg_dict:
        env_config = cfg_dict["environment"]
        log.info(f"Using environment config: {env_config}")

    # 환경 생성 시 env_config 전달
    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        # SerialVectorEnv는 config 전달 기능 없을 수 있음 -> 필요시 수정
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        # SubprocVectorEnv에 환경 생성 함수 전달 시 config 포함
        env = SubprocVectorEnv(
            [
                # lambda 함수 내부에 **env_config 추가
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs, **env_config
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    # ... (기존 코드: PlanWorkspace 생성, perform_planning 호출) ...
    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    logs = plan_workspace.perform_planning()
    if wandb_run is not None:
         wandb_run.finish() # wandb 종료 추가
    return logs

# ... (기존 코드: @hydra.main, if __name__ == "__main__":) ...

@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()