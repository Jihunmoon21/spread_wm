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

# 불필요한 경고 메시지 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
# 환경 변수로 Python warnings 완전히 비활성화
os.environ["PYTHONWARNINGS"] = "ignore"

# OpenGL accelerate 경고 무시
logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.ERROR)
# Robosuite macro 경고 무시
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)
# 환경 변수로 robosuite 경고 완전히 숨김
os.environ["ROBOSUITE_NO_MACRO_WARNING"] = "1"

# MuJoCo offscreen 렌더링 설정 (headless 환경)
# os.environ["MUJOCO_GL"] = "osmesa"  # CPU 렌더링 (느림, 비활성화)
os.environ["MUJOCO_GL"] = "egl"  # GPU 렌더링 (빠름)

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed, select_data_from_indices # select_data_from_indices 추가
from collections import deque

from models.vit import ViTPredictor
from models.lora import LoRA_ViT_spread
from planning.online import OnlineLora

# libero 관련 추가
import h5py 
import cv2
from PIL import Image

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

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
        # Planning 시에는 frameskip을 사용하지 않음 (학습 시 action_encoder가 원본 action_dim으로 학습됨)
        self.action_dim = self.dset.action_dim
        self.debug_dset_init = cfg_dict["debug_dset_init"]
        # 데이터 원본 프레임 수 로깅용 저장소
        self.data_traj_lens = []

        # --- 목적 함수 로드 ---
        # create_trajectory_objective_fn 사용 시 objective 설정이 달라지므로 유의
        # frameskip을 고려하여 goal_indices를 action step 기준으로 변환
        objective_cfg = cfg_dict["objective"].copy()
        if objective_cfg.get("_target_", "").endswith("create_trajectory_objective_fn"):
            objective_cfg["frameskip"] = frameskip
            # goal_indices가 없으면 goal_frame_indices를 사용 (환경 프레임 기준 → 액션 스텝 기준으로 변환)
            if "goal_indices" not in objective_cfg or objective_cfg.get("goal_indices") is None:
                goal_frame_indices = cfg_dict.get("goal_frame_indices", [-1])
                # goal_frame_indices를 action step 기준으로 변환 (이미 frameskip으로 나눠짐)
                objective_cfg["goal_indices"] = goal_frame_indices
        objective_fn = hydra.utils.call(objective_cfg)

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

        # gt_actions는 분기별로 설정되므로, 사용 전 기본값을 보장한다
        self.gt_actions = None

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
        # evaluator는 최종 목표(obs_g)와 목표 궤적(obs_g_traj) 모두 필요할 수 있음
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
            obs_g_traj=self.obs_g_traj,  # 추가: 목표 궤적 (다중 이미지 골용)
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

        # 플래닝 호라이즌 설정: 우선순위 sub_planner.horizon > planner.horizon > goal_H
        from planning.mpc import MPCPlanner
        planner_cfg = self.cfg_dict.get("planner", {})
        sub_planner_cfg = planner_cfg.get("sub_planner", {})
        planner_horizon = sub_planner_cfg.get(
            "horizon",
            planner_cfg.get("horizon", self.goal_H),
        )
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = planner_horizon
            # MPC의 n_taken_actions도 goal_H 대신 별도 설정 권장
            self.planner.n_taken_actions = planner_cfg.get("n_taken_actions", 1)
        else:
            self.planner.horizon = planner_horizon

        self.dump_targets() # 디버깅용으로 유지

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
            # update_env는 각 worker마다 env_info를 기대하므로 n_evals 만큼 복제
            if env_info_sample and env_info_sample[0]: # env_info_sample이 비어있지 않은지 확인
                self.env.update_env([env_info_sample[0]] * self.n_evals)

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

        # --- ✨ 여기가 수정/대체된 부분 (HDF5 직접 로드) ✨ ---
        elif self.goal_source == 'trajectory':
            log.info("Preparing targets using 'trajectory' source.")
            
            # --- 1. HDF5 파일 직접 접근을 위한 준비 ---
            try:
                # self.dset은 LiberoSliceWrapper, .dataset은 LiberoDataset
                full_dataset = self.dset.dataset 
            except AttributeError:
                 raise TypeError("self.dset does not appear to be a LiberoSliceWrapper. "
                                 "Make sure you are using 'datasets.libero_dset.load_libero_slice_train_val'.")

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
            env_info_list = [] # 각 데모별 환경 정보 저장 (빈 dict 사용)

            # LiberoDataset에서 필요한 속성 가져오기
            main_view = full_dataset.main_view
            transform = full_dataset.transform
            # CPU로 이동하고 numpy로 변환
            proprio_mean = full_dataset.proprio_mean.cpu().numpy()
            proprio_std = full_dataset.proprio_std.cpu().numpy()

            for i in range(self.n_evals):
                traj_idx = goal_traj_indices[i]
                
                try:
                    traj_path = full_dataset.metas[traj_idx]
                except IndexError:
                    raise IndexError(f"goal_traj_index {traj_idx} is out of bounds for dataset metas list of size {len(full_dataset.metas)}.")

                try:
                    with h5py.File(traj_path, 'r') as f:
                        # --- 2. HDF5에서 *전체* 궤적 데이터 로드 ---
                        
                        # (A) Proprio/State 로드
                        # libero_dset은 state = proprio_normalized로 처리함
                        proprio_full_traj_np = f['proprio'][()].astype('float32')
                        proprio_normalized = (proprio_full_traj_np - proprio_mean) / proprio_std
                        
                        # Full simulator state 로드 (qpos + qvel)
                        if 'states' in f:
                            state_full_traj_np = f['states'][()].astype('float32')
                        else:
                            # LIBERO HDF5는 'states'가 없으므로 None 사용
                            # Evaluator에서 환경 기본 reset을 사용하도록 처리
                            state_full_traj_np = None
                        
                        traj_len = proprio_full_traj_np.shape[0]
                        # 데이터 원본 프레임 수 로그 (환경 프레임 기준)
                        log.info(
                            f"[Data] traj_idx={traj_idx} traj_len(env frames)={traj_len} "
                            f"main_view='{main_view}' goal_frame_indices={goal_frame_indices}"
                        )
                        try:
                            self.data_traj_lens.append(int(traj_len))
                        except Exception:
                            pass

                        # (B) Visual (Images) 로드
                        if 'observation' not in f or main_view not in f['observation']:
                            raise KeyError(f"Observation key '{main_view}' not found in {traj_path}.")
                        
                        compressed_images = f['observation'][main_view][()] # (T, bytes)
                        
                        images_list = []
                        for img_data in compressed_images:
                            # 이미지 디코딩 (libero_dset.py __getitem__ 로직과 동일)
                            if isinstance(img_data, np.ndarray) and img_data.dtype == np.uint8: buf = img_data.reshape(-1)
                            elif isinstance(img_data, (bytes, bytearray)): buf = np.frombuffer(img_data, dtype=np.uint8)
                            elif isinstance(img_data, np.void): buf = np.frombuffer(img_data.tobytes(), dtype=np.uint8)
                            else: raise TypeError(f"Unsupported image buffer type: {type(img_data)}")

                            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                            if img_bgr is None: raise ValueError(f"cv2.imdecode failed for traj {traj_idx}")
                            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                            # raw numpy 형식으로 저장 (H, W, C) - preprocessor에서 transform 적용
                            images_list.append(img_rgb)
                        
                        # numpy 배열로 스택: (T, H, W, C)
                        visual_full_traj_np = np.stack(images_list, axis=0)

                        # obs 딕셔너리 생성 (전체 궤적)
                        obs_traj = {
                            'visual': visual_full_traj_np,
                            'proprio': proprio_normalized # normalized proprio for observations
                        }
                        # state 궤적 (full simulator state for environment rollout)
                        state_traj = state_full_traj_np
                        e_info = {} # env_info는 HDF5에 없으므로 빈 dict

                except Exception as e:
                    log.error(f"Failed to load full trajectory from HDF5 file {traj_path} (traj_idx {traj_idx}): {e}")
                    # 이 eval 건너뛰기 또는 중단 (여기서는 중단)
                    raise

                # --- 3. 궤적에서 obs_0, obs_g_traj, state_0, state_g 추출 ---
                
                # (A) 초기 상태 설정
                obs_0_i = {k: v[0] for k, v in obs_traj.items()} # 첫 프레임
                
                # state_traj가 None이면 더미 state 사용 (evaluator에서 환경 reset만 사용)
                if state_traj is not None:
                    state_0_i = state_traj[0]  # 첫 상태
                else:
                    state_0_i = None

                obs_0_list.append(obs_0_i)
                state_0_list.append(state_0_i)
                env_info_list.append(e_info) # 환경 정보 저장 (빈 dict)

                # (B) 목표 프레임(들) 설정
                obs_g_frames = select_data_from_indices(obs_traj, goal_frame_indices, traj_len) # 목표 인덱스 프레임 추출
                obs_g_traj_list.append(obs_g_frames)

                # (C) 최종 목표 상태 설정 (Evaluator용) - goal_frame_indices의 마지막 인덱스 사용
                final_goal_idx = goal_frame_indices[-1]
                actual_final_goal_idx = final_goal_idx if final_goal_idx >= 0 else traj_len + final_goal_idx
                if state_traj is not None:
                    if 0 <= actual_final_goal_idx < traj_len:
                        state_g_i = state_traj[actual_final_goal_idx]
                    else:
                        log.error(
                            f"Final goal index {actual_final_goal_idx} is out of bounds for trajectory {traj_idx} of length {traj_len}."
                        )
                        state_g_i = state_traj[-1]  # 폴백: 마지막 프레임 상태
                    state_g_list.append(state_g_i)
                else:
                    state_g_list.append(None)

            # --- 4. 리스트들을 batch 형태로 변환 (기존 로직 동일) ---
            self.obs_0 = stack_obs_list(obs_0_list, add_time_dim=True)
            self.obs_g_traj = stack_obs_list(obs_g_traj_list, add_time_dim=False) # 이미 시간(N_goals) 차원 있음

            self.obs_g = {k: v[:, -1:] for k, v in self.obs_g_traj.items()} # (n_evals, 1, ...)

            # state_0_list에 None이 있으면 환경 기본 reset 사용
            if all(s is None for s in state_0_list):
                self.state_0 = None
                self.state_g = None
                log.info("No simulator states available. Will use environment default reset.")
            else:
                self.state_0 = np.array(state_0_list)  # (n_evals, d)
                self.state_g = np.array(state_g_list)  # (n_evals, d) - 최종 목표 상태
                self.gt_actions = None  # 플래닝 모드에서는 정답 액션 없음

            if env_info_list and env_info_list[0]:  # 비어있지 않은 경우만
                log.info("Updating environment settings based on loaded trajectory.")
                # update_env는 각 worker마다 env_info를 기대하므로 n_evals 만큼 복제
                self.env.update_env([env_info_list[0]] * self.n_evals)

        # --- 기존 dset / random_action 로직 ---
        elif self.goal_source in ["dset", "random_action"]:
            log.info(f"Preparing targets using '{self.goal_source}' source with goal_H={self.goal_H}.")
            traj_len_needed = self.frameskip * self.goal_H + 1

            # update env config from val trajs
            observations, states, actions, env_info_list = (
                self.sample_traj_segment_from_dset(traj_len=traj_len_needed, n_samples=self.n_evals)
            )
            # 환경 설정 업데이트 (위 trajectory 로직과 동일한 방식 적용)
            if env_info_list and env_info_list[0]:
                log.info("Updating environment settings based on the first sampled trajectory.")
                # update_env는 각 worker마다 env_info를 기대하므로 n_evals 만큼 복제
                self.env.update_env([env_info_list[0]] * self.n_evals)

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

        # --- planner.plan 호출 시 obs_g_traj 전달 (항상) ---
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            obs_g_traj=self.obs_g_traj,
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


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    # accelerate safetensors 형식 지원
    if model_ckpt.is_dir():
        from safetensors.torch import load_file
        
        safetensors_path = model_ckpt / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"model.safetensors not found in {model_ckpt}")
        
        state_dict = load_file(str(safetensors_path), device=str(device))
        
        # epoch 정보 로드
        epoch_path = model_ckpt / "custom_epoch.pth"
        if epoch_path.exists():
            epoch_data = torch.load(epoch_path, map_location=device, weights_only=False)
            epoch = epoch_data.get("epoch", 0)
        else:
            epoch = 0
        
        print(f"Resuming from epoch {epoch}: {model_ckpt}")
        
        # 각 컴포넌트 생성
        encoder = hydra.utils.instantiate(train_cfg.encoder)
        
        # Proprio/Action encoder - state_dict에서 차원 추론
        from omegaconf import OmegaConf
        
        proprio_encoder_state = {k[len("proprio_encoder."):]: v for k, v in state_dict.items() if k.startswith("proprio_encoder.")}
        if proprio_encoder_state and "patch_embed.weight" in proprio_encoder_state:
            weight_shape = proprio_encoder_state["patch_embed.weight"].shape
            proprio_cfg = OmegaConf.create(train_cfg.proprio_encoder)
            proprio_cfg.in_chans = int(weight_shape[1])
            proprio_cfg.emb_dim = int(weight_shape[0])
            proprio_encoder = hydra.utils.instantiate(proprio_cfg)
        else:
            proprio_encoder = hydra.utils.instantiate(train_cfg.proprio_encoder)
        
        action_encoder_state = {k[len("action_encoder."):]: v for k, v in state_dict.items() if k.startswith("action_encoder.")}
        if action_encoder_state and "patch_embed.weight" in action_encoder_state:
            weight_shape = action_encoder_state["patch_embed.weight"].shape
            action_cfg = OmegaConf.create(train_cfg.action_encoder)
            action_cfg.in_chans = int(weight_shape[1])
            action_cfg.emb_dim = int(weight_shape[0])
            action_encoder = hydra.utils.instantiate(action_cfg)
        else:
            action_encoder = hydra.utils.instantiate(train_cfg.action_encoder)
        
        # encoder에서 필요한 정보 추출 (먼저 정의)
        encoder_emb_dim = encoder.emb_dim
        num_patches = (train_cfg.img_size // encoder.patch_size) ** 2
        num_frames = train_cfg.num_hist
        
        # Decoder
        if train_cfg.has_decoder:
            # decoder의 channel과 emb_dim 모두 encoder_emb_dim과 같아야 함
            decoder_cfg = OmegaConf.create(train_cfg.decoder)
            decoder_cfg.channel = encoder_emb_dim
            decoder_cfg.emb_dim = encoder_emb_dim  # quantize_b의 차원도 맞춰야 함!
            decoder = hydra.utils.instantiate(decoder_cfg)
        else:
            decoder = None
        
        # Predictor 생성 (VWorldModel이 LoRA로 감싸므로 미리 생성 필요)
        
        # concat_dim=1이면 proprio와 action embedding도 포함
        if train_cfg.concat_dim == 1:
            predictor_dim = encoder_emb_dim + train_cfg.proprio_emb_dim + train_cfg.action_emb_dim
        else:
            predictor_dim = encoder_emb_dim
        
        # Predictor config 수정 후 instantiate
        predictor_cfg = OmegaConf.create(train_cfg.predictor)
        predictor_cfg.num_patches = num_patches
        predictor_cfg.num_frames = num_frames
        predictor_cfg.dim = predictor_dim
        predictor = hydra.utils.instantiate(predictor_cfg)
        
        # 모델 생성
        model = hydra.utils.instantiate(
            train_cfg.model,
            encoder=encoder,
            proprio_encoder=proprio_encoder,
            action_encoder=action_encoder,
            predictor=predictor,
            decoder=decoder,
            proprio_dim=train_cfg.proprio_emb_dim,
            action_dim=train_cfg.action_emb_dim,
            concat_dim=train_cfg.concat_dim,
            num_action_repeat=num_action_repeat,
            num_proprio_repeat=train_cfg.num_proprio_repeat,
        )
        model.to(device)
        
        # state_dict 로드
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {len(state_dict)} parameters into model")
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
        
        return model
    
    # 기존 .pth 형식
    else:
        result = {}
        if model_ckpt.exists():
            result = load_ckpt(model_ckpt, device)
            print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

        if "encoder" not in result:
            result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
        if "predictor" not in result:
            raise ValueError("Predictor not found in model checkpoint")

        if train_cfg.has_decoder and "decoder" not in result:
            base_path = os.path.dirname(os.path.abspath(__file__))
            if train_cfg.env.decoder_path is not None:
                decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
                ckpt = torch.load(decoder_path)
                if isinstance(ckpt, dict):
                    result["decoder"] = ckpt["decoder"]
                else:
                    result["decoder"] = torch.load(decoder_path)
            else:
                raise ValueError("Decoder not found")
        elif not train_cfg.has_decoder:
            result["decoder"] = None

        model = hydra.utils.instantiate(
            train_cfg.model,
            encoder=result["encoder"],
            proprio_encoder=result["proprio_encoder"],
            action_encoder=result["action_encoder"],
            predictor=result["predictor"],
            decoder=result["decoder"],
            proprio_dim=train_cfg.proprio_emb_dim,
            action_dim=train_cfg.action_emb_dim,
            concat_dim=train_cfg.concat_dim,
            num_action_repeat=num_action_repeat,
            num_proprio_repeat=train_cfg.num_proprio_repeat,
        )
        model.to(device)
        return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


# planning_main 내부 env 생성 부분 수정 (환경 설정 전달)
def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict.get("wandb_logging", True):
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    # ========================================
    # Task name과 goal_traj_index 자동 매칭
    # ========================================
    def find_traj_indices_for_task(metas, task_name):
        """task_name에 해당하는 모든 데모 인덱스 찾기"""
        indices = []
        for idx, traj_path in enumerate(metas):
            filename = os.path.basename(traj_path)
            # 형식 1: pick_up_the_task_name_demo.hdf5
            if filename.startswith(f"{task_name}_"):
                indices.append(idx)
            # 형식 2: pick_up_the_task_name/demo_0.hdf5
            dir_name = os.path.basename(os.path.dirname(traj_path))
            if dir_name == task_name:
                indices.append(idx)
        return indices

    env_config = cfg_dict.get("env", {})
    if hasattr(model_cfg.env, 'kwargs'):
        env_kwargs = model_cfg.env.kwargs.copy()
        env_kwargs.update(env_config)
        env_config = env_kwargs
    
    task_name = env_config.get('task_name')
    
    # task_name이 있으면 해당 태스크의 데모 자동 찾기
    if task_name:
        full_dataset = dset.dataset
        matching_indices = find_traj_indices_for_task(full_dataset.metas, task_name)
        
        if not matching_indices:
            raise ValueError(f"No demo found for task '{task_name}' in dataset")
        
        # 첫 번째 매칭 데모 사용
        auto_goal_traj_index = matching_indices[0]
        
        # goal_traj_index가 명시적으로 설정되지 않았거나, 설정값이 없으면 자동 매칭 사용
        current_goal_traj_index = cfg_dict.get('goal_traj_index', [None])[0] if isinstance(cfg_dict.get('goal_traj_index', 0), list) else cfg_dict.get('goal_traj_index', None)
        
        if current_goal_traj_index is None or current_goal_traj_index not in matching_indices:
            cfg_dict['goal_traj_index'] = auto_goal_traj_index
            log.info(f"Auto-matched goal_traj_index={auto_goal_traj_index} for task '{task_name}'")
        else:
            log.info(f"Using specified goal_traj_index={current_goal_traj_index} for task '{task_name}'")
    else:
        log.info("No task_name specified, using goal_traj_index from config")

    if env_config:
        log.info(f"Environment config: {env_config}")

    # OmegaConf DictConfig를 Python dict로 변환 (robosuite가 dict를 기대함)
    env_config_dict = OmegaConf.to_container(env_config, resolve=True)
    if not isinstance(env_config_dict, dict):
        env_config_dict = {}

    num_action_repeat = model_cfg.num_action_repeat
    # accelerate는 디렉토리 형식으로 저장하므로 .pth 확장자 제거
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}"
    )
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    # 환경 생성 시 env_config 전달
    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        # SerialVectorEnv는 config 전달 기능 없을 수 있음 -> 필요시 수정
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **env_config_dict
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
                    model_cfg.env.name, *model_cfg.env.args, **env_config_dict
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