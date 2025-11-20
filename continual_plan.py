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
from utils import move_to_device
import time
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict
import sys

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed
from collections import deque

from models.vit import ViTPredictor
from models.lora import LoRA_ViT_spread
from planning.online import OnlineLora
from planning.lora_ensemble import EnsembleOnlineLora

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts

# PlanWorkspace 클래스의 __init__ 메서드 
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
        current_task_id: int = None,  # 현재 태스크 ID 추가
    ):
        # --- 1. 기본 속성 초기화 ---
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.current_task_id = current_task_id  # 현재 태스크 ID 저장
        self.device = next(wm.parameters()).device
        
        # 🔧 각 평가마다 다른 시드 사용 (다양성 보장)
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        # --- LoRA 학습 관련 설정 및 객체 생성 ---
        self.online_learner = None # OnlineLora 또는 EnsembleOnlineLora 객체를 담을 변수
        self.is_lora_enabled = self.cfg_dict.get("lora", {}).get("enabled", False)
        self.is_online_lora = self.cfg_dict.get("lora", {}).get("online", False)

        if self.is_lora_enabled:
            # 앙상블 LoRA 사용 여부 확인
            # lora.ensemble로 플래그 경로 변경
            use_ensemble_lora = self.cfg_dict.get("lora", {}).get("ensemble", False)
            
            if use_ensemble_lora:
                print("INFO: Ensemble LoRA training enabled. Initializing EnsembleOnlineLora module.")
                self.online_learner = EnsembleOnlineLora(workspace=self)
            else:
                print("INFO: Standard LoRA training enabled. Initializing OnlineLora module.")
                self.online_learner = OnlineLora(workspace=self)

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
            workspace=self, # workspace를 evaluator에 전달하는 것은 그대로 유지
            is_lora_enabled=self.is_lora_enabled,
            is_online_lora=self.is_online_lora,
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        # 🔧 앙상블 매니저를 플래너에 전달
        planner_kwargs = {
            "wm": self.wm,
            "env": self.env,  # only for mpc
            "action_dim": self.action_dim,
            "objective_fn": objective_fn,
            "preprocessor": self.data_preprocessor,
            "evaluator": self.evaluator,
            "wandb_run": self.wandb_run,
            "log_filename": self.log_filename,
        }
        
        # 앙상블 LoRA가 활성화된 경우 앙상블 매니저 전달
        if (self.is_online_lora and 
            hasattr(self.online_learner, 'ensemble_manager')):
            planner_kwargs["ensemble_manager"] = self.online_learner.ensemble_manager
            print(f"🔧 Passing ensemble manager to planner")
        
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            **planner_kwargs
        )

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]
            
        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None

        # 🔧 태스크 전환 감지 및 처리 (재호출 제거: 바깥에서 이미 설정된 플래그 사용)
        if self.is_online_lora and hasattr(self.online_learner, 'task_changed'):
            # 현재 태스크 ID는 워크스페이스 생성 시 주입됨; 여기서 재계산/재호출하지 않음
            task_changed = self.online_learner.task_changed
        
        # 🔧 태스크 전환 시에만 앙상블 추론 사용
        # lora.ensemble_cfg로 경로 변경 (없으면 빈 dict)
        ensemble_cfg = self.cfg_dict.get("lora", {}).get("ensemble_cfg", {})
        inference_cfg = ensemble_cfg.get("inference", {})
        usage_strategy = inference_cfg.get("usage_strategy", "task_change_only")
        
        if (usage_strategy in ["task_change_only", "always"] and 
            self.is_online_lora and 
            hasattr(self.online_learner, 'ensemble_manager') and
            (len(self.online_learner.ensemble_manager.ensemble_members) > 0 or 
             getattr(self, 'current_task_id', 1) == 1) and  # 첫 번째 태스크도 허용
            (usage_strategy == "always" or 
             (hasattr(self.online_learner, 'task_changed') and self.online_learner.task_changed))):
            if usage_strategy == "always":
                print(f"🔄 Using ensemble for optimal member selection (always mode)...")
            else:
                print(f"🔄 Task changed! Using ensemble for optimal member selection...")
            
            # 태스크 전환 시 앙상블 기반 최적 멤버 선택
            self.online_learner.perform_task_change_ensemble_selection(self)
            
            # task_changed 플래그 리셋
            if hasattr(self.online_learner, 'reset_task_changed_flag'):
                self.online_learner.reset_task_changed_flag()
            
            # 일반 플래닝 수행 (선택된 멤버 사용)
            actions, action_len = self.planner.plan(
                obs_0=self.obs_0,
                obs_g=self.obs_g,
                actions=actions_init,
            )
        else:
            # 기존 방식: 일반 플래닝 수행
            actions, action_len = self.planner.plan(
                obs_0=self.obs_0,
                obs_g=self.obs_g,
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
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs
    
    


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    loaded_keys = []
    result = {}
    # 디버깅: 체크포인트에 있는 모든 키 출력
    print(f"Checkpoint keys: {list(payload.keys())}")
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            # None 체크 추가 (dummy checkpoint 지원)
            if v is not None:
                result[k] = v.to(device)
            else:
                result[k] = None
    result["epoch"] = payload.get("epoch", 0)
    print(f"Loaded model keys: {loaded_keys}")
    print(f"Missing model keys: {set(ALL_MODEL_KEYS) - set(loaded_keys)}")
    return result

def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError(
            f"Predictor not found in model checkpoint: {model_ckpt}\n"
            f"Available keys in checkpoint: {list(result.keys())}\n"
            f"Expected keys: {ALL_MODEL_KEYS}"
        )

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
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
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


def measure_forgetting_on_past_tasks(model, past_task_envs, plan_workspace, current_task_id, ensemble_cache_dir=None):
    """
    과거 태스크들에 대한 현재 모델의 성능을 측정하여 파국적 망각을 분석합니다.
    granular_plan.py처럼 해당 태스크의 앙상블 멤버를 로드하여 평가합니다.
    
    Args:
        model: 현재 학습된 모델
        past_task_envs: 과거 태스크 환경들의 리스트
        plan_workspace: 현재 플래닝 워크스페이스
        current_task_id: 현재 태스크 ID
        ensemble_cache_dir: 앙상블 멤버 캐시 디렉토리 경로
    
    Returns:
        forgetting_results: 각 과거 태스크에 대한 성능 측정 결과 (현재 모델 + 앙상블 멤버)
    """
    forgetting_results = []
    device = next(model.parameters()).device
    
    # 앙상블 캐시 디렉토리 설정
    if ensemble_cache_dir is None:
        ensemble_cfg = plan_workspace.cfg_dict.get("lora", {}).get("ensemble_cfg", {})
        ensemble_cache_dir = os.path.abspath(ensemble_cfg.get("cache_dir", "./lora_cache"))
    
    for past_task in past_task_envs[:-1]:  # 마지막(현재) 태스크 제외
        past_task_id = past_task['task_id']
        past_env = past_task['env']
        past_env_config = past_task['env_config']
        
        print(f"   Testing on Task {past_task_id}: {past_env_config}")
        
        # 두 가지 모드로 측정: 현재 모델과 앙상블 멤버
        for mode_label, use_ensemble_member in [("current_model", False), ("ensemble_member", True)]:
            if use_ensemble_member and not plan_workspace.is_online_lora:
                continue  # 앙상블이 없으면 건너뜀
            
            try:
                # 과거 태스크 환경에서 플래닝 수행
                past_plan_workspace = PlanWorkspace(
                    cfg_dict=plan_workspace.cfg_dict.copy(),
                    wm=model,  # 현재 학습된 모델 사용
                    dset=plan_workspace.dset,
                    env=past_env,
                    env_name=plan_workspace.env_name,
                    frameskip=plan_workspace.frameskip,
                    wandb_run=None,  # 망각 측정 시에는 wandb 로깅 비활성화
                    current_task_id=past_task_id,
                )
                
                # 앙상블 멤버 로드 (해당 태스크의 멤버)
                loaded_file = "N/A"
                if use_ensemble_member:
                    ensemble_member_lora_path = os.path.join(
                        ensemble_cache_dir, f"lora_task_{past_task_id}.pth"
                    )
                    if os.path.exists(ensemble_member_lora_path):
                        try:
                            lora_weights = torch.load(ensemble_member_lora_path, map_location=device)
                            learner = getattr(past_plan_workspace, "online_learner", None)
                            if (
                                past_plan_workspace.is_online_lora
                                and learner is not None
                                and hasattr(learner, "_apply_lora_weights")
                            ):
                                success = learner._apply_lora_weights(lora_weights)
                                if success and hasattr(learner, "last_selected_member_task_id"):
                                    setattr(learner, "last_selected_member_task_id", past_task_id)
                                loaded_file = os.path.basename(ensemble_member_lora_path)
                                print(f"     📥 Loaded ensemble member for Task {past_task_id} from {loaded_file}")
                            else:
                                print(f"     ⚠️  Could not apply LoRA weights (is_online_lora={past_plan_workspace.is_online_lora})")
                        except Exception as e:
                            print(f"     ⚠️  Failed to load ensemble member for Task {past_task_id}: {e}")
                            continue
                    else:
                        print(f"     ⚠️  Ensemble member file not found: {ensemble_member_lora_path}")
                        continue
                else:
                    # 현재 모델 모드: 현재 태스크의 LoRA 사용 (있는 경우)
                    current_model_lora_path = os.path.join(
                        ensemble_cache_dir, f"lora_task_{current_task_id}.pth"
                    )
                    if os.path.exists(current_model_lora_path):
                        try:
                            lora_weights = torch.load(current_model_lora_path, map_location=device)
                            learner = getattr(past_plan_workspace, "online_learner", None)
                            if (
                                past_plan_workspace.is_online_lora
                                and learner is not None
                                and hasattr(learner, "_apply_lora_weights")
                            ):
                                learner._apply_lora_weights(lora_weights)
                                loaded_file = os.path.basename(current_model_lora_path)
                                print(f"     📥 Loaded current model LoRA from {loaded_file}")
                        except Exception as e:
                            print(f"     ⚠️  Failed to load current model LoRA: {e}")
                
                # 간단한 성능 측정 (1회 플래닝)
                logs = past_plan_workspace.perform_planning()
                
                # 성능 지표 추출
                success_rate = logs.get('success_rate', 0.0)
                avg_loss = None
                
                # Loss 측정 (온라인 LoRA가 활성화된 경우)
                if (past_plan_workspace.is_online_lora and 
                    hasattr(past_plan_workspace.online_learner, 'last_loss') and
                    past_plan_workspace.online_learner.last_loss is not None):
                    avg_loss = past_plan_workspace.online_learner.last_loss
                
                # 망각 측정 결과 저장
                result = {
                    'past_task_id': past_task_id,
                    'current_task_id': current_task_id,
                    'env_config': past_env_config,
                    'mode': mode_label,
                    'success_rate': success_rate,
                    'avg_loss': avg_loss if avg_loss is not None else 0.0,
                    'loaded_file': loaded_file
                }
                forgetting_results.append(result)
                
                avg_loss_str = f"{avg_loss:.6f}" if avg_loss is not None else "N/A"
                print(f"     → Task {past_task_id} ({mode_label}): Success Rate: {success_rate:.3f}, Loss: {avg_loss_str} [loaded: {loaded_file}]")
                
            except Exception as e:
                print(f"     → Error measuring Task {past_task_id} ({mode_label}): {e}")
                # 오류 발생 시 기본값으로 기록
                forgetting_results.append({
                    'past_task_id': past_task_id,
                    'current_task_id': current_task_id,
                    'env_config': past_env_config,
                    'mode': mode_label,
                    'success_rate': 0.0,
                    'avg_loss': 0.0,
                    'error': str(e),
                    'loaded_file': loaded_file if 'loaded_file' in locals() else "N/A"
                })
    
    return forgetting_results


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

def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"continual_plan_{cfg_dict['planner']['name']}", config=cfg_dict
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

    # --- ▼ 1. 모델 로딩을 루프 밖으로 이동 ▼ ---
    # 모델은 모든 태스크에 걸쳐 상태가 유지되어야 하므로 한 번만 로드합니다.
    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    # --- ▼ 2. 연속 학습을 위한 태스크 정의 ▼ ---
    # 11개의 서로 다른 환경 설정을 정의합니다.
    task_configs = [
        {'shape': 'T', 'color': 'Yellow', 'background_color': 'White'},             # Task 8: 블록 노랑


        {'shape': 'T', 'color': 'Black', 'background_color': 'White'},              # Task 7: 블록 검정
        {'shape': 'T', 'color': 'Black', 'background_color': 'Red'},              # Task 11: 블록 검정 + 배경 빨강

        # 복합적 변화 (맨 뒤에 배치)
        {'shape': 'square', 'color': 'LightSlateGray', 'background_color': 'Black'}, # Task 9: 정사각형 + 배경 검정
        # 블록 모양 변화
        {'shape': 'T', 'color': 'LightSlateGray', 'background_color': 'Black'},      # Task 5: 배경 검정

        {'shape': 'T',       'color': 'LightSlateGray', 'background_color': 'White'},  # Task 1: A (baseline)

        {'shape': 'L', 'color': 'Yellow', 'background_color': 'White'},             # Task 10: L + 블록 노랑

        # 블록 색상 변화
        {'shape': 'L',       'color': 'LightSlateGray', 'background_color': 'White'},  # Task 1: A (baseline)
    ]
    num_tasks = len(task_configs)
    overall_logs = []
    
    # 태스크 추적을 위한 변수들
    task_summary = []
    current_task_start_time = None
    
    # 파국적 망각 측정을 위한 변수들
    past_task_envs = []  # 과거 태스크 환경들 저장
    forgetting_metrics = []  # 각 태스크에서의 과거 성능 측정 결과
    
    # 앙상블 캐시 디렉토리 설정
    ensemble_cfg = cfg_dict.get("lora", {}).get("ensemble_cfg", {})
    ensemble_cache_dir = os.path.abspath(ensemble_cfg.get("cache_dir", "./lora_cache"))

    # --- ▼ 3. 태스크를 순회하는 최상위 제어 루프 생성 ▼ ---
    for task_id, env_config in enumerate(task_configs):
        print(f"\n{'='*25} Starting Task {task_id + 1}/{num_tasks} {'='*25}")
        print(f"Environment Config: {env_config}")
        
        # 태스크 시작 시간 기록
        current_task_start_time = time.time()
        task_planning_steps = 0
        task_lora_stacks = 0
        
        # 태스크 변경 시 LoRA 적층 트리거 (첫 번째 태스크가 아닌 경우)
        if task_id > 0:
            print(f"\nTask transition detected: Task {task_id} → Task {task_id + 1}")
            print(f"Environment change: {task_configs[task_id-1]} → {env_config}")

        # 3-A. 현재 태스크에 맞는 환경을 동적으로 생성합니다.
        #      (pusht_env.py가 이 인자들을 받도록 수정되었다고 가정)
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name,
                    *model_cfg.env.args,
                    **model_cfg.env.kwargs,
                    **env_config  # 여기에 shape, color 등 인자 전달
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

        task_cfg_dict = cfg_dict.copy()
        task_cfg_dict["planner"]["logging_prefix"] = f"task_{task_id+1:02d}_plan"
        task_cfg_dict["planner"]["sub_planner"]["logging_prefix"] = f"task_{task_id+1:02d}_plan"

        # 🔧 앙상블 멤버 백업 (태스크 전환 시)
        ensemble_backup = None
        if task_id > 0 and 'plan_workspace' in locals() and plan_workspace and hasattr(plan_workspace.online_learner, 'ensemble_manager'):
            ensemble_backup = {
                'members': dict(plan_workspace.online_learner.ensemble_manager.ensemble_members),
                'memory_usage': plan_workspace.online_learner.ensemble_manager.memory_usage,
                'access_frequency': dict(plan_workspace.online_learner.ensemble_manager.access_frequency) if hasattr(plan_workspace.online_learner.ensemble_manager, 'access_frequency') else {}
            }
            print(f"🔧 Backed up {len(ensemble_backup['members'])} ensemble members from previous task")

        # 3-B. 새로운 PlanWorkspace 생성 (각 태스크마다)
        print(f"🔧 Creating new PlanWorkspace for Task {task_id + 1}")
        plan_workspace = PlanWorkspace(
            cfg_dict=task_cfg_dict,
            wm=model,
            dset=dset,
            env=env,
            env_name=model_cfg.env.name,
            frameskip=model_cfg.frameskip,
            wandb_run=wandb_run,
            current_task_id=task_id + 1,
        )
        
        # 🔧 앙상블 멤버 복원
        if ensemble_backup and hasattr(plan_workspace.online_learner, 'ensemble_manager'):
            plan_workspace.online_learner.ensemble_manager.ensemble_members = ensemble_backup['members']
            plan_workspace.online_learner.ensemble_manager.memory_usage = ensemble_backup['memory_usage']
            if hasattr(plan_workspace.online_learner.ensemble_manager, 'access_frequency'):
                plan_workspace.online_learner.ensemble_manager.access_frequency = ensemble_backup['access_frequency']
            print(f"🔧 Restored {len(ensemble_backup['members'])} ensemble members to new PlanWorkspace")
            print(f"   - Memory usage: {ensemble_backup['memory_usage']:.2f}MB")
        else:
            print(f"🔧 Starting with fresh ensemble (no previous members)")
        
        # 🔧 태스크 전환 플래그 설정: 태스크 전환 시 앙상블 평가 조건 만족
        try:
            if (task_id >= 0 and hasattr(plan_workspace, 'online_learner') and
                hasattr(plan_workspace.online_learner, 'check_task_change')):
                changed = plan_workspace.online_learner.check_task_change(task_id + 1)
                print(f"🔄 Task change flag updated via OnlineLora.check_task_change: {changed}")
                # PlanWorkspace에도 현재 태스크 ID 반영 (perform_planning 등에서 사용)
                plan_workspace.current_task_id = getattr(
                    plan_workspace.online_learner, 'current_task_id', task_id + 1
                )
        except Exception as e:
            print(f"⚠️  Failed to set task change flag: {e}")

        # 🔧 앙상블 전용 모드에서 최초 적층 강제 수행 (task_based_stacking=false여도 1회 적층)
        try:
            if (hasattr(plan_workspace, 'online_learner') and
                hasattr(plan_workspace.online_learner, 'hybrid_enabled') and
                plan_workspace.online_learner.hybrid_enabled and
                hasattr(plan_workspace.online_learner, 'task_based_stacking') and
                not plan_workspace.online_learner.task_based_stacking and
                hasattr(plan_workspace.online_learner, 'ensemble_manager')):
                # cfg 플래그 확인
                hybrid_cfg = cfg_dict.get("lora", {}).get("hybrid_stacking", {})
                force_initial = hybrid_cfg.get("force_initial_stacking", True)
                # 현재 앙상블 멤버 수 확인 및 최초 적층 여부 결정
                num_members = len(plan_workspace.online_learner.ensemble_manager.ensemble_members)
                if force_initial and num_members == 0 and getattr(plan_workspace.online_learner, 'stacks_in_current_task', 0) == 0:
                    print(f"\n🎯 Forcing initial LoRA stacking for Task {task_id + 1} (ensemble_initial)")
                    success = plan_workspace.online_learner.trigger_task_based_stacking(task_id + 1, "ensemble_initial")
                    if success:
                        task_lora_stacks += 1
                        print(f"✅ Initial ensemble-based LoRA stacking completed. Total stacks in task: {task_lora_stacks}")
                    else:
                        print(f"⚠️  Initial ensemble-based LoRA stacking skipped or failed.")
        except Exception as e:
            print(f"⚠️  Failed to perform forced initial stacking: {e}")

        # LoRA 적층 콜백 설정 (태스크 추적 + 앙상블 저장 둘 다 수행)
        if plan_workspace.is_online_lora and hasattr(plan_workspace.online_learner, 'base_online_lora'):
            old_cb = getattr(plan_workspace.online_learner.base_online_lora, 'on_lora_stack_callback', None)

            def on_lora_stack(steps, loss, task_id, stack_type, reason):
                # 1) 기존 콜백 호출 (앙상블 멤버 저장 등)
                if callable(old_cb):
                    try:
                        old_cb(steps, loss, task_id, stack_type, reason)
                    except Exception as e:
                        print(f"⚠️  Error in chained base callback: {e}")

                # 2) 태스크별 스택 카운트 업데이트
                nonlocal task_lora_stacks
                task_lora_stacks += 1
                loss_str = f"{loss:.6f}" if loss is not None else "N/A"
                # 선택된 앙상블 멤버 ID 추적 (있으면 표시)
                selected_member_id = None
                try:
                    if hasattr(plan_workspace.online_learner, 'last_selected_member_task_id'):
                        selected_member_id = getattr(plan_workspace.online_learner, 'last_selected_member_task_id')
                except Exception:
                    selected_member_id = None
                if selected_member_id is not None:
                    print(f"🔥 LoRA Stack #{task_lora_stacks} at step {steps} (task {task_id}, type {stack_type}, reason {reason}, on_member {selected_member_id}) loss {loss_str}")
                else:
                    print(f"🔥 LoRA Stack #{task_lora_stacks} at step {steps} (task {task_id}, type {stack_type}, reason {reason}) loss {loss_str}")
                # 최근 스택 히스토리에 선택 멤버 ID 주석 추가
                try:
                    if hasattr(plan_workspace.online_learner, 'stack_history') and isinstance(plan_workspace.online_learner.stack_history, list):
                        if len(plan_workspace.online_learner.stack_history) > 0 and isinstance(plan_workspace.online_learner.stack_history[-1], dict):
                            plan_workspace.online_learner.stack_history[-1]['selected_member_task_id'] = selected_member_id
                except Exception:
                    pass

            # Base OnlineLora가 호출하는 콜백을 래핑하여 교체
            plan_workspace.online_learner.base_online_lora.on_lora_stack_callback = on_lora_stack
        
        # 태스크 변경 시 LoRA 적층 트리거 (하이브리드 모드에서만)
        if (hasattr(plan_workspace.online_learner, 'trigger_task_based_stacking') and
            hasattr(plan_workspace.online_learner, 'hybrid_enabled') and
            plan_workspace.online_learner.hybrid_enabled and
            hasattr(plan_workspace.online_learner, 'task_based_stacking') and
            plan_workspace.online_learner.task_based_stacking):
            print(f"\n🎯 Triggering task-based LoRA stacking for Task {task_id + 1} (adds new LoRA layers to model)")
            success = plan_workspace.online_learner.trigger_task_based_stacking(task_id + 1, "task_change")
            if success:
                task_lora_stacks += 1
                print(f"✅ Task-based LoRA stacking successful. Total stacks in task: {task_lora_stacks}")
            else:
                print(f"Task-based LoRA stacking skipped or failed.")
        else:
            # 🔧 앙상블 LoRA 사용 여부에 따른 메시지 출력
            if hasattr(plan_workspace.online_learner, 'ensemble_manager'):
                print(f"Task-based LoRA stacking disabled (hybrid_enabled: {getattr(plan_workspace.online_learner, 'hybrid_enabled', False)}, task_based_stacking: {getattr(plan_workspace.online_learner, 'task_based_stacking', False)})")
            else:
                print(f"⚠️  Standard OnlineLora mode - using default LoRA stacking behavior")

        # 3-C. Loss 기반 태스크 전환 로직
        task_switch_config = cfg_dict.get("lora", {}).get("task_switch", {})
        use_loss_based_switching = task_switch_config.get("enabled", False)
        
        if use_loss_based_switching:
            print("📊 Using loss-based task transition logic (determines when to move to next task)")
            loss_threshold = task_switch_config.get("loss_threshold", )
            max_planning = task_switch_config.get("max_planning_per_task", )
            min_planning = task_switch_config.get("min_planning_per_task", )
            
            # Loss 윈도우 초기화 (태스크 전환용)
            loss_window_size = task_switch_config.get("loss_window_size", )
            loss_window = deque(maxlen=loss_window_size)
            planning_count = 0
            
            while planning_count < max_planning:
                planning_count += 1
                print(f"\n--- Task {task_id + 1}, Planning Step {planning_count} ---")
                
                # 플래닝 수행
                logs = plan_workspace.perform_planning()
                task_planning_steps += 1
                
                # Loss 값 추출 (evaluator에서 온라인 학습이 수행된 경우)
                current_loss = None
                if plan_workspace.is_online_lora and hasattr(plan_workspace.online_learner, 'last_loss'):
                    current_loss = plan_workspace.online_learner.last_loss
                    if current_loss is not None:
                        print(f"Current loss: {current_loss:.6f} (threshold: {loss_threshold})")
                    else:
                        print(f"Current loss: N/A (threshold: {loss_threshold})")
                
                # 로그 저장
                log_entry_with_task = {f"task_{task_id+1}/{k}": v for k, v in logs.items()}
                if current_loss is not None:
                    log_entry_with_task[f"task_{task_id+1}/current_loss"] = current_loss
                overall_logs.append(log_entry_with_task)
                
                if wandb_run:
                    wandb_run.log(log_entry_with_task)
                
                # Loss 기반 전환 조건 확인
                if current_loss is not None:
                    loss_window.append(current_loss)
                    
                    # 최소 플래닝 횟수 확인 후 Loss 조건 체크
                    if (planning_count >= min_planning and 
                        len(loss_window) >= loss_window_size and
                        all(loss < loss_threshold for loss in list(loss_window)[-1:])):  # 최근 3개가 모두 임계값 이하
                        print(f"✓ Loss converged below threshold {loss_threshold}. Moving to next task.")
                        break
                else:
                    # Loss 정보가 없는 경우 (온라인 학습이 비활성화된 경우) 기본 횟수로 진행
                    if planning_count >= min_planning:
                        print("No loss information available. Using default planning count.")
                        break
        else:
            # 기존 방식: 고정 횟수
            num_planning_per_task = 10
            for i in range(num_planning_per_task):
                print(f"\n--- Task {task_id + 1}, Planning Step {i+1}/{num_planning_per_task} ---")
                logs = plan_workspace.perform_planning()
                task_planning_steps += 1
                log_entry_with_task = {f"task_{task_id+1}/{k}": v for k, v in logs.items()}
                overall_logs.append(log_entry_with_task)
                if wandb_run:
                    wandb_run.log(log_entry_with_task)

        # --- ▼ 4. 태스크 전환 시 LoRA 적층(Stacking) 로직 호출 ▼ ---
        # Online-LoRA 기능이 활성화되어 있고 마지막 태스크가 아닌 경우에만 실행합니다.
        if (plan_workspace.is_online_lora and task_id < num_tasks - 1):
            print(f"\n--- End of Task {task_id + 1}. Adding new LoRA stack for the next task. ---")
            
            # 모델에 LoRA 스택 추가 및 옵티마이저 리셋을 요청합니다.
            # 이 기능들은 models/lora.py 와 planning/online.py에 구현되어 있어야 합니다.
            if hasattr(model.predictor, 'add_new_lora_stack'):
                model.predictor.add_new_lora_stack()
            if hasattr(plan_workspace.online_learner, 'reset_optimizer'):
                plan_workspace.online_learner.reset_optimizer()

        # --- 태스크 종료 시 앙상블에 최종 LoRA 멤버 저장 (최종 학습 상태 반영) ---
        try:
            if (plan_workspace.is_online_lora and hasattr(plan_workspace, 'online_learner') and
                hasattr(plan_workspace.online_learner, 'save_current_lora_member') and
                getattr(plan_workspace.online_learner, 'save_on_task_end', True)):
                
                # 🔧 태스크 완료 시 w에 wnew 누적 (누락된 부분!)
                if hasattr(plan_workspace.online_learner, 'update_and_reset_lora_parameters'):
                    plan_workspace.online_learner.update_and_reset_lora_parameters()
                    print(f"🔄 Updated w with wnew for Task {task_id + 1}")
                
                current_task_for_save = getattr(plan_workspace.online_learner, 'current_task_id', task_id + 1)
                print(f"💾 Saving finalized LoRA member for Task {current_task_for_save} at task end...")
                plan_workspace.online_learner.save_current_lora_member(task_id=current_task_for_save, reason="task_end")
                
                # 🔧 디스크에 저장 (granular_plan.py와 동일한 로직)
                if (hasattr(plan_workspace.online_learner, 'ensemble_manager') and
                    plan_workspace.online_learner.ensemble_manager is not None):
                    manager = plan_workspace.online_learner.ensemble_manager
                    if current_task_for_save in manager.ensemble_members:
                        try:
                            manager._save_member_to_disk(current_task_for_save, manager.ensemble_members[current_task_for_save])
                            print(f"💾 Saved LoRA member for Task {current_task_for_save} to disk: {os.path.join(manager.cache_dir, f'lora_task_{current_task_for_save}.pth')}")
                        except Exception as e:
                            print(f"⚠️  Failed to save LoRA member to disk for Task {current_task_for_save}: {e}")
        except Exception as e:
            print(f"⚠️  Failed to save finalized LoRA member at task end: {e}")

        # 태스크 완료 요약 생성
        task_duration = time.time() - current_task_start_time
        
        # 적층 히스토리 수집 (하이브리드 모드에서)
        stack_history = []
        if (plan_workspace.is_online_lora and 
            hasattr(plan_workspace.online_learner, 'stack_history')):
            stack_history = plan_workspace.online_learner.stack_history.copy()
        
        task_summary.append({
            'task_id': task_id + 1,
            'env_config': env_config,
            'planning_steps': task_planning_steps,
            'lora_stacks': task_lora_stacks,
            'duration_seconds': task_duration,
            'final_loss': current_loss if 'current_loss' in locals() else None,
            'stack_history': stack_history
        })
        
        print(f"\n📊 Task {task_id + 1} Summary:")
        print(f"   - Environment: {env_config}")
        print(f"   - Planning Steps: {task_planning_steps}")
        print(f"   - LoRA Stacks: {task_lora_stacks}")
        print(f"   - Duration: {task_duration:.2f}s")
        if 'current_loss' in locals() and current_loss is not None:
            print(f"   - Final Loss: {current_loss:.6f}")

        # 현재 태스크 환경을 과거 태스크 리스트에 저장
        past_task_envs.append({
            'task_id': task_id + 1,
            'env_config': env_config,
            'env': env  # 환경 객체 저장 (성능 측정용)
        })
        
        # 파국적 망각 측정: 과거 태스크들에 대한 현재 모델 성능 평가
        if len(past_task_envs) > 1:  # 첫 번째 태스크 이후부터 측정
            print(f"\n🧠 Measuring Catastrophic Forgetting...")
            forgetting_results = measure_forgetting_on_past_tasks(
                model, past_task_envs, plan_workspace, task_id, ensemble_cache_dir
            )
            forgetting_metrics.append(forgetting_results)
            
            # 망각 측정 결과 출력
            print(f"📊 Forgetting Analysis for Task {task_id + 1}:")
            for result in forgetting_results:
                avg_loss_str = f"{result['avg_loss']:.6f}" if result['avg_loss'] is not None else "N/A"
                mode_label = result.get('mode', 'unknown')
                loaded_file = result.get('loaded_file', 'N/A')
                print(f"   - Task {result['past_task_id']} ({mode_label}): Success Rate {result['success_rate']:.3f}, "
                      f"Loss {avg_loss_str} [loaded: {loaded_file}]")

        # 리소스 정리를 위해 현재 태스크의 환경을 닫습니다.
        # env.close()  # 과거 태스크 측정을 위해 환경을 닫지 않음

    print(f"\n{'='*25} Continual Learning Experiment Finished {'='*25}")
    
    # 전체 실험 요약 리포트 출력
    print(f"\n{'='*60}")
    print(f"📈 CONTINUAL LEARNING EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tasks: {len(task_summary)}")
    print(f"Total Duration: {sum(task['duration_seconds'] for task in task_summary):.2f}s")
    total_lora_stacks = sum(task['lora_stacks'] for task in task_summary)
    # 앙상블 멤버 수는 총 스택 수와 일치해야 함 (태스크 종료 시마다 멤버 저장)
    ensemble_member_count = total_lora_stacks
    print(f"Total LoRA Stacks: {total_lora_stacks} (Ensemble Members: {ensemble_member_count})")
    print(f"Total Planning Steps: {sum(task['planning_steps'] for task in task_summary)}")
    
    print(f"\n📋 TASK-BY-TASK BREAKDOWN:")
    print(f"{'Task':<6} {'Environment':<40} {'Steps':<6} {'LoRA':<5} {'Duration':<8} {'Final Loss':<10}")
    print(f"{'-'*80}")
    
    for task in task_summary:
        env_str = f"{task['env_config'].get('shape', 'N/A')}-{task['env_config'].get('color', 'N/A')}-{task['env_config'].get('background_color', 'N/A')}"
        final_loss_str = f"{task['final_loss']:.6f}" if task['final_loss'] is not None else "N/A"
        print(f"{task['task_id']:<6} {env_str:<40} {task['planning_steps']:<6} {task['lora_stacks']:<5} {task['duration_seconds']:.2f}s{'':<4} {final_loss_str:<10}")
    
    print(f"\n🎯 KEY INSIGHTS:")
    fastest_task = min(task_summary, key=lambda x: x['duration_seconds'])
    slowest_task = max(task_summary, key=lambda x: x['duration_seconds'])
    most_stacks = max(task_summary, key=lambda x: x['lora_stacks'])
    most_steps = max(task_summary, key=lambda x: x['planning_steps'])
    
    print(f"   - Fastest Task: Task {fastest_task['task_id']} ({fastest_task['duration_seconds']:.2f}s)")
    print(f"   - Slowest Task: Task {slowest_task['task_id']} ({slowest_task['duration_seconds']:.2f}s)")
    print(f"   - Most LoRA Stacks: Task {most_stacks['task_id']} ({most_stacks['lora_stacks']} stacks)")
    print(f"   - Most Planning Steps: Task {most_steps['task_id']} ({most_steps['planning_steps']} steps)")
    
    avg_steps = sum(task['planning_steps'] for task in task_summary) / len(task_summary)
    avg_stacks = sum(task['lora_stacks'] for task in task_summary) / len(task_summary)
    print(f"   - Average Planning Steps: {avg_steps:.1f}")
    print(f"   - Average LoRA Stacks: {avg_stacks:.1f}")
    
    # 하이브리드 적층 분석
    print(f"\n🎯 HYBRID STACKING ANALYSIS")
    print(f"{'='*60}")
    
    total_task_based_stacks = 0
    total_loss_based_stacks = 0
    
    for task in task_summary:
        if 'stack_history' in task and task['stack_history']:
            task_based_count = sum(1 for stack in task['stack_history'] if stack['type'] == 'task_based')
            loss_based_count = sum(1 for stack in task['stack_history'] if stack['type'] == 'loss_based')
            total_task_based_stacks += task_based_count
            total_loss_based_stacks += loss_based_count
            
            print(f"Task {task['task_id']}: {task_based_count} task-based, {loss_based_count} loss-based stacks")
    
    print(f"\n📊 STACKING TYPE SUMMARY:")
    print(f"   - Total Task-based Stacks: {total_task_based_stacks}")
    print(f"   - Total Loss-based Stacks: {total_loss_based_stacks}")
    print(f"   - Total Stacks: {total_task_based_stacks + total_loss_based_stacks}")
    if total_task_based_stacks + total_loss_based_stacks > 0:
        task_based_ratio = total_task_based_stacks / (total_task_based_stacks + total_loss_based_stacks) * 100
        print(f"   - Task-based Ratio: {task_based_ratio:.1f}%")
    
    print(f"{'='*60}")
    
    # 파국적 망각 분석 리포트
    if forgetting_metrics:
        print(f"\n🧠 CATASTROPHIC FORGETTING ANALYSIS")
        print(f"{'='*60}")
        
        # 전체 망각 통계
        all_losses = []
        task_loss_summary = {}
        
        for task_idx, forgetting_results in enumerate(forgetting_metrics):
            current_task_id = task_idx + 2  # 첫 번째 태스크 이후부터 측정
            print(f"\n📊 Forgetting Analysis after Task {current_task_id}:")
            
            for result in forgetting_results:
                past_task_id = result['past_task_id']
                avg_loss = result['avg_loss']
                mode_label = result.get('mode', 'unknown')
                loaded_file = result.get('loaded_file', 'N/A')
                if avg_loss is not None and avg_loss > 0:
                    all_losses.append(avg_loss)
                
                # 각 과거 태스크별 평균 loss 계산 (모드별로 구분)
                key = f"{past_task_id}_{mode_label}"
                if key not in task_loss_summary:
                    task_loss_summary[key] = []
                task_loss_summary[key].append(avg_loss)
                
                env_str = f"{result['env_config'].get('shape', 'N/A')}-{result['env_config'].get('color', 'N/A')}-{result['env_config'].get('background_color', 'N/A')}"
                loss_str = f"{avg_loss:.6f}" if avg_loss is not None else "N/A"
                print(f"   Task {past_task_id} ({mode_label}, {env_str}): Loss {loss_str} [loaded: {loaded_file}]")
        
        # 전체 망각 분석
        if all_losses:
            avg_loss = sum(all_losses) / len(all_losses)
            min_loss = min(all_losses)
            max_loss = max(all_losses)
            
            print(f"\n🎯 OVERALL FORGETTING STATISTICS:")
            print(f"   - Average Loss: {avg_loss:.6f}")
            print(f"   - Minimum Loss: {min_loss:.6f}")
            print(f"   - Maximum Loss: {max_loss:.6f}")
            print(f"   - Total Measurements: {len(all_losses)}")
            
            # 각 태스크별 평균 loss (모드별로 구분)
            print(f"\n📈 TASK-SPECIFIC LOSS SUMMARY:")
            # 태스크 ID별로 그룹화
            task_grouped = {}
            for key, losses in task_loss_summary.items():
                parts = key.split('_', 1)
                if len(parts) == 2:
                    task_id, mode = parts
                    if task_id not in task_grouped:
                        task_grouped[task_id] = {}
                    task_grouped[task_id][mode] = losses
            
            for past_task_id in sorted(task_grouped.keys(), key=int):
                for mode in sorted(task_grouped[past_task_id].keys()):
                    valid_losses = [loss for loss in task_grouped[past_task_id][mode] if loss is not None and loss > 0]
                    if valid_losses:
                        avg_loss_for_task = sum(valid_losses) / len(valid_losses)
                        print(f"   Task {past_task_id} ({mode}): {avg_loss_for_task:.6f} average loss")
            
            # 망각 심각도 평가 (loss 기반)
            print(f"\n⚠️  FORGETTING SEVERITY ASSESSMENT:")
            if avg_loss <= 0.1:
                print(f"   🟢 LOW FORGETTING: (Loss {avg_loss:.6f})")
            elif avg_loss <= 0.2:
                print(f"   🟡 MODERATE FORGETTING: (Loss {avg_loss:.6f})")
            elif avg_loss <= 0.3:
                print(f"   🟠 HIGH FORGETTING: (Loss {avg_loss:.6f})")
            else:
                print(f"   🔴 SEVERE FORGETTING: (Loss {avg_loss:.6f})")
        print(f"{'='*60}")
    
    # 환경 정리
    for past_task in past_task_envs:
        if 'env' in past_task and past_task['env']:
            past_task['env'].close()

    return overall_logs


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