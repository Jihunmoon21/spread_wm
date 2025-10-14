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
import time
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed
from collections import deque

from models.vit import ViTPredictor
from models.lora import LoRA_ViT_spread
from planning.online import OnlineLora

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
        self.online_learner = None # OnlineLora 객체를 담을 변수
        self.is_lora_enabled = self.cfg_dict.get("lora", {}).get("enabled", False)
        self.is_online_lora = self.cfg_dict.get("lora", {}).get("online", False)

        if self.is_lora_enabled:
            print("INFO: LoRA training enabled. Initializing OnlineLora module.")
            # OnlineLora 객체를 생성하고, workspace 전체를 넘겨 Evaluator에서 접근할 수 있도록
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
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
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
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result

def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
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


def measure_forgetting_on_past_tasks(model, past_task_envs, plan_workspace, current_task_id):
    """
    과거 태스크들에 대한 현재 모델의 성능을 측정하여 파국적 망각을 분석합니다.
    
    Args:
        model: 현재 학습된 모델
        past_task_envs: 과거 태스크 환경들의 리스트
        plan_workspace: 현재 플래닝 워크스페이스
        current_task_id: 현재 태스크 ID
    
    Returns:
        forgetting_results: 각 과거 태스크에 대한 성능 측정 결과
    """
    forgetting_results = []
    
    for past_task in past_task_envs[:-1]:  # 마지막(현재) 태스크 제외
        past_task_id = past_task['task_id']
        past_env = past_task['env']
        past_env_config = past_task['env_config']
        
        print(f"   Testing on Task {past_task_id}: {past_env_config}")
        
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
            )
            
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
            
            # Loss 기반 retention rate 계산 (0.1과의 근접도)
            target_loss = 0.1
            if avg_loss is not None:
                # Loss가 0.1보다 작으면 무조건 100% retention
                if avg_loss <= target_loss:
                    retention_rate = 100.0
                else:
                    # Loss가 0.1보다 클 때만 거리 기반 계산
                    loss_distance = avg_loss - target_loss
                    # 지수적 감소: distance가 클수록 retention이 빠르게 감소
                    retention_rate = max(0, 100 * (1.0 - loss_distance / target_loss))
            else:
                retention_rate = 0.0
            
            # 망각 측정 결과 저장
            result = {
                'past_task_id': past_task_id,
                'current_task_id': current_task_id,
                'env_config': past_env_config,
                'success_rate': success_rate,
                'avg_loss': avg_loss if avg_loss is not None else 0.0,
                'retention_rate': retention_rate,  # Loss 기반 유지율
                'target_loss': target_loss  # 목표 Loss 추가
            }
            forgetting_results.append(result)
            
            avg_loss_str = f"{avg_loss:.6f}" if avg_loss is not None else "N/A"
            print(f"     → Task {past_task_id} Success Rate: {success_rate:.3f}, Loss: {avg_loss_str} (Target: {target_loss}, Retention: {retention_rate:.1f}%)")
            
        except Exception as e:
            print(f"     → Error measuring Task {past_task_id}: {e}")
            # 오류 발생 시 기본값으로 기록
            forgetting_results.append({
                'past_task_id': past_task_id,
                'current_task_id': current_task_id,
                'env_config': past_env_config,
                'success_rate': 0.0,
                'avg_loss': 0.0,
                'retention_rate': 0.0,
                'error': str(e)
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
        # Original (기본 설정)
        {'shape': 'T', 'color': 'LightSlateGray', 'background_color': 'White'},
        
        # 블록 모양 변화
        {'shape': 'square', 'color': 'LightSlateGray', 'background_color': 'White'}, # Task 2: 정사각형
        {'shape': 'small_tee', 'color': 'LightSlateGray', 'background_color': 'White'}, # Task 3: small_tee
        {'shape': 'L', 'color': 'LightSlateGray', 'background_color': 'White'},     # Task 4: L
        
        # 블록 색상 변화
        {'shape': 'T', 'color': 'Black', 'background_color': 'White'},              # Task 7: 블록 검정
        {'shape': 'T', 'color': 'Yellow', 'background_color': 'White'},             # Task 8: 블록 노랑

        # 배경색 변화
        {'shape': 'T', 'color': 'LightSlateGray', 'background_color': 'Black'},      # Task 5: 배경 검정
        {'shape': 'T', 'color': 'LightSlateGray', 'background_color': 'Red'},       # Task 6: 배경 빨강
        
        # 복합적 변화 (맨 뒤에 배치)
        {'shape': 'square', 'color': 'LightSlateGray', 'background_color': 'Black'}, # Task 9: 정사각형 + 배경 검정
        {'shape': 'L', 'color': 'Yellow', 'background_color': 'White'},             # Task 10: L + 블록 노랑
        {'shape': 'T', 'color': 'Black', 'background_color': 'Red'},              # Task 11: 블록 검정 + 배경 빨강
    ]
    num_tasks = len(task_configs)
    overall_logs = []
    
    # 태스크 추적을 위한 변수들
    task_summary = []
    current_task_start_time = None
    
    # 파국적 망각 측정을 위한 변수들
    past_task_envs = []  # 과거 태스크 환경들 저장
    forgetting_metrics = []  # 각 태스크에서의 과거 성능 측정 결과

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

        # 3-B. PlanWorkspace를 생성합니다. 모델(wm)은 계속 유지됩니다.
        plan_workspace = PlanWorkspace(
            cfg_dict=task_cfg_dict,
            wm=model,
            dset=dset,
            env=env,
            env_name=model_cfg.env.name,
            frameskip=model_cfg.frameskip,
            wandb_run=wandb_run,
        )
        
        # LoRA 적층 콜백 설정 (태스크 추적을 위해)
        if plan_workspace.is_online_lora and hasattr(plan_workspace.online_learner, 'on_lora_stack_callback'):
            def on_lora_stack(steps, loss):
                nonlocal task_lora_stacks
                task_lora_stacks += 1
                loss_str = f"{loss:.6f}" if loss is not None else "N/A"
                print(f"🔥 LoRA Stack #{task_lora_stacks} at step {steps} with loss {loss_str}")
            plan_workspace.online_learner.on_lora_stack_callback = on_lora_stack
        
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
                print(f"⚠️  Task-based LoRA stacking skipped or failed.")
        else:
            print(f"⚠️  Task-based LoRA stacking disabled (hybrid_enabled: {getattr(plan_workspace.online_learner, 'hybrid_enabled', False)}, task_based_stacking: {getattr(plan_workspace.online_learner, 'task_based_stacking', False)})")

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
                    print(f"Current loss: {current_loss:.6f} (threshold: {loss_threshold})")
                
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
            num_planning_per_task = 1
            for i in range(num_planning_per_task):
                print(f"\n--- Task {task_id + 1}, Planning Step {i+1}/{num_planning_per_task} ---")
                logs = plan_workspace.perform_planning()
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
                model, past_task_envs, plan_workspace, task_id
            )
            forgetting_metrics.append(forgetting_results)
            
            # 망각 측정 결과 출력
            print(f"📊 Forgetting Analysis for Task {task_id + 1}:")
            for result in forgetting_results:
                avg_loss_str = f"{result['avg_loss']:.6f}" if result['avg_loss'] is not None else "N/A"
                target_loss = result.get('target_loss', 0.1)
                print(f"   - Task {result['past_task_id']}: Success Rate {result['success_rate']:.3f}, "
                      f"Loss {avg_loss_str} (Target: {target_loss}, Retention: {result['retention_rate']:.1f}%)")

        # 리소스 정리를 위해 현재 태스크의 환경을 닫습니다.
        # env.close()  # 과거 태스크 측정을 위해 환경을 닫지 않음

    print(f"\n{'='*25} Continual Learning Experiment Finished {'='*25}")
    
    # 전체 실험 요약 리포트 출력
    print(f"\n{'='*60}")
    print(f"📈 CONTINUAL LEARNING EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tasks: {len(task_summary)}")
    print(f"Total Duration: {sum(task['duration_seconds'] for task in task_summary):.2f}s")
    print(f"Total LoRA Stacks: {sum(task['lora_stacks'] for task in task_summary)}")
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
        all_retention_rates = []
        task_retention_summary = {}
        
        for task_idx, forgetting_results in enumerate(forgetting_metrics):
            current_task_id = task_idx + 2  # 첫 번째 태스크 이후부터 측정
            print(f"\n📊 Forgetting Analysis after Task {current_task_id}:")
            
            for result in forgetting_results:
                past_task_id = result['past_task_id']
                retention_rate = result['retention_rate']
                all_retention_rates.append(retention_rate)
                
                # 각 과거 태스크별 평균 유지율 계산
                if past_task_id not in task_retention_summary:
                    task_retention_summary[past_task_id] = []
                task_retention_summary[past_task_id].append(retention_rate)
                
                env_str = f"{result['env_config'].get('shape', 'N/A')}-{result['env_config'].get('color', 'N/A')}-{result['env_config'].get('background_color', 'N/A')}"
                print(f"   Task {past_task_id} ({env_str}): {retention_rate:.1f}% retention")
        
        # 전체 망각 분석
        avg_retention = sum(all_retention_rates) / len(all_retention_rates) if all_retention_rates else 0
        min_retention = min(all_retention_rates) if all_retention_rates else 0
        max_retention = max(all_retention_rates) if all_retention_rates else 0
        
        print(f"\n🎯 OVERALL FORGETTING STATISTICS:")
        print(f"   - Average Retention Rate: {avg_retention:.1f}%")
        print(f"   - Minimum Retention Rate: {min_retention:.1f}%")
        print(f"   - Maximum Retention Rate: {max_retention:.1f}%")
        print(f"   - Total Measurements: {len(all_retention_rates)}")
        
        # 각 태스크별 평균 유지율
        print(f"\n📈 TASK-SPECIFIC RETENTION RATES:")
        for past_task_id in sorted(task_retention_summary.keys()):
            avg_retention_for_task = sum(task_retention_summary[past_task_id]) / len(task_retention_summary[past_task_id])
            print(f"   Task {past_task_id}: {avg_retention_for_task:.1f}% average retention")
        
        # 망각 심각도 평가
        print(f"\n⚠️  FORGETTING SEVERITY ASSESSMENT:")
        if avg_retention >= 80:
            print(f"   🟢 LOW FORGETTING: ({avg_retention:.1f}%)")
        elif avg_retention >= 60:
            print(f"   🟡 MODERATE FORGETTING: ({avg_retention:.1f}%)")
        elif avg_retention >= 40:
            print(f"   🟠 HIGH FORGETTING: ({avg_retention:.1f}%)")
        else:
            print(f"   🔴 SEVERE FORGETTING: ({avg_retention:.1f}%)")
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