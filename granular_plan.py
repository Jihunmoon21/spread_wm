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
import time
import multiprocessing as mp
import traceback
import queue as mp_queue
from copy import deepcopy
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed, move_to_device
from collections import deque, OrderedDict

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
        current_task_id: int = None,
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
        self.forward_transfer_metrics = None
        self.current_task_id = current_task_id
        self.is_training_mode = True
        self.final_mpc_visual_loss = None

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
        self.online_learner = None
        self.is_lora_enabled = self.cfg_dict.get("lora", {}).get("enabled", False)
        self.is_online_lora = self.cfg_dict.get("lora", {}).get("online", False)

        if self.is_lora_enabled:
            use_ensemble_lora = self.cfg_dict.get("lora", {}).get("ensemble", False)
            if use_ensemble_lora:
                print("INFO: Ensemble LoRA training enabled. Initializing EnsembleOnlineLora module.")
                self.online_learner = EnsembleOnlineLora(workspace=self)
            else:
                print("INFO: Standard LoRA training enabled. Initializing OnlineLora module.")
                self.online_learner = OnlineLora(workspace=self) # 온라인 모드 강제
            if hasattr(self.online_learner, "check_task_change") and current_task_id is not None:
                try:
                    changed = self.online_learner.check_task_change(current_task_id)
                    print(f"🔄 Task change flag updated via OnlineLora.check_task_change: {changed}")
                except Exception as e:
                    print(f"⚠️  Failed to set task change flag: {e}")

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
        planner_kwargs = {
            "wm": self.wm,
            "env": self.env,
            "action_dim": self.action_dim,
            "objective_fn": objective_fn,
            "preprocessor": self.data_preprocessor,
            "evaluator": self.evaluator,
            "wandb_run": self.wandb_run,
            "log_filename": self.log_filename,
        }

        if (
            self.is_online_lora
            and hasattr(self.online_learner, "ensemble_manager")
            and self.online_learner.ensemble_manager is not None
        ):
            planner_kwargs["ensemble_manager"] = self.online_learner.ensemble_manager
            print("🔧 Passing ensemble manager to planner")

        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            **planner_kwargs,
        )

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            # Only set horizon to goal_H, but keep n_taken_actions from config
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            # n_taken_actions is already set from config, don't override it
            # self.planner.n_taken_actions = cfg_dict["goal_H"]  # Commented out to use config value
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

    def evaluate_loss_only(self, label="loss_eval", max_iter_override=None):
        # Prepare evaluation actions without running MPC planning
        if self.gt_actions is not None:
            actions_eval = (
                self.gt_actions.to(device=self.device, dtype=torch.float32).detach()
            )
            action_len = np.full(actions_eval.shape[0], actions_eval.shape[1])
        else:
            batch_size = self.obs_0["visual"].shape[0]
            horizon = 1 if self.goal_H <= 0 else self.goal_H
            actions_eval = torch.zeros(
                (batch_size, horizon, self.action_dim),
                device=self.device,
                dtype=torch.float32,
            )
            action_len = np.full(batch_size, horizon)

        self.evaluator.force_recenter_for_next_rollout()
        logs, successes, e_obses, _ = self.evaluator.eval_actions(
            actions_eval,
            action_len,
            save_video=False,
            filename=label,
            learning_enabled=False,
        )

        metrics = None
        if (
            self.is_online_lora
            and hasattr(self, "online_learner")
            and hasattr(self.online_learner, "compute_loss_only")
        ):
            trans_obs_0 = move_to_device(
                self.data_preprocessor.transform_obs(self.obs_0), self.device
            )
            metrics = self.online_learner.compute_loss_only(
                trans_obs_0, actions_eval, e_obses
            )

        result = {
            "task_id": self.current_task_id,
            "label": label,
            "logs": {f"{label}/{k}": v for k, v in logs.items()},
            "successes": successes.tolist() if hasattr(successes, "tolist") else successes,
            "metrics": metrics or {},
        }
        return result

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None

        if (
            self.is_online_lora
            and hasattr(self.online_learner, "task_changed")
        ):
            task_changed = self.online_learner.task_changed
        else:
            task_changed = False

        actions, action_len = None, None
        self.final_mpc_visual_loss = None

        ensemble_cfg = self.cfg_dict.get("lora", {}).get("ensemble_cfg", {})
        inference_cfg = ensemble_cfg.get("inference", {})
        usage_strategy = inference_cfg.get("usage_strategy", "task_change_only")

        use_ensemble_selection = (
            usage_strategy in ["task_change_only", "always"]
            and self.is_online_lora
            and hasattr(self.online_learner, "ensemble_manager")
            and (
                len(getattr(self.online_learner.ensemble_manager, "ensemble_members", {})) > 0
                or (self.current_task_id or 1) == 1
            )
            and (
                usage_strategy == "always"
                or task_changed
            )
        )

        # 🔬 대조군 모드 확인
        no_ensemble_control = self.cfg_dict.get("lora", {}).get("no_ensemble_control", False)
        
        if use_ensemble_selection:
            if no_ensemble_control:
                # 대조군 모드: 앙상블 평가 없이 최근 멤버만 사용
                print("🔬 Control Group Mode: Using latest member without ensemble evaluation...")
                try:
                    if hasattr(self.online_learner, "apply_latest_member_without_evaluation"):
                        self.online_learner.apply_latest_member_without_evaluation(self)
                    else:
                        print("⚠️  apply_latest_member_without_evaluation method not found")
                except Exception as e:
                    print(f"⚠️  Control Group: Failed to apply latest member: {e}")
            else:
                # 실험군 모드: 앙상블 평가로 최적 멤버 선택
                if usage_strategy == "always":
                    print("🔄 Using ensemble for optimal member selection (always mode)...")
                else:
                    print("🔄 Task changed! Using ensemble for optimal member selection...")

                try:
                    self.online_learner.perform_task_change_ensemble_selection(self)
                except Exception as e:
                    print(f"⚠️  Ensemble selection failed: {e}")

            if hasattr(self.online_learner, "reset_task_changed_flag"):
                try:
                    self.online_learner.reset_task_changed_flag()
                except Exception as e:
                    print(f"⚠️  Failed to reset task_changed flag: {e}")

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
        self.planning_iteration_metrics = getattr(self.planner, "iteration_metrics", [])
        self.evaluated_iterations_count = getattr(self.planner, "evaluated_iterations_count", 0)
        print(f"🔍 perform_planning(): planning_iteration_metrics count = {len(self.planning_iteration_metrics) if self.planning_iteration_metrics else 0}, evaluated iterations = {self.evaluated_iterations_count}")
        return logs


def _task_worker(
    task_idx: int,
    task_cfg_dict: dict,
    overrides: dict,
    base_env_overrides: dict,
    model_cfg_path: str,
    mpc_iters_per_task: int,
    initial_ensemble_state,
    result_queue,
    mode: str = "train",
    eval_metadata: dict | None = None,
):
    os.environ.setdefault("WANDB_MODE", "disabled")
    wandb_run = None
    env = None
    plan_workspace = None
    try:
        model_cfg = OmegaConf.load(model_cfg_path)

        seed(task_cfg_dict["seed"])
        _, dset = hydra.utils.call(
            model_cfg.env.dataset,
            num_hist=model_cfg.num_hist,
            num_pred=model_cfg.num_pred,
            frameskip=model_cfg.frameskip,
        )
        dset = dset["valid"]

        num_action_repeat = model_cfg.num_action_repeat
        model_path = Path(model_cfg_path).parent
        model_ckpt = (
            Path(model_path) / "checkpoints" / f"model_{task_cfg_dict['model_epoch']}.pth"
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

        env_args = list(model_cfg.env.args) if model_cfg.env.args is not None else []
        env_kwargs = (
            OmegaConf.to_container(model_cfg.env.kwargs, resolve=True)
            if model_cfg.env.kwargs is not None
            else {}
        )

        env_override = {**(base_env_overrides or {}), **overrides}
        n_evals = task_cfg_dict["n_evals"]
        task_start = time.time()

        if model_cfg.env.name in ("wall", "deformable_env"):
            from env.serial_vector_env import SerialVectorEnv

            envs = [
                gym.make(
                    model_cfg.env.name,
                    *env_args,
                    **env_kwargs,
                    **env_override,
                )
                for _ in range(n_evals)
            ]
            env = SerialVectorEnv(envs)
        else:
            env = SubprocVectorEnv(
                [
                    (lambda env_override=env_override: gym.make(
                        model_cfg.env.name,
                        *env_args,
                        **env_kwargs,
                        **env_override,
                    ))
                    for _ in range(n_evals)
                ]
            )

        task_cfg_dict_local = deepcopy(task_cfg_dict)
        planner_cfg = task_cfg_dict_local.setdefault("planner", {})
        logging_prefix = f"task_{task_idx + 1:02d}_plan"
        planner_cfg["logging_prefix"] = logging_prefix
        if "sub_planner" in planner_cfg and isinstance(planner_cfg["sub_planner"], dict):
            planner_cfg["sub_planner"]["logging_prefix"] = logging_prefix
        task_cfg_dict_local["wandb_logging"] = False

        plan_workspace = PlanWorkspace(
            cfg_dict=task_cfg_dict_local,
            wm=model,
            dset=dset,
            env=env,
            env_name=model_cfg.env.name,
            frameskip=model_cfg.frameskip,
            wandb_run=wandb_run,
            current_task_id=task_idx + 1,
        )

        mode = (mode or "train").lower()
        eval_metadata = eval_metadata or {}
        if mode not in {"train", "evaluate"}:
            print(f"⚠️  Unknown worker mode '{mode}', defaulting to 'train'.")
            mode = "train"
        is_training = mode == "train"
        plan_workspace.is_training_mode = is_training

        if not is_training:
            forced_lora_path = eval_metadata.get("forced_lora_path")
            if forced_lora_path:
                try:
                    if os.path.exists(forced_lora_path):
                        lora_weights = torch.load(forced_lora_path, map_location=device)
                        learner = getattr(plan_workspace, "online_learner", None)
                        if (
                            plan_workspace.is_online_lora
                            and learner is not None
                            and hasattr(learner, "_apply_lora_weights")
                        ):
                            success = learner._apply_lora_weights(lora_weights)
                            if success and hasattr(learner, "last_selected_member_task_id"):
                                setattr(
                                    learner,
                                    "last_selected_member_task_id",
                                    eval_metadata.get("forced_lora_task_id"),
                                )
                            print(
                                f"📥 Applied forced LoRA weights from {forced_lora_path} "
                                f"for task_id={eval_metadata.get('forced_lora_task_id')}"
                            )
                        else:
                            print(
                                f"⚠️  Forced LoRA weights provided but could not be applied "
                                f"(is_online_lora={plan_workspace.is_online_lora}, learner={type(learner) if learner else None})"
                            )
                    else:
                        print(f"⚠️  Forced LoRA path not found: {forced_lora_path}")
                        result_queue.put(
                            {
                                "status": "error",
                                "error": f"forced_lora_path not found: {forced_lora_path}",
                                "task_id": task_idx + 1,
                                "mode": mode,
                            }
                        )
                        return
                except Exception as e:
                    print(f"⚠️  Failed to apply forced LoRA weights from {forced_lora_path}: {e}")
                    result_queue.put(
                        {
                            "status": "error",
                            "error": f"failed to load forced LoRA weights: {e}",
                            "task_id": task_idx + 1,
                            "mode": mode,
                        }
                    )
                    return

        if (
            initial_ensemble_state
            and plan_workspace.is_online_lora
            and hasattr(plan_workspace.online_learner, "ensemble_manager")
            and plan_workspace.online_learner.ensemble_manager is not None
        ):
            manager = plan_workspace.online_learner.ensemble_manager
            members_meta = initial_ensemble_state.get("members_meta", [])
            manager.ensemble_members = OrderedDict()
            manager.memory_usage = 0.0
            if hasattr(manager, "access_frequency"):
                manager.access_frequency = {}

            for member in members_meta:
                task_id_meta = int(member["task_id"])
                file_path = member.get("file_path")
                try:
                    if not file_path or not os.path.exists(file_path):
                        print(f"⚠️  Cached LoRA file missing for task {task_id_meta}: {file_path}")
                        continue
                    weights = torch.load(file_path, map_location=device)
                    size_mb = manager._calculate_lora_size(weights)
                    member_info = {
                        "task_id": task_id_meta,
                        "lora_weights": weights,
                        "performance": member.get("performance", {}),
                        "metadata": member.get("metadata", {}),
                        "created_at": time.time(),
                        "last_accessed": time.time(),
                        "access_count": 0,
                        "size_mb": size_mb,
                        "cached_path": file_path,
                    }
                    manager.ensemble_members[task_id_meta] = member_info
                    manager.memory_usage += size_mb
                    if hasattr(manager, "access_frequency"):
                        manager.access_frequency[task_id_meta] = member.get("access_count", 0)
                except Exception as e:
                    print(f"⚠️  Failed to load cached LoRA member for task {task_id_meta}: {e}")

            active_member_task_id = initial_ensemble_state.get("active_member_task_id")
            # forced_lora_path를 사용한 경우에는 이미 가중치가 적용되었으므로 건너뜀
            forced_lora_used = eval_metadata and eval_metadata.get("forced_lora_path")
            if (
                not forced_lora_used
                and active_member_task_id is not None
                and active_member_task_id in manager.ensemble_members
            ):
                try:
                    plan_workspace.online_learner._apply_lora_weights(
                        manager.ensemble_members[active_member_task_id]["lora_weights"]
                    )
                    setattr(
                        plan_workspace.online_learner,
                        "last_selected_member_task_id",
                        active_member_task_id,
                    )
                    print(
                        f"🔁 Applied ensemble member {active_member_task_id} at task start"
                    )
                except Exception as e:
                    print(f"⚠️  Failed to apply initial ensemble member: {e}")

        if hasattr(plan_workspace, "planner"):
            if hasattr(plan_workspace.planner, "max_iter"):
                plan_workspace.planner.max_iter = mpc_iters_per_task
            if hasattr(plan_workspace.planner, "iter"):
                plan_workspace.planner.iter = 0

        if not is_training:
            eval_label = eval_metadata.get("label") or f"evaluation_task_{task_idx + 1}"
            max_iter_override = eval_metadata.get("max_iter", 1)
            try:
                evaluation_result = plan_workspace.evaluate_loss_only(
                    label=eval_label,
                    max_iter_override=max_iter_override,
                )
                evaluation_result["metadata"] = {
                    "mode": mode,
                    **eval_metadata,
                    "task_id": plan_workspace.current_task_id,
                }
                result_queue.put(
                    {
                        "status": "success",
                        "evaluation": evaluation_result,
                    }
                )
            except Exception:
                result_queue.put(
                    {
                        "status": "error",
                        "error": traceback.format_exc(),
                        "task_id": task_idx + 1,
                        "mode": mode,
                    }
                )
            return

        task_lora_stacks = 0

        try:
            if (
                plan_workspace.is_online_lora
                and hasattr(plan_workspace.online_learner, "hybrid_enabled")
                and plan_workspace.online_learner.hybrid_enabled
                and hasattr(plan_workspace.online_learner, "ensemble_manager")
                and plan_workspace.online_learner.ensemble_manager is not None
            ):
                hybrid_cfg = plan_workspace.cfg_dict.get("lora", {}).get("hybrid_stacking", {})
                force_initial = hybrid_cfg.get("force_initial_stacking", True)
                num_members = len(plan_workspace.online_learner.ensemble_manager.ensemble_members)
                stacks_in_current = getattr(plan_workspace.online_learner, "stacks_in_current_task", 0)
                if force_initial and num_members == 0 and stacks_in_current == 0:
                    print(f"\n🎯 Forcing initial LoRA stacking for Task {task_idx + 1} (ensemble_initial)")
                    success = plan_workspace.online_learner.trigger_task_based_stacking(task_idx + 1, "ensemble_initial")
                    if success:
                        task_lora_stacks += 1
                        print(f"✅ Initial ensemble-based LoRA stacking completed. Total stacks in task: {task_lora_stacks}")
                    else:
                        print("⚠️  Initial ensemble-based LoRA stacking skipped or failed.")
        except Exception as e:
            print(f"⚠️  Failed to perform forced initial stacking: {e}")

        if plan_workspace.is_online_lora and hasattr(plan_workspace.online_learner, "base_online_lora"):
            base_online_lora = plan_workspace.online_learner.base_online_lora
            old_cb = getattr(base_online_lora, "on_lora_stack_callback", None)

            def on_lora_stack(steps, loss, task_id, stack_type, reason):
                nonlocal task_lora_stacks
                if callable(old_cb):
                    try:
                        old_cb(steps, loss, task_id, stack_type, reason)
                    except Exception as e:
                        print(f"⚠️  Error in chained base callback: {e}")
                task_lora_stacks += 1
                loss_str = f"{loss:.6f}" if loss is not None else "N/A"
                selected_member_id = getattr(plan_workspace.online_learner, "last_selected_member_task_id", None)
                if selected_member_id is not None:
                    print(
                        f"🔥 LoRA Stack #{task_lora_stacks} at step {steps} "
                        f"(task {task_id}, type {stack_type}, reason {reason}, on_member {selected_member_id}) loss {loss_str}"
                    )
                else:
                    print(
                        f"🔥 LoRA Stack #{task_lora_stacks} at step {steps} "
                        f"(task {task_id}, type {stack_type}, reason {reason}) loss {loss_str}"
                    )
                try:
                    history = getattr(plan_workspace.online_learner, "stack_history", None)
                    if history and isinstance(history[-1], dict):
                        history[-1]["selected_member_task_id"] = selected_member_id
                except Exception:
                    pass

            base_online_lora.on_lora_stack_callback = on_lora_stack

        if (
            plan_workspace.is_online_lora
            and hasattr(plan_workspace.online_learner, "trigger_task_based_stacking")
            and hasattr(plan_workspace.online_learner, "hybrid_enabled")
            and plan_workspace.online_learner.hybrid_enabled
            and hasattr(plan_workspace.online_learner, "task_based_stacking")
            and plan_workspace.online_learner.task_based_stacking
        ):
            print(f"\n🎯 Triggering task-based LoRA stacking for Task {task_idx + 1} (adds new LoRA layers to model)")
            try:
                success = plan_workspace.online_learner.trigger_task_based_stacking(task_idx + 1, "task_change")
                if success:
                    task_lora_stacks += 1
                    print(f"✅ Task-based LoRA stacking successful. Total stacks in task: {task_lora_stacks}")
                else:
                    print("Task-based LoRA stacking skipped or failed.")
            except Exception as e:
                print(f"Task-based stacking failed: {e}")
        else:
            if plan_workspace.is_online_lora:
                if hasattr(plan_workspace.online_learner, "ensemble_manager"):
                    print(
                        f"Task-based LoRA stacking disabled "
                        f"(hybrid_enabled: {getattr(plan_workspace.online_learner, 'hybrid_enabled', False)}, "
                        f"task_based_stacking: {getattr(plan_workspace.online_learner, 'task_based_stacking', False)})"
                    )
                else:
                    print("⚠️  Standard OnlineLora mode - using default LoRA stacking behavior")

        print(
            f"--- Task {task_idx + 1} | Running MPC until iteration {mpc_iters_per_task} ---"
        )
        logs = plan_workspace.perform_planning()
        prefixed_logs = {f"task_{task_idx + 1}/{k}": v for k, v in logs.items()}
        log_entries = [prefixed_logs]
        forward_transfer_metrics = getattr(plan_workspace, "forward_transfer_metrics", None)
        if forward_transfer_metrics:
            ft_logs_prefixed = {
                f"task_{task_idx + 1}/forward_transfer/{k}": v
                for k, v in forward_transfer_metrics.items()
            }
            log_entries.append(ft_logs_prefixed)

        actual_mpc_iters = getattr(plan_workspace.planner, "iter", None)
        task_planning_steps = actual_mpc_iters if actual_mpc_iters is not None else 0

        try:
            if (
                plan_workspace.is_online_lora
                and hasattr(plan_workspace, "online_learner")
                and hasattr(plan_workspace.online_learner, "save_current_lora_member")
                and getattr(plan_workspace.online_learner, "save_on_task_end", True)
            ):
                if hasattr(plan_workspace.online_learner, "update_and_reset_lora_parameters"):
                    try:
                        plan_workspace.online_learner.update_and_reset_lora_parameters()
                        print(f"🔄 Updated w with wnew for Task {task_idx + 1}")
                    except Exception as e:
                        print(f"⚠️  update_and_reset_lora_parameters failed: {e}")

                current_task_for_save = task_idx + 1
                print(f"💾 Saving finalized LoRA member for Task {current_task_for_save} at task end...")
                plan_workspace.online_learner.save_current_lora_member(task_id=current_task_for_save, reason="task_end")
        except Exception as e:
            print(f"⚠️  Failed to save finalized LoRA member at task end: {e}")

        final_mpc_visual = getattr(plan_workspace, "final_mpc_visual_loss", None)

        planning_iteration_metrics = getattr(plan_workspace, "planning_iteration_metrics", [])
        evaluated_iterations_count = getattr(plan_workspace, "evaluated_iterations_count", 0)
        print(f"🔍 Task {task_idx + 1}: planning_iteration_metrics count = {len(planning_iteration_metrics) if planning_iteration_metrics else 0}, evaluated iterations = {evaluated_iterations_count}")

        summary = {
            "task_id": task_idx + 1,
            "overrides": overrides,
            "planning_steps": task_planning_steps,
            "lora_stacks": task_lora_stacks,
            "stack_history": getattr(plan_workspace.online_learner, "stack_history", []) if plan_workspace.is_online_lora else [],
            "duration_seconds": time.time() - task_start,
            "mpc_iterations": planning_iteration_metrics,
            "evaluated_iterations_count": evaluated_iterations_count,
        }
        if forward_transfer_metrics:
            summary["forward_transfer"] = forward_transfer_metrics
        if final_mpc_visual:
            summary["final_mpc_visual_loss"] = final_mpc_visual
            log_entries.append(
                {
                    f"task_{task_idx + 1}/mpc/final_visual_loss": final_mpc_visual.get("value"),
                }
            )

        if (
            plan_workspace.is_online_lora
            and hasattr(plan_workspace.online_learner, "last_ensemble_evaluation_summary")
        ):
            ensemble_eval_summary = getattr(
                plan_workspace.online_learner, "last_ensemble_evaluation_summary", None
            )
            if ensemble_eval_summary:
                summary["ensemble_evaluation"] = ensemble_eval_summary

        ensemble_state = None
        if (
            plan_workspace.is_online_lora
            and hasattr(plan_workspace.online_learner, "ensemble_manager")
            and plan_workspace.online_learner.ensemble_manager is not None
        ):
            manager = plan_workspace.online_learner.ensemble_manager
            members_meta = []
            for task_id_member, member_info in manager.ensemble_members.items():
                file_path = os.path.join(
                    manager.cache_dir, f"lora_task_{task_id_member}.pth"
                )
                try:
                    manager._save_member_to_disk(task_id_member, member_info)
                except Exception as e:
                    print(
                        f"⚠️  Failed to save ensemble member {task_id_member} to disk: {e}"
                    )
                members_meta.append(
                    {
                        "task_id": task_id_member,
                        "file_path": file_path,
                        "performance": member_info.get("performance", {}),
                        "metadata": member_info.get("metadata", {}),
                        "access_count": member_info.get("access_count", 0),
                    }
                )
            ensemble_state = {
                "members_meta": members_meta,
                "active_member_task_id": getattr(
                    plan_workspace.online_learner,
                    "last_selected_member_task_id",
                    None,
                ),
            }

        result_queue.put(
            {
                "status": "success",
                "logs": log_entries,
                "summary": summary,
                "ensemble_state": ensemble_state,
            }
        )
    except Exception:
        result_queue.put(
            {
                "status": "error",
                "error": traceback.format_exc(),
                "task_id": task_idx + 1,
            }
        )
    finally:
        if env is not None:
            if hasattr(env, "envs"):
                for e in env.envs:
                    close_fn = getattr(e, "close", None)
                    if callable(close_fn):
                        try:
                            close_fn()
                        except Exception:
                            pass
            else:
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
        try:
            import pyflex

            if hasattr(pyflex, "is_initialized") and pyflex.is_initialized():
                pyflex.clean()
        except Exception:
            pass


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            # None 체크 추가 (dummy checkpoint 지원)
            if v is not None:
                result[k] = v.to(device)
            else:
                result[k] = None
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

    base_env_overrides = cfg_dict.get("environment", {}) or {}
    if base_env_overrides:
        print(f"Base environment config: {base_env_overrides}")

    # 연속 학습 태스크 시퀀스 정의
    task_sequence = [
        {"table_color": "purple", "camera_view": 0},
        {"table_color": "brown"},
        {"granular_radius": 0.25, "camera_view": 0},
        {"table_color": "brown", "granular_radius": 0.25},
        {"camera_view": 0},
        {"table_color": "purple"},
        {"table_color": "brown", "camera_view": 0},
        {"table_color": "purple", "granular_radius": 0.25},
    ]
    mpc_iters_per_task = 25    # 각 태스크당 최대 MPC iteration 수
    overall_logs = []
    task_summaries = []
    forgetting_history = []
    forward_transfer_history = []
    ensemble_evaluation_history = []

    ensemble_cfg = cfg_dict.get("lora", {}).get("ensemble_cfg", {})
    ensemble_cache_dir = os.path.abspath(ensemble_cfg.get("cache_dir", "./lora_cache"))

    ctx = mp.get_context("spawn")
    timeout_seconds = cfg_dict.get("task_timeout_seconds", 1000) # 자식 프로세스를 기다리는 시간
    failures = []
    wandb_global_step = 0
    ensemble_forgetting = cfg_dict.get("lora", {}).get("ensemble_forgetting", False)

    def _to_scalar(value):
        if value is None:
            return None
        if isinstance(value, (np.floating, float, int, np.integer)):
            return float(value)
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            return float(np.mean(value))
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return float(np.mean(value))
        return None

    def _log_to_wandb(name, value):
        nonlocal wandb_global_step
        if wandb_run is None:
            return
        scalar = _to_scalar(value)
        if scalar is None:
            return
        try:
            wandb_run.log({name: scalar}, step=wandb_global_step)
            wandb_global_step += 1
        except (ValueError, TypeError):
            pass

    def _format_metric(value):
        if isinstance(value, (int, float, np.floating)):
            return f"{float(value):.4f}"
        return value

    def run_forgetting_evaluation(prev_idx, initial_state, mode_label, forced_lora_path=None):
        cf_overrides = task_sequence[prev_idx]
        cf_task_cfg = deepcopy(cfg_dict)
        cf_planner_cfg = cf_task_cfg.setdefault("planner", {})
        cf_prefix = f"cf_task_{prev_idx + 1:02d}_eval"
        cf_planner_cfg["logging_prefix"] = cf_prefix
        if "sub_planner" in cf_planner_cfg and isinstance(cf_planner_cfg["sub_planner"], dict):
            cf_planner_cfg["sub_planner"]["logging_prefix"] = cf_prefix
        cf_task_cfg["wandb_logging"] = False

        cf_queue = ctx.Queue()
        eval_metadata = {
            "kind": "catastrophic_forgetting",
            "source_task": task_idx + 1,
            "target_task": prev_idx + 1,
            "label": f"cf_task_{prev_idx + 1}_after_{task_idx + 1}_{mode_label}",
            "max_iter": 1,
            "mode": mode_label,
            "use_ensemble_member": bool(initial_state),
            "forced_lora_path": forced_lora_path,
            "forced_lora_task_id": task_idx + 1 if mode_label == "current_model" else (prev_idx + 1),
        }
        cf_proc = ctx.Process(
            target=_task_worker,
            args=(
                prev_idx,
                cf_task_cfg,
                cf_overrides,
                base_env_overrides,
                model_cfg_path,
                1,
                initial_state,
                cf_queue,
            ),
            kwargs={
                "mode": "evaluate",
                "eval_metadata": eval_metadata,
            },
        )
        cf_proc.start()
        cf_proc.join(timeout_seconds)

        if cf_proc.is_alive():
            print(
                f"⏱️  CF evaluation ({mode_label}) for Task {prev_idx + 1} timed out after {timeout_seconds} seconds. Terminating process."
            )
            cf_proc.terminate()
            cf_proc.join()
            cf_queue.close()
            cf_queue.join_thread()
            return {
                "task_id": prev_idx + 1,
                "source_task": task_idx + 1,
                "status": "timeout",
                "mode": mode_label,
                "evaluation": None,
                "error": "timeout",
            }, None

        try:
            cf_result = cf_queue.get_nowait()
        except mp_queue.Empty:
            cf_result = {
                "status": "error",
                "error": "No result received",
                "task_id": prev_idx + 1,
            }
        finally:
            cf_queue.close()
            cf_queue.join_thread()

        if cf_result.get("status") == "success":
            evaluation_payload = cf_result.get("evaluation", {})
            evaluation_payload["mode"] = mode_label
            metrics = evaluation_payload.get("metrics", {})
            
            # 로드한 파일명 추출
            loaded_file = "N/A"
            if forced_lora_path:
                loaded_file = os.path.basename(forced_lora_path) if os.path.exists(forced_lora_path) else f"{os.path.basename(forced_lora_path)} (not found)"
            else:
                loaded_file = "default model (no LoRA file)"
            
            print(
                f"🧠 CF evaluation ({mode_label}) Task {prev_idx + 1} after Task {task_idx + 1}: "
                f"total_loss={metrics.get('total_loss')} visual_loss={metrics.get('visual_loss')} "
                f"[loaded: {loaded_file}]"
            )
            eval_logs = evaluation_payload.get("logs", {})
            return {
                "task_id": prev_idx + 1,
                "source_task": task_idx + 1,
                "status": "success",
                "mode": mode_label,
                "evaluation": evaluation_payload,
                "loaded_file": loaded_file,
            }, eval_logs
        else:
            print(
                f"⚠️  CF evaluation ({mode_label}) for Task {prev_idx + 1} failed: {cf_result.get('error')}"
            )
            return {
                "task_id": prev_idx + 1,
                "source_task": task_idx + 1,
                "status": cf_result.get("status"),
                "error": cf_result.get("error"),
                "mode": mode_label,
                "evaluation": None,
            }, None

    model_cfg_path = os.path.join(model_path, "hydra.yaml")

    current_ensemble_state = {
        "members": {},
        "memory_usage": 0.0,
        "access_frequency": {},
        "active_member_task_id": None,
    }

    for task_idx, overrides in enumerate(task_sequence):
        print(f"\n{'=' * 25} Starting Granular Task {task_idx + 1}/{len(task_sequence)} {'=' * 25}")
        combined_overrides = {**base_env_overrides, **overrides}
        print(f"Environment overrides: {combined_overrides}")

        task_cfg_dict = deepcopy(cfg_dict)
        planner_cfg = task_cfg_dict.setdefault("planner", {})
        logging_prefix = f"task_{task_idx + 1:02d}_plan"
        planner_cfg["logging_prefix"] = logging_prefix
        if "sub_planner" in planner_cfg and isinstance(planner_cfg["sub_planner"], dict):
            planner_cfg["sub_planner"]["logging_prefix"] = logging_prefix

        queue = ctx.Queue()
        proc = ctx.Process(
            target=_task_worker,
            args=(
                task_idx,
                task_cfg_dict,
                overrides,
                base_env_overrides,
                model_cfg_path,
                mpc_iters_per_task,
                current_ensemble_state,
                queue,
            ),
        )
        proc.start()
        proc.join(timeout_seconds)

        if proc.is_alive():
            print(f"⏱️  Task {task_idx + 1} timed out after {timeout_seconds} seconds. Terminating process.")
            proc.terminate()
            proc.join()
            failures.append({"task_id": task_idx + 1, "status": "timeout"})
            queue.close()
            queue.join_thread()
            continue

        result = None
        try:
            result = queue.get_nowait()
        except mp_queue.Empty:
            result = {"status": "error", "error": "No result received", "task_id": task_idx + 1}
        finally:
            queue.close()
            queue.join_thread()

        if result["status"] == "success":
            overall_logs.extend(result.get("logs", []))
            summary = result.get("summary", {})
            task_summaries.append(summary)
            ensemble_state = result.get("ensemble_state")
            if ensemble_state is not None:
                current_ensemble_state = ensemble_state

            forward_transfer = summary.get("forward_transfer")
            if isinstance(forward_transfer, dict):
                print(
                    f"🧪 Forward transfer summary for Task {task_idx + 1}: "
                    f"total_loss={forward_transfer.get('total_loss')} "
                    f"visual_loss={forward_transfer.get('visual_loss')}"
                )
                forward_transfer_history.append(
                    {
                        "task_id": summary.get("task_id"),
                        "metrics": forward_transfer,
                    }
                )

            iteration_metrics = summary.get("mpc_iterations", [])
            print(f"🔍 Task {summary['task_id']}: iteration_metrics count = {len(iteration_metrics) if iteration_metrics else 0}")

            final_mpc_visual = summary.get("final_mpc_visual_loss")
            if isinstance(final_mpc_visual, dict):
                value = final_mpc_visual.get("value")
                print(
                    f"🎯 Final MPC visual loss for Task {task_idx + 1}: {value}"
                )
                overall_logs.append(
                    {
                        f"task_{summary['task_id']}/mpc/final_visual_loss": value
                    }
                )

            if wandb_run is not None:
                task_prefix = f"task_{summary['task_id']:02d}/mpc"
                if iteration_metrics:
                    print(f"📊 Logging {len(iteration_metrics)} MPC iterations for Task {summary['task_id']}")
                    logged_count = 0
                    for iter_entry in iteration_metrics:
                        iter_idx = iter_entry.get("iter")
                        iter_logs = iter_entry.get("logs", {})
                        
                        # 🔧 visual_loss 찾기: prefix가 붙은 키에서도 찾기
                        visual_loss_value = None
                        chamfer_value = None
                        
                        for key, value in iter_logs.items():
                            if key == "step":
                                continue
                            scalar = _to_scalar(value)
                            if scalar is None:
                                continue
                            key_lower = key.lower()
                            
                            # 🔧 visual_loss 찾기 (한 번만)
                            if visual_loss_value is None and "visual" in key_lower and "loss" in key_lower and "dist" not in key_lower:
                                visual_loss_value = scalar
                            
                            # 🔧 chamfer_distance 찾기 (한 번만)
                            if chamfer_value is None and "chamfer" in key_lower:
                                chamfer_value = scalar
                        
                        # 🔧 visual_loss가 있으면 로깅
                        if visual_loss_value is not None:
                            _log_to_wandb(f"{task_prefix}/visual_loss", visual_loss_value)
                            _log_to_wandb("visual_loss", visual_loss_value)
                            logged_count += 1
                        
                        # 🔧 chamfer_distance가 있으면 로깅
                        if chamfer_value is not None:
                            _log_to_wandb(f"{task_prefix}/chamfer_distance", chamfer_value)
                            _log_to_wandb("chamfer_distance", chamfer_value)
                            logged_count += 1
                    print(f"✅ Logged {logged_count} metrics to wandb for Task {summary['task_id']}")
                else: # 들여쓰기 오류
                    print(f"⚠️  No iteration_metrics found for Task {summary['task_id']} (summary keys: {list(summary.keys())})")

            ensemble_eval = summary.get("ensemble_evaluation")
            if isinstance(ensemble_eval, dict):
                entry = {"task_id": summary.get("task_id"), **ensemble_eval}
                ensemble_evaluation_history.append(entry)

                ensemble_log = {}
                members = ensemble_eval.get("members", [])
                if isinstance(members, list):
                    for member in members:
                        if not isinstance(member, dict):
                            continue
                        member_task = member.get("task_id")
                        if member_task is None:
                            continue
                        if "loss" in member:
                            ensemble_log[
                                f"task_{summary['task_id']}/ensemble/member_{member_task}_loss"
                            ] = member["loss"]
                best_member = ensemble_eval.get("best_member", {})
                best_perf = best_member.get("performance", {}) if isinstance(best_member, dict) else {}
                if isinstance(best_perf, dict) and "loss" in best_perf:
                    ensemble_log[f"task_{summary['task_id']}/ensemble/best_loss"] = best_perf["loss"]
                if "stacking_applied" in ensemble_eval:
                    ensemble_log[f"task_{summary['task_id']}/ensemble/stacking_applied"] = int(bool(ensemble_eval["stacking_applied"]))
                if "stacking_triggered" in ensemble_eval:
                    ensemble_log[f"task_{summary['task_id']}/ensemble/stacking_triggered"] = int(bool(ensemble_eval["stacking_triggered"]))
                if ensemble_log:
                    overall_logs.append(ensemble_log)

            if task_idx > 0:
                cf_results = []
                # Evaluate current model (without loading ensemble member)
                for prev_idx in range(task_idx):
                    current_model_lora_path = os.path.join(
                        ensemble_cache_dir, f"lora_task_{task_idx + 1}.pth"
                    )
                    result_current, logs_current = run_forgetting_evaluation(
                        prev_idx,
                        None,
                        "current_model",
                        forced_lora_path=current_model_lora_path,
                    )
                    cf_results.append(result_current)
                    if logs_current:
                        overall_logs.append(
                            {
                                f"task_{prev_idx + 1}/{k}": v
                                for k, v in logs_current.items()
                            }
                        )
                    # Optionally evaluate with saved ensemble member
                    if ensemble_forgetting and current_ensemble_state:
                        ensemble_member_lora_path = os.path.join(
                            ensemble_cache_dir, f"lora_task_{prev_idx + 1}.pth"
                        )
                        result_member, logs_member = run_forgetting_evaluation(
                            prev_idx,
                            current_ensemble_state,
                            "ensemble_member",
                            forced_lora_path=ensemble_member_lora_path,
                        )
                        cf_results.append(result_member)
                        if logs_member:
                            overall_logs.append(
                                {
                                    f"task_{prev_idx + 1}/{k}": v
                                    for k, v in logs_member.items()
                                }
                            )

                forgetting_history.append(
                    {
                        "source_task": task_idx + 1,
                        "results": cf_results,
                    }
                )

            if wandb_run is not None:
                for log_entry in result.get("logs", []):
                    for key, value in log_entry.items():
                        if "visual" in key and "loss" in key and "dist" not in key:
                            _log_to_wandb("visual_loss", value)
                        if "chamfer" in key:
                            _log_to_wandb("chamfer_distance", value)

                if isinstance(summary.get("forward_transfer"), dict):
                    ft = summary["forward_transfer"]
                    if "visual_loss" in ft:
                        _log_to_wandb("visual_loss", ft["visual_loss"])
                    if "chamfer_distance" in ft:
                        _log_to_wandb("chamfer_distance", ft["chamfer_distance"])
        else:
            failures.append(result)
            print(f"⚠️  Task {task_idx + 1} failed with status {result['status']}: {result.get('error')}")

    print(f"\n{'=' * 25} Granular Continual Planning Finished {'=' * 25}")
    for summary in task_summaries:
        lora_info = f", LoRA stacks={summary['lora_stacks']}" if "lora_stacks" in summary else ""
        ft_metrics = summary.get("forward_transfer", {})
        ft_info = ""
        if ft_metrics:
            ft_info = (
                f", FT total_loss={ft_metrics.get('total_loss')}"
                f" visual_loss={ft_metrics.get('visual_loss')}"
                f" proprio_loss={ft_metrics.get('proprio_loss')}"
            )
        final_mpc_info = ""
        final_mpc_visual = summary.get("final_mpc_visual_loss")
        if final_mpc_visual is not None:
            if isinstance(final_mpc_visual, dict):
                final_value = final_mpc_visual.get("value")
                if final_value is not None:
                    final_key = final_mpc_visual.get("key", "unknown")
                    final_iter = final_mpc_visual.get("iteration", "unknown")
                    final_value = _format_metric(final_value)
                    final_mpc_info = f", MPC final_visual_loss={final_value} (key={final_key}, iter={final_iter})"
            else:
                # final_mpc_visual이 dict가 아닌 경우 (직접 값일 수도 있음)
                final_value = _format_metric(final_mpc_visual)
                final_mpc_info = f", MPC final_visual_loss={final_value}"
        evaluated_count = summary.get("evaluated_iterations_count", 0)
        evaluated_info = f", evaluated_iterations={evaluated_count}" if evaluated_count > 0 else ""
        print(
            f"Task {summary['task_id']:02d} | overrides={summary['overrides']} | "
            f"steps={summary['planning_steps']} | duration={summary['duration_seconds']:.2f}s"
            f"{lora_info}{ft_info}{final_mpc_info}{evaluated_info}"
        )

    if failures:
        print("\n⚠️  Failures / Timeouts:")
        for item in failures:
            print(f" - Task {item.get('task_id', 'unknown')}: {item.get('status')}, {item.get('error', 'no details')}")

    if forward_transfer_history:
        print("\n🚀 Forward Transfer Summary:")
        for entry in forward_transfer_history:
            task_id = entry.get("task_id")
            metrics = entry.get("metrics", {})
            print(f"  Task {task_id}:")
            print(
                "    "
                f"total_loss={_format_metric(metrics.get('total_loss'))} "
                f"visual_loss={_format_metric(metrics.get('visual_loss'))} "
                f"proprio_loss={_format_metric(metrics.get('proprio_loss'))}"
            )
            if "chamfer_distance" in metrics:
                print(f"    chamfer_distance={_format_metric(metrics.get('chamfer_distance'))}")
            if "iteration" in metrics:
                print(f"    iteration={metrics.get('iteration')}")

    if ensemble_evaluation_history:
        print("\n🤝 Ensemble Evaluation Summary:")
        for entry in ensemble_evaluation_history:
            task_id = entry.get("task_id")
            status = entry.get("status", "unknown")
            print(f"  Task {task_id}: status={status}")
            if status == "evaluated":
                members = entry.get("members", [])
                for member in members:
                    member_task = member.get("task_id")
                    member_loss = member.get("loss")
                    member_visual = member.get("visual_loss")
                    member_proprio = member.get("proprio_loss")
                    print(
                        f"    Member Task {member_task}: loss={_format_metric(member_loss)} "
                        f"visual_loss={_format_metric(member_visual)} proprio_loss={_format_metric(member_proprio)}"
                    )
                best_member = entry.get("best_member", {})
                best_perf = best_member.get("performance", {}) if isinstance(best_member, dict) else {}
                if best_member:
                    print(
                        f"    Best Member Task {best_member.get('task_id')}: "
                        f"loss={_format_metric(best_perf.get('loss'))} "
                        f"(threshold={entry.get('threshold')})"
                    )
                print(
                    "    "
                    f"stacking_triggered={entry.get('stacking_triggered')} "
                    f"stacking_applied={entry.get('stacking_applied')} "
                    f"reason={entry.get('stacking_reason')}"
                )
            else:
                reason = entry.get("reason")
                if reason:
                    print(f"    reason={reason}")

    if forgetting_history:
        print("\n🧠 Catastrophic Forgetting Summary:")
        for entry in forgetting_history:
            source_task = entry.get("source_task")
            print(f"  After Task {source_task}:")
            for result in entry.get("results", []):
                target_task = result.get("task_id")
                status = result.get("status", "success")
                mode_label = result.get("mode", "current_model")
                metrics = result.get("evaluation", {}).get("metrics", {}) if status == "success" else {}
                loaded_file = result.get("loaded_file", "N/A")
                if status == "success":
                    # 🔧 각 태스크의 final visual loss 찾기
                    target_final_visual_loss = None
                    for summary in task_summaries:
                        if summary.get("task_id") == target_task:
                            final_mpc_visual = summary.get("final_mpc_visual_loss")
                            if final_mpc_visual is not None:
                                if isinstance(final_mpc_visual, dict):
                                    target_final_visual_loss = final_mpc_visual.get("value")
                                else:
                                    target_final_visual_loss = final_mpc_visual
                            break
                    
                    # 🔧 오차 계산 및 표시
                    forgetting_visual_loss = metrics.get('visual_loss')
                    error_info = ""
                    if forgetting_visual_loss is not None and target_final_visual_loss is not None:
                        error = forgetting_visual_loss - target_final_visual_loss
                        error_percent = (error / target_final_visual_loss * 100) if target_final_visual_loss != 0 else 0
                        error_info = f" (error: {_format_metric(error)}, {_format_metric(error_percent)}%)"
                    
                    print(
                        f"    Task {target_task} ({mode_label}): total_loss={_format_metric(metrics.get('total_loss'))} visual_loss={_format_metric(metrics.get('visual_loss'))}{error_info} [loaded: {loaded_file}]"
                    )
                else:
                    print(
                        f"    Task {target_task} ({mode_label}): status={status} error={result.get('error')} [loaded: {loaded_file}]"
                    )

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