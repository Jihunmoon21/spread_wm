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
        self.current_task_id = current_task_id

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
                self.online_learner = OnlineLora(workspace=self)
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
            print(f"[DEBUG] MPC planner: n_taken_actions={self.planner.n_taken_actions}, horizon={self.planner.sub_planner.horizon}")
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

        if (
            self.is_online_lora
            and hasattr(self.online_learner, "task_changed")
        ):
            task_changed = self.online_learner.task_changed
        else:
            task_changed = False

        actions, action_len = None, None

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

        if use_ensemble_selection:
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
        return logs


def _task_worker(
    task_idx: int,
    task_cfg_dict: dict,
    overrides: dict,
    base_env_overrides: dict,
    model_cfg_path: str,
    mpc_iters_per_task: int,
    result_queue,
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

        if hasattr(plan_workspace, "planner"):
            if hasattr(plan_workspace.planner, "max_iter"):
                plan_workspace.planner.max_iter = mpc_iters_per_task
            if hasattr(plan_workspace.planner, "iter"):
                plan_workspace.planner.iter = 0

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
                    print("⚠️  Task-based LoRA stacking skipped or failed.")
            except Exception as e:
                print(f"⚠️  Task-based stacking failed: {e}")
        else:
            if plan_workspace.is_online_lora:
                if hasattr(plan_workspace.online_learner, "ensemble_manager"):
                    print(
                        f"⚠️  Task-based LoRA stacking disabled "
                        f"(hybrid_enabled: {getattr(plan_workspace.online_learner, 'hybrid_enabled', False)}, "
                        f"task_based_stacking: {getattr(plan_workspace.online_learner, 'task_based_stacking', False)})"
                    )
                else:
                    print("⚠️  Standard OnlineLora mode - using default LoRA stacking behavior")

        print(
            f"--- Task {task_idx + 1} | Running MPC until iteration {mpc_iters_per_task} (max_iter) ---"
        )
        logs = plan_workspace.perform_planning()
        prefixed_logs = {f"task_{task_idx + 1}/{k}": v for k, v in logs.items()}

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

                current_task_for_save = getattr(plan_workspace.online_learner, "current_task_id", task_idx + 1)
                print(f"💾 Saving finalized LoRA member for Task {current_task_for_save} at task end...")
                plan_workspace.online_learner.save_current_lora_member(task_id=current_task_for_save, reason="task_end")
        except Exception as e:
            print(f"⚠️  Failed to save finalized LoRA member at task end: {e}")

        summary = {
            "task_id": task_idx + 1,
            "overrides": overrides,
            "planning_steps": task_planning_steps,
            "lora_stacks": task_lora_stacks,
            "stack_history": getattr(plan_workspace.online_learner, "stack_history", []) if plan_workspace.is_online_lora else [],
            "duration_seconds": time.time() - task_start,
        }

        result_queue.put({"status": "success", "logs": [prefixed_logs], "summary": summary})
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
        {"table_color": "brown"},
        {"table_color": "brown", "camera_view": 0},
        # {"camera_view": 0},
        # {"table_color": "brown", "granular_radius": 0.25},
        # {"table_color": "purple"},
        # {"table_color": "purple", "camera_view": 0},
        # {"table_color": "purple", "granular_radius": 0.25},
        # {"granular_radius": 0.25, "camera_view": 0},
    ]
    mpc_iters_per_task = 1    # 각 태스크당 목표 MPC iteration 수
    overall_logs = []
    task_summaries = []

    ctx = mp.get_context("spawn")
    timeout_seconds = cfg_dict.get("task_timeout_seconds", 300) # 자식 프로세스를 기다리는 시간
    failures = []

    model_cfg_path = os.path.join(model_path, "hydra.yaml")

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
            task_summaries.append(result.get("summary", {}))
        else:
            failures.append(result)
            print(f"⚠️  Task {task_idx + 1} failed with status {result['status']}: {result.get('error')}")

    print(f"\n{'=' * 25} Granular Continual Planning Finished {'=' * 25}")
    for summary in task_summaries:
        lora_info = f", LoRA stacks={summary['lora_stacks']}" if "lora_stacks" in summary else ""
        print(
            f"Task {summary['task_id']:02d} | overrides={summary['overrides']} | "
            f"steps={summary['planning_steps']} | duration={summary['duration_seconds']:.2f}s{lora_info}"
        )

    if failures:
        print("\n⚠️  Failures / Timeouts:")
        for item in failures:
            print(f" - Task {item.get('task_id', 'unknown')}: {item.get('status')}, {item.get('error', 'no details')}")

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