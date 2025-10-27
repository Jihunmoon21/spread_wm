import os
import time
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
import torch.nn as nn # nn import 추가
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from metrics.image_metrics import eval_images
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
import custom_resolvers

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        # --- accelerate 사용 시 수동 DDP 초기화 불필요 ---
        # if HydraConfig.get().mode == RunMode.MULTIRUN:
        #     log.info(" Multirun setup begin...")
        #     log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
        #     log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
        #     # ==== init ddp process group ====
        #     os.environ["RANK"] = os.environ["SLURM_PROCID"]
        #     os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        #     os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        #     try:
        #         dist.init_process_group(
        #             backend="nccl",
        #             init_method="env://",
        #             timeout=timedelta(minutes=5),  # Set a 5-minute timeout
        #         )
        #         log.info("Multirun setup completed.")
        #     except Exception as e:
        #         log.error(f"DDP setup failed: {e}")
        #         raise
        #     torch.distributed.barrier()
        #     # # ==== /init ddp process group ====
        # --- 여기까지 주석 처리 ---

        self.accelerator = Accelerator(log_with="wandb")
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg.get("wandb_run_id", None) # Use .get for safety
                if wandb_run_id:
                    log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project="dino_wm_debug",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                self.wandb_run = wandb.init(
                    project="dino_wm",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            # Save hydra config only if main process finishes initialization
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed + self.accelerator.process_index) # Add process index for different seeds per process
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False, # already shuffled in TrajSlicerDataset
                num_workers=self.cfg.env.num_workers,
                pin_memory=True, # Recommended for GPU training
                collate_fn=None, # Use default collate_fn
            )
            for x in ["train", "valid"]
        }

        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        # --- 모델 초기화 전에 변수 설정 ---
        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        # 아래 train_ flags 는 init_models/init_optimizers 에서 self.model.module 을 통해 접근하도록 변경됨
        # self.train_encoder = self.cfg.model.train_encoder
        # self.train_predictor = self.cfg.model.train_predictor
        # self.train_decoder = self.cfg.model.train_decoder
        log.info(f"Configured Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}")

        # --- 모델, 옵티마이저 초기화 ---
        self.init_models()
        self.init_optimizers()

        # --- 데이터 로더 준비 (모델/옵티마이저 준비 후) ---
        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.epoch_log = OrderedDict()

        # --- Accelerator 상태 로드 (prepare 이후) ---
        model_ckpt_dir = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest"
        if model_ckpt_dir.exists():
            log.info(f"Resuming accelerator state from {model_ckpt_dir}")
            self.accelerator.load_state(str(model_ckpt_dir))
            # Epoch 정보는 별도로 로드해야 할 수 있음 (accelerate는 모델/옵티마이저 상태 위주 관리)
            epoch_ckpt_path = model_ckpt_dir / "custom_epoch.pth"
            if epoch_ckpt_path.exists():
                 epoch_state = torch.load(epoch_ckpt_path, map_location='cpu')
                 self.epoch = epoch_state.get('epoch', 0)
                 log.info(f"Resuming from epoch {self.epoch}")
        else:
            log.info("Starting training from scratch.")


        # Keys to save 는 accelerate save_state 와는 별개, 필요시 직접 저장 로직 유지 가능하나 복잡
        # self._keys_to_save = [
        #     "epoch",
        # ]
        # self._keys_to_save += (
        #     ["encoder", "encoder_optimizer"] if self.cfg.model.train_encoder else [] # cfg 값 사용
        # )
        # self._keys_to_save += (
        #     ["predictor", "predictor_optimizer"]
        #     if self.cfg.model.train_predictor and self.cfg.has_predictor # cfg 값 사용
        #     else []
        # )
        # self._keys_to_save += (
        #     ["decoder", "decoder_optimizer"] if self.cfg.model.train_decoder else [] # cfg 값 사용
        # )
        # self._keys_to_save += ["action_encoder", "proprio_encoder"]


    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        # 저장 경로 설정
        save_dir_latest = Path("checkpoints") / "model_latest"
        save_dir_epoch = Path("checkpoints") / f"model_{self.epoch}"

        # Accelerator 상태 저장 (모델, 옵티마이저 등)
        self.accelerator.save_state(str(save_dir_latest))
        self.accelerator.save_state(str(save_dir_epoch))

        # Epoch 정보 별도 저장 (메인 프로세스에서만)
        ckpt_path = None
        if self.accelerator.is_main_process:
            epoch_state = {'epoch': self.epoch}
            torch.save(epoch_state, save_dir_latest / "custom_epoch.pth")
            torch.save(epoch_state, save_dir_epoch / "custom_epoch.pth")
            log.info("Saved model and accelerator state to {}".format(os.getcwd()))
            ckpt_path = str(save_dir_epoch) # 평가 스크립트를 위해 epoch 경로 반환
        else:
            ckpt_path = None # 다른 프로세스는 경로 없음

        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    # --- 기존 load_ckpt 함수는 accelerator.load_state 로 대체되므로 주석 처리 또는 삭제 ---
    # def load_ckpt(self, filename="model_latest.pth"):
    #     # checkpoint 로드는 map_location='cpu' 또는 accelerator.load_state 사용 권장
    #     ckpt = torch.load(filename, map_location='cpu') # CPU로 로드
    #     loaded_keys = set()
    #     for k, v in ckpt.items():
    #         if k in self._keys_to_save: # 저장하기로 한 키만 로드 시도
    #             target_attr = self.__dict__.get(k, None)
    #             if isinstance(target_attr, torch.nn.Module):
    #                 # Unwrap model if needed before loading state dict
    #                 unwrapped_model = self.accelerator.unwrap_model(target_attr) if hasattr(target_attr, 'module') else target_attr
    #                 try:
    #                     # Check if v is state_dict or the model itself
    #                     if isinstance(v, dict): # state_dict case
    #                         unwrapped_model.load_state_dict(v)
    #                     elif isinstance(v, torch.nn.Module): # model object case (less common)
    #                         unwrapped_model.load_state_dict(v.state_dict())
    #                     loaded_keys.add(k)
    #                 except Exception as e:
    #                     log.warning(f"Failed to load state dict for {k}: {e}")
    #             elif isinstance(target_attr, torch.optim.Optimizer):
    #                 # Optimizer state dict loading might need adjustment with accelerate
    #                 try:
    #                     if isinstance(v, dict):
    #                         target_attr.load_state_dict(v)
    #                         loaded_keys.add(k)
    #                 except Exception as e:
    #                      log.warning(f"Failed to load optimizer state for {k}: {e}")
    #             elif k == 'epoch':
    #                  self.__dict__[k] = v
    #                  loaded_keys.add(k)
    #             # else: handle other types if needed

    #     not_in_ckpt = set(self._keys_to_save) - loaded_keys
    #     if len(not_in_ckpt):
    #         log.warning("Keys not found or failed to load from ckpt: %s", not_in_ckpt)

    def init_models(self):
        # --- 체크포인트 로드는 init 마지막으로 이동 ---
        # model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        # if model_ckpt.exists():
        #     self.load_ckpt(model_ckpt)
        #     log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")

        # initialize encoder
        encoder = hydra.utils.instantiate( # 로컬 변수로 변경
            self.cfg.encoder,
        )
        if not self.cfg.model.train_encoder: # cfg 값 사용
            for param in encoder.parameters():
                param.requires_grad = False

        proprio_encoder = hydra.utils.instantiate( # 로컬 변수로 변경
            self.cfg.proprio_encoder,
            in_chans=self.datasets["train"].dataset.proprio_dim, # self.datasets 는 prepare 전 접근 가능
            emb_dim=self.cfg.proprio_emb_dim,
        )
        proprio_emb_dim = proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(proprio_encoder)}")

        action_encoder = hydra.utils.instantiate( # 로컬 변수로 변경
            self.cfg.action_encoder,
            in_chans=self.datasets["train"].dataset.action_dim, # self.datasets 는 prepare 전 접근 가능
            emb_dim=self.cfg.action_emb_dim,
        )
        action_emb_dim = action_encoder.emb_dim
        print(f"Action encoder type: {type(action_encoder)}")

        # initialize predictor
        if encoder.latent_ndim == 1:
            num_patches = 1
        else:
            decoder_scale = 16
            num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        predictor = None # 로컬 변수로 변경
        if self.cfg.has_predictor:
            predictor = hydra.utils.instantiate(
                self.cfg.predictor,
                num_patches=num_patches,
                num_frames=self.cfg.num_hist,
                dim=encoder.emb_dim # encoder 로컬 변수 사용
                + (
                    proprio_emb_dim * self.cfg.num_proprio_repeat
                    + action_emb_dim * self.cfg.num_action_repeat
                )
                * (self.cfg.concat_dim),
            )
            if not self.cfg.model.train_predictor: # cfg 값 사용
                for param in predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        decoder = None # 로컬 변수로 변경
        if self.cfg.has_decoder:
            if self.cfg.env.decoder_path is not None:
                decoder_path = os.path.join(
                    self.base_path, self.cfg.env.decoder_path
                )
                ckpt = torch.load(decoder_path, map_location='cpu')
                # 가중치 로드 로직은 동일
                if isinstance(ckpt, dict):
                    if 'decoder' in ckpt:
                        decoder = ckpt['decoder']
                    elif 'state_dict' in ckpt:
                         decoder = hydra.utils.instantiate(
                             self.cfg.decoder,
                             emb_dim=encoder.emb_dim # encoder 로컬 변수 사용
                         )
                         decoder_state_dict = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('decoder.')}
                         decoder.load_state_dict(decoder_state_dict, strict=False)
                    else:
                         decoder = ckpt # Assume it's the model object
                else:
                    decoder = ckpt
                log.info(f"Loaded decoder structure/weights from {decoder_path}")
            else:
                decoder = hydra.utils.instantiate(
                    self.cfg.decoder,
                    emb_dim=encoder.emb_dim, # encoder 로컬 변수 사용
                )
            if not self.cfg.model.train_decoder: # cfg 값 사용
                for param in decoder.parameters():
                    param.requires_grad = False

        # --- 주 모델 인스턴스화 (로컬 변수 사용) ---
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=encoder,
            proprio_encoder=proprio_encoder,
            action_encoder=action_encoder,
            predictor=predictor,
            decoder=decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
        )

        # --- 주 모델을 accelerator로 준비 ---
        # prepare 호출 전에 self.model 에 할당해야 함
        self.model = self.accelerator.prepare(self.model)

        # 모델 컴포넌트들을 self 에 저장 (옵티마이저 생성 시 필요)
        # prepare 후에는 self.model.module 또는 self.model 에서 접근 가능
        # 편의상 unwrap 된 모델을 저장해 둘 수 있으나, 상태 관리 복잡해질 수 있음
        # 옵티마이저 생성 시 model_to_optim 에서 접근하므로 별도 저장 불필요


    def init_optimizers(self):
        # 모델 파라미터 접근 시 self.model.module 사용
        inner_model = self.model.module if hasattr(self.model, 'module') else self.model

        if self.cfg.model.train_encoder: # cfg 값 사용
            self.encoder_optimizer = torch.optim.Adam(
                inner_model.encoder.parameters(),
                # lr=self.cfg.training.encoder_lr,
            )
            # 옵티마이저 준비는 나중에 일괄 처리
        else:
            self.encoder_optimizer = None # 명시적으로 None 할당

        if self.cfg.has_predictor:
            if self.cfg.model.train_predictor: # cfg 값 사용
                predictor_params = filter(lambda p: p.requires_grad, inner_model.predictor.parameters())
                self.predictor_optimizer = torch.optim.AdamW(
                    predictor_params,
                    lr=self.cfg.training.predictor_lr,
                )
            else:
                self.predictor_optimizer = None # 명시적으로 None 할당

            # Action/Proprio Encoder 는 항상 학습 가능하다고 가정 (requires_grad=True)
            action_proprio_params = itertools.chain(
                inner_model.action_encoder.parameters(),
                inner_model.proprio_encoder.parameters()
            )
            self.action_encoder_optimizer = torch.optim.AdamW(
                 action_proprio_params,
                lr=self.cfg.training.action_encoder_lr,
            )
        else:
             self.predictor_optimizer = None
             self.action_encoder_optimizer = None

        if self.cfg.has_decoder and self.cfg.model.train_decoder: # cfg 값 사용
            decoder_params = inner_model.decoder.parameters()
            self.decoder_optimizer = torch.optim.Adam(
                decoder_params, lr=self.cfg.training.decoder_lr
            )
        else:
             self.decoder_optimizer = None # 명시적으로 None 할당

        # --- 모든 옵티마이저를 한 번에 prepare ---
        (
            self.encoder_optimizer,
            self.predictor_optimizer,
            self.action_encoder_optimizer,
            self.decoder_optimizer
        ) = self.accelerator.prepare(
            self.encoder_optimizer,
            self.predictor_optimizer,
            self.action_encoder_optimizer,
            self.decoder_optimizer
        )
        # prepare는 None 객체도 처리 가능


    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        # --- Wandb 로깅은 main process 에서만 ---
        if not self.accelerator.is_main_process:
            return

        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    try: # job.result() can raise exceptions
                        result = job.result()
                        print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                        log_data = {
                            f"{job_name}/{key}": value for key, value in result.items()
                        }
                        log_data["epoch"] = epoch
                        self.wandb_run.log(log_data)
                    except Exception as e:
                         log.error(f"Error getting result for job {job_name} at epoch {epoch}: {e}")
                    finally:
                        self.job_set.remove((epoch, job_name, job)) # Ensure job is removed
            time.sleep(5) # Check less frequently


    def run(self):
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        init_epoch = self.epoch + 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            # Validation can often be done on main process only if needed and feasible
            # This requires adjusting how metrics are gathered/logged
            self.val() # Currently runs on all processes
            self.logs_flash(step=self.epoch)

            # --- Checkpoint saving ---
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt() # save_ckpt already handles main process logic

                # Launch planning jobs on main process
                if self.accelerator.is_main_process and self.cfg.plan_settings.plan_cfg_path is not None and ckpt_path is not None:
                    from plan import build_plan_cfg_dicts, launch_plan_jobs
                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.cfg.plan_settings.plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path, # Ensure this path is accessible
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.cfg.plan_settings.planner,
                        goal_source=self.cfg.plan_settings.goal_source,
                        goal_H=self.cfg.plan_settings.goal_H,
                        alpha=self.cfg.plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)

        # Ensure all planning jobs are finished before exiting (on main process)
        if self.accelerator.is_main_process:
             log.info("Waiting for remaining planning jobs to finish...")
             while len(self.job_set) > 0:
                  time.sleep(10) # Wait and let monitor thread handle logging
             executor.shutdown()
             log.info("All planning jobs finished.")
        self.accelerator.wait_for_everyone() # Final barrier


    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        # .module 추가
        inner_model = self.model.module if hasattr(self.model, 'module') else self.model
        for k in z_pred.keys():
            # .module 추가
            loss = inner_model.emb_criterion(z_pred[k], z_tgt[k])
            # Ensure loss is detached before adding to logs if it requires grad
            logs[k] = loss.detach() if loss.requires_grad else loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, t, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, t, n_patches, emb_dim), doesn't include action dims
        state:  (b, t, dim)
        """
        logs = {}
        # .module 추가
        inner_model = self.model.module if hasattr(self.model, 'module') else self.model
        slices = {
            "full": (None, None),
            # .module 추가
            "pred": (-inner_model.num_pred, None),
            # .module 추가
            "next1": (-inner_model.num_pred, -inner_model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            # Pass inner_model to err_eval_single if needed, or rely on self.model access inside it
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        # 모델을 train 모드로 설정 (accelerate 가 DDP 내부 모델도 처리)
        self.model.train()
        inner_model = self.model.module if hasattr(self.model, 'module') else self.model

        # tqdm 은 main process 에서만 사용
        iterable_dataloader = tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train", disable=not self.accelerator.is_main_process)

        for i, data in enumerate(iterable_dataloader):
            # 데이터는 이미 prepare 된 dataloader 에서 오므로 자동으로 해당 device 로 이동됨
            obs, act, state, _ = data
            plot = i == 0

            # --- Optimizer Zero Grad ---
            if inner_model.train_encoder and self.encoder_optimizer:
                self.encoder_optimizer.zero_grad()
            if self.cfg.has_decoder and inner_model.train_decoder and self.decoder_optimizer:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor and inner_model.train_predictor:
                if self.predictor_optimizer:
                     self.predictor_optimizer.zero_grad()
                if self.action_encoder_optimizer:
                     self.action_encoder_optimizer.zero_grad()

            # --- Forward Pass ---
            # self.model(...) 은 DDP 래퍼를 통해 자동으로 forward 호출
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            # --- Backward Pass ---
            self.accelerator.backward(loss)

            # --- Optimizer Step ---
            # Clip gradients if configured (add self.accelerator.clip_grad_norm_ or clip_grad_value_)
            # Example:
            # if self.cfg.training.max_grad_norm is not None:
            #     self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)

            if inner_model.train_encoder and self.encoder_optimizer:
                self.encoder_optimizer.step()
            if self.cfg.has_decoder and inner_model.train_decoder and self.decoder_optimizer:
                self.decoder_optimizer.step()
            if self.cfg.has_predictor and inner_model.train_predictor:
                if self.predictor_optimizer:
                    self.predictor_optimizer.step()
                if self.action_encoder_optimizer:
                    self.action_encoder_optimizer.step()

            # --- Metrics and Logging (Gather across processes) ---
            # loss 는 이미 스칼라이므로 gather 후 mean
            avg_loss = self.accelerator.gather(loss).mean().item()

            # loss_components 는 dict 이므로 각 value 를 gather 후 mean
            gathered_loss_components = self.accelerator.gather_for_metrics(loss_components)
            avg_loss_components = {
                f"train_{key}": value.mean().item() for key, value in gathered_loss_components.items()
            }
            avg_loss_components['train_loss'] = avg_loss # Include main loss

            # 이미지 관련 메트릭 및 플로팅 (main process 에서만 수행하거나, gather 필요)
            if self.cfg.has_decoder and plot:
                if self.cfg.has_predictor:
                    # separate_emb, encode_obs 는 inner_model 에서 호출
                    z_obs_out, z_act_out = inner_model.separate_emb(z_out)
                    z_gt = inner_model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=inner_model.num_pred)

                    # err_eval 호출
                    err_logs = self.err_eval(z_obs_out, z_tgt) # err_eval 내부에서 .module 처리

                    # Gather err_logs
                    gathered_err_logs = self.accelerator.gather_for_metrics(err_logs)
                    avg_err_logs = {
                        f"train_{key}": value.mean().item() for key, value in gathered_err_logs.items()
                    }
                    self.logs_update(avg_err_logs) # logs_update 는 내부적으로 list 변환하므로 dict 전달

                # Image eval metrics gathering
                if visual_out is not None:
                    # Loop and calculate metrics on each process, then gather
                    all_img_pred_scores = {}
                    for t in range(self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred):
                        # Ensure obs['visual'][:, t] exists on the current device
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t].to(self.device)
                        )
                        # Gather results across all processes
                        gathered_scores = self.accelerator.gather_for_metrics(img_pred_scores)
                        for k, v in gathered_scores.items():
                             key_name = f"train_img_{k}_pred_t{t}" # Add timestep info
                             if key_name not in all_img_pred_scores:
                                 all_img_pred_scores[key_name] = []
                             all_img_pred_scores[key_name].append(v.mean().item()) # Append mean from this batch
                    # Average over batches/timesteps after loop if needed, or log per timestep
                    self.logs_update(all_img_pred_scores)


                if visual_reconstructed is not None:
                     all_img_recon_scores = {}
                     for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t].to(self.device)
                        )
                        gathered_scores = self.accelerator.gather_for_metrics(img_reconstruction_scores)
                        for k, v in gathered_scores.items():
                            key_name = f"train_img_{k}_reconstructed_t{t}" # Add timestep info
                            if key_name not in all_img_recon_scores:
                                all_img_recon_scores[key_name] = []
                            all_img_recon_scores[key_name].append(v.mean().item())
                     self.logs_update(all_img_recon_scores)

                # Plotting should only happen on the main process
                if self.accelerator.is_main_process:
                    self.plot_samples(
                        obs["visual"], # Pass tensors directly
                        visual_out,
                        visual_reconstructed,
                        self.epoch,
                        batch=i,
                        num_samples=self.num_reconstruct_samples,
                        phase="train",
                    )

            # Update logs with main loss components
            self.logs_update(avg_loss_components) # logs_update 는 내부적으로 list 변환하므로 dict 전달

    def val(self):
        self.model.eval()
        inner_model = self.model.module if hasattr(self.model, 'module') else self.model

        # --- Openloop Rollout (수정 필요) ---
        # openloop_rollout 함수는 현재 DDP 환경에서 바로 작동하기 어려움 (데이터셋 인덱싱, gather 등)
        # 1. 각 프로세스가 데이터셋의 일부만 처리하도록 수정 (예: DistributedSampler 사용 또는 수동 분할)
        # 2. rollout 결과를 모든 프로세스에서 gather 하여 main process 에서 집계/로깅
        # 임시로 주석 처리 또는 main process 에서만 실행하도록 변경 필요
        # if len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
        #     with torch.no_grad():
        #         if self.accelerator.is_main_process: # 예시: 메인 프로세스에서만 실행
        #             train_rollout_logs = self.openloop_rollout(
        #                 self.train_traj_dset, mode="train"
        #             )
        #             train_rollout_logs = {
        #                 f"train_{k}": [v] for k, v in train_rollout_logs.items()
        #             }
        #             self.logs_update(train_rollout_logs)

        #             val_rollout_logs = self.openloop_rollout(self.val_traj_dset, mode="val")
        #             val_rollout_logs = {
        #                 f"val_{k}": [v] for k, v in val_rollout_logs.items()
        #             }
        #             self.logs_update(val_rollout_logs)
        # self.accelerator.wait_for_everyone() # 메인 프로세스만 실행 시 배리어 필요

        # --- Validation Loop ---
        iterable_dataloader = tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid", disable=not self.accelerator.is_main_process)
        for i, data in enumerate(iterable_dataloader):
            obs, act, state, _ = data
            plot = i == 0

            with torch.no_grad(): # Validation은 no_grad 컨텍스트 사용
                z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                    obs, act
                )

            # --- Metrics and Logging (Gather) ---
            avg_loss = self.accelerator.gather(loss).mean().item()
            gathered_loss_components = self.accelerator.gather_for_metrics(loss_components)
            avg_loss_components = {
                f"val_{key}": value.mean().item() for key, value in gathered_loss_components.items()
            }
            avg_loss_components['val_loss'] = avg_loss

            # 이미지 관련 메트릭 및 플로팅
            if self.cfg.has_decoder and plot:
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = inner_model.separate_emb(z_out)
                    z_gt = inner_model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=inner_model.num_pred)

                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    gathered_err_logs = self.accelerator.gather_for_metrics(err_logs)
                    avg_err_logs = {
                        f"val_{key}": value.mean().item() for key, value in gathered_err_logs.items()
                    }
                    self.logs_update(avg_err_logs)

                if visual_out is not None:
                    all_img_pred_scores = {}
                    for t in range(self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t].to(self.device)
                        )
                        gathered_scores = self.accelerator.gather_for_metrics(img_pred_scores)
                        for k, v in gathered_scores.items():
                             key_name = f"val_img_{k}_pred_t{t}"
                             if key_name not in all_img_pred_scores:
                                 all_img_pred_scores[key_name] = []
                             all_img_pred_scores[key_name].append(v.mean().item())
                    self.logs_update(all_img_pred_scores)


                if visual_reconstructed is not None:
                    all_img_recon_scores = {}
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t].to(self.device)
                        )
                        gathered_scores = self.accelerator.gather_for_metrics(img_reconstruction_scores)
                        for k, v in gathered_scores.items():
                            key_name = f"val_img_{k}_reconstructed_t{t}"
                            if key_name not in all_img_recon_scores:
                                all_img_recon_scores[key_name] = []
                            all_img_recon_scores[key_name].append(v.mean().item())
                    self.logs_update(all_img_recon_scores)

                if self.accelerator.is_main_process:
                    self.plot_samples(
                        obs["visual"],
                        visual_out,
                        visual_reconstructed,
                        self.epoch,
                        batch=i,
                        num_samples=self.num_reconstruct_samples,
                        phase="valid",
                    )

            self.logs_update(avg_loss_components)

    # --- openloop_rollout 함수 수정 필요 ---
    # DDP 환경에서는 각 프로세스가 전체 데이터셋의 일부만 보거나,
    # 각 프로세스가 전체 데이터셋을 보더라도 rollout 결과를 gather 해야 함.
    # 여기서는 함수 내부에서 self.model.module 사용하도록 수정
    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        # Seed setting needs care in multiprocessing. Setting based on rank is one way.
        np.random.seed(self.cfg.training.seed + self.accelerator.process_index)
        torch.manual_seed(self.cfg.training.seed + self.accelerator.process_index)

        inner_model = self.model.module if hasattr(self.model, 'module') else self.model
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = Path(f"rollout_plots/e{self.epoch}_rollout")

        # Create directory only on main process
        if self.accelerator.is_main_process:
            plotting_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone() # Ensure directory exists before proceeding

        logs = {}

        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # Each process might need to sample different trajectories or gather results
        # Simple approach: Each process runs rollout on its sampled trajectories
        # More complex: Use DistributedSampler for dset or manually split indices

        # --- 이 부분은 DDP 환경에서 데이터셋 인덱싱 주의 필요 ---
        # 현재 구현은 모든 프로세스가 동일한 traj_idx 를 샘플링할 수 있음
        # 더 나은 방법: 1) dset 자체를 prepare 하거나, 2) 인덱스를 나눠서 처리
        indices_to_rollout = np.random.choice(len(dset), num_rollout, replace=False)

        # for idx in range(num_rollout): # Instead, iterate through assigned indices
        for i, traj_idx in enumerate(indices_to_rollout):
            # traj_idx = np.random.randint(0, len(dset)) # Use pre-sampled index

            # --- 데이터셋 접근 ---
            # dset[traj_idx] 접근 방식은 DDP에서 비효율적일 수 있음 (메모리 중복)
            # TrajSlicerDataset 이 메모리 효율적 로더를 사용하면 괜찮을 수 있음
            obs, act, state, _ = dset[traj_idx] # Assume dset is accessible

            # Move data to current device
            act = act.to(self.device)
            # obs needs device transfer later

            # Determine start and horizon (logic seems ok)
            if rand_start_end:
                 if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                     start = np.random.randint(0, obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,)
                 else: start = 0
                 max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                 if max_horizon <= min_horizon: # Ensure horizon is valid
                     horizon = min_horizon # Or skip this trajectory
                 else:
                     horizon = np.random.randint(min_horizon, max_horizon + 1)
            else:
                 start = 0
                 horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip
                 if horizon < min_horizon: continue # Skip if too short


            # Slice data
            obs_sliced = {}
            for k in obs.keys():
                obs_sliced[k] = obs[k][start : start + horizon * self.cfg.frameskip + 1 : self.cfg.frameskip]
            act_sliced = act[start : start + horizon * self.cfg.frameskip]
            act_sliced = rearrange(act_sliced, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            # --- Goal Encoding ---
            obs_g = {}
            for k in obs_sliced.keys():
                obs_g[k] = obs_sliced[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            # Use inner_model for encoding
            with torch.no_grad(): # Ensure no gradients calculated here
                z_g = inner_model.encode_obs(obs_g) # .module

            actions = act_sliced.unsqueeze(0) # Already on device

            for n_past, postfix in num_past:
                obs_0 = {}
                for k in obs_sliced.keys():
                    obs_0[k] = obs_sliced[k][:n_past].unsqueeze(0).to(self.device)

                # --- Rollout using inner_model ---
                with torch.no_grad():
                    # .module
                    z_obses, z = inner_model.rollout(obs_0, actions)

                # --- Error Calculation ---
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                # Use self.err_eval_single which handles .module internally
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k, v in div_loss.items():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key not in logs: logs[log_key] = []
                    logs[log_key].append(v) # Append tensor

                # --- Decoding and Plotting (Main process only) ---
                if self.cfg.has_decoder and self.accelerator.is_main_process:
                    with torch.no_grad():
                         # .module
                        visuals = inner_model.decode_obs(z_obses)[0]["visual"]
                    # Ground truth obs also needs to be available for plotting
                    gt_visuals_for_plot = obs_sliced["visual"].cpu() # Get relevant GT frames
                    # Ensure visuals are on CPU for concatenation and saving
                    imgs = torch.cat([gt_visuals_for_plot, visuals[0].cpu()], dim=0)

                    plot_filename = plotting_dir / f"e{self.epoch}_{mode}_{i}{postfix}.png" # Use Path object
                    self.plot_imgs(imgs, gt_visuals_for_plot.shape[0], str(plot_filename)) # Pass str path

        # --- Aggregate logs across processes ---
        # Gather the lists of tensors for each log key
        gathered_logs = {}
        for key, tensor_list in logs.items():
             if tensor_list: # Only gather if list is not empty
                 # Concatenate tensors in the list before gathering
                 gathered_tensors = self.accelerator.gather(torch.stack(tensor_list))
                 # Calculate mean on the gathered tensors (across all processes and rollouts)
                 gathered_logs[key] = gathered_tensors.mean().item()
             else:
                 gathered_logs[key] = 0.0 # Or some placeholder like float('nan')

        # Return aggregated logs (dict of scalars)
        return gathered_logs


    def logs_update(self, logs):
        # This function needs adjustment for DDP logging
        # Option 1: Log directly if accelerator handles it (e.g., self.accelerator.log)
        # Option 2: Store locally and average in logs_flash (current approach, needs care with gather)

        # Current approach assumes logs are dicts of scalars or lists of scalars
        for key, value in logs.items():
            if isinstance(value, torch.Tensor): # Should not happen if gathered correctly
                current_val = value.detach().cpu().item()
                count, total = self.epoch_log.get(key, (0, 0.0))
                self.epoch_log[key] = (count + 1, total + current_val)
            elif isinstance(value, (list, tuple)): # Handles lists from image metrics etc.
                 length = len(value)
                 total_sum = sum(v.item() if isinstance(v, torch.Tensor) else v for v in value)
                 count, total = self.epoch_log.get(key, (0, 0.0))
                 self.epoch_log[key] = (count + length, total + total_sum)
            elif isinstance(value, (int, float)): # Handles single scalar values
                 count, total = self.epoch_log.get(key, (0, 0.0))
                 self.epoch_log[key] = (count + 1, total + value)


    def logs_flash(self, step):
        epoch_log = OrderedDict()
        # --- Averaging logs across processes ---
        # This needs to happen *before* logging to Wandb
        # Average the accumulated sums and counts
        for key, (count, total) in self.epoch_log.items():
             # Create tensors for count and total sum
             count_tensor = torch.tensor(count, device=self.device, dtype=torch.long)
             total_tensor = torch.tensor(total, device=self.device, dtype=torch.float)

             # Sum counts and totals across all processes
             dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
             dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

             # Calculate global average (avoid division by zero)
             global_count = count_tensor.item()
             global_total = total_tensor.item()
             if global_count > 0:
                 epoch_log[key] = global_total / global_count
             else:
                 epoch_log[key] = 0.0 # Or float('nan')

        epoch_log["epoch"] = step

        # --- Log on main process ---
        if self.accelerator.is_main_process:
            # Safely get train/val loss for logging message
            train_loss_log = epoch_log.get('train_loss', float('nan'))
            val_loss_log = epoch_log.get('val_loss', float('nan'))
            log.info(f"Epoch {self.epoch}  Avg Training loss: {train_loss_log:.4f}  \
                    Avg Validation loss: {val_loss_log:.4f}")
            if self.wandb_run: # Check if wandb run exists
                self.wandb_run.log(epoch_log)

        # Clear logs for next epoch on all processes
        self.epoch_log = OrderedDict()


    def plot_samples(
        self,
        gt_imgs_tensor, # Renamed to avoid confusion
        pred_imgs_tensor,
        reconstructed_gt_imgs_tensor,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        Plots samples, should only be called by the main process.
        Inputs are tensors assumed to be on the correct device or CPU.
        """
        if not self.accelerator.is_main_process:
            return # Only main process handles plotting

        # .module 추가
        inner_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Ensure tensors are on CPU for sampling and manipulation
        gt_imgs = gt_imgs_tensor.detach().cpu()
        pred_imgs = pred_imgs_tensor.detach().cpu() if pred_imgs_tensor is not None else None
        reconstructed_gt_imgs = reconstructed_gt_imgs_tensor.detach().cpu() if reconstructed_gt_imgs_tensor is not None else None

        num_frames = gt_imgs.shape[1]
        num_available_samples = gt_imgs.shape[0]
        actual_num_samples = min(num_samples, num_available_samples)

        if actual_num_samples == 0: return # Nothing to plot

        # Sample tensors
        indices = list(range(actual_num_samples))
        gt_imgs = gt_imgs[indices]
        if pred_imgs is not None: pred_imgs = pred_imgs[indices]
        if reconstructed_gt_imgs is not None: reconstructed_gt_imgs = reconstructed_gt_imgs[indices]

        # fill in blank images for frameskips
        if pred_imgs is not None:
             # Use inner_model.num_pred
            num_pred = inner_model.num_pred
            # Ensure shape matches gt_imgs time dimension if necessary
            # Current logic adds num_pred blanks at the start, check if this is correct
            blank_shape = (actual_num_samples, num_pred, *pred_imgs.shape[2:])
            blanks = torch.full(blank_shape, -1.0, dtype=pred_imgs.dtype) # Use float, same dtype
            # Check dimensions before concat
            # Expected pred_imgs shape: (actual_num_samples, num_hist, C, H, W)
            # Expected result shape: (actual_num_samples, num_hist + num_pred, C, H, W) ? -> Seems off
            # Let's assume pred_imgs contains predictions for num_pred steps following num_hist context
            # So pred_imgs shape is (actual_num_samples, num_pred, C, H, W)
            # We need to prepend blanks for the history frames
            num_hist = inner_model.num_hist
            blank_shape_hist = (actual_num_samples, num_hist, *pred_imgs.shape[2:])
            blanks_hist = torch.full(blank_shape_hist, -1.0, dtype=pred_imgs.dtype)
            pred_imgs_padded = torch.cat((blanks_hist, pred_imgs), dim=1) # Pad hist frames
        else:
            pred_imgs_padded = torch.full_like(gt_imgs, -1.0) # Match gt_imgs shape

        # Rearrange and concatenate
        # Ensure dimensions match before rearrange: gt, pred_padded, recon should all be (N, T, C, H, W)
        T_gt = gt_imgs.shape[1]
        T_pred = pred_imgs_padded.shape[1]
        T_recon = reconstructed_gt_imgs.shape[1] if reconstructed_gt_imgs is not None else T_gt

        # Make time dimensions consistent if needed, e.g., pad pred/recon if shorter than gt
        # Assuming T_gt = T_pred = T_recon = num_hist + num_pred for simplicity here
        # Adjust based on actual model output shapes if different

        pred_imgs_flat = rearrange(pred_imgs_padded, "b t c h w -> (b t) c h w")
        gt_imgs_flat = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        recon_flat = torch.full_like(gt_imgs_flat, -1.0) # Default blanks
        if reconstructed_gt_imgs is not None:
             # Ensure time dim matches gt_imgs before rearranging
             if reconstructed_gt_imgs.shape[1] == T_gt:
                 recon_flat = rearrange(reconstructed_gt_imgs, "b t c h w -> (b t) c h w")

        imgs = torch.cat([gt_imgs_flat, pred_imgs_flat, recon_flat], dim=0)

        # Directory creation already handled
        # self.accelerator.wait_for_everyone() # Not needed here as only main process runs this

        self.plot_imgs(
            imgs,
            num_columns=actual_num_samples * T_gt, # Use actual time dimension
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        # This function is called only by the main process
        try:
            img_dir = os.path.dirname(img_name)
            # 디렉토리가 존재하지 않으면 생성 (exist_ok=True는 이미 있어도 오류 안냄)
            if img_dir: # 디렉토리 경로가 비어있지 않은 경우에만 생성 시도
                os.makedirs(img_dir, exist_ok=True)
            utils.save_image(
                imgs,
                img_name,
                nrow=num_columns,
                normalize=True,
                value_range=(-1, 1), # Assuming input is in [-1, 1]
            )
        except Exception as e:
             log.error(f"Failed to save image {img_name}: {e}")


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()