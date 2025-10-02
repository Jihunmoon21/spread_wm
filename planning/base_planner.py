import json
import numpy as np
from abc import ABC, abstractmethod


class BasePlanner(ABC):
    def __init__(
        self,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        log_filename,
        **kwargs,
    ):
        self.wm = wm
        self.action_dim = action_dim
        self.objective_fn = objective_fn
        self.preprocessor = preprocessor
        self.device = next(wm.parameters()).device

        self.evaluator = evaluator
        self.wandb_run = wandb_run
        self.log_filename = log_filename  # do not log if None

    def dump_logs(self, logs):
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # ndarray를 리스트로 변환
            elif isinstance(obj, (np.float32, np.int32, np.int64, np.float64)):
                return obj.item()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            else:
                return obj
        
        logs_entry = {
            key: convert_to_json_serializable(value)
            for key, value in logs.items()
        }
        
        if self.log_filename is not None:
            try:
                with open(self.log_filename, "a") as file:
                    file.write(json.dumps(logs_entry) + "\n")
            except (TypeError, ValueError) as e:
                print(f"[WARNING] Failed to log to file: {e}")

    @abstractmethod
    def plan(self):
        pass
