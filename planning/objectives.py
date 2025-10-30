import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # F 추가

# --- 기존 함수 (수정 없음) ---
def create_objective_fn(alpha, base, mode="last"):
    """
    Loss calculated on the last pred frame vs the target NEXT frame.
    Used during world model training or planning where the target is the next step.
    Args:
        alpha: int, weight for proprioceptive loss
        base: int. only used for objective_fn_all weighting
        mode: 'last' or 'all'
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt):
        """
        Compares the last predicted frame with the target frame(s).
        Args:
            z_obs_pred: dict, {'visual': (B, T_pred, *D_visual), 'proprio': (B, T_pred, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T_tgt, *D_visual), 'proprio': (B, T_tgt, *D_proprio)} T_tgt is usually 1 for planning goal.
        Returns:
            loss: tensor (B, )
        """
        # Compare last prediction with the single target frame
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim)) # Avg over T, P, D
        )
        loss_proprio = 0
        if 'proprio' in z_obs_tgt and z_obs_tgt['proprio'] is not None:
            loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
                dim=tuple(range(1, z_obs_pred["proprio"].ndim)) # Avg over T, D_p
            )
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt):
        """
        Compares all predicted frames with target frames (assuming T_pred == T_tgt).
        Loss calculated on all pred frames, weighted exponentially.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        T_pred = z_obs_pred["visual"].shape[1]
        coeffs = np.array(
            [base**i for i in range(T_pred)], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device) # (T,)

        # Calculate loss for each frame: (B, T)
        loss_visual_per_frame = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim)) # Avg over P, D -> (B, T)
        )
        loss_proprio_per_frame = 0
        if 'proprio' in z_obs_tgt and z_obs_tgt['proprio'] is not None:
             loss_proprio_per_frame = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
                dim=tuple(range(2, z_obs_pred["proprio"].ndim)) # Avg over D_p -> (B, T)
            )

        # Apply weights and sum over time dimension
        loss_visual = (loss_visual_per_frame * coeffs).sum(dim=1) # (B,)
        loss_proprio = (loss_proprio_per_frame * coeffs).sum(dim=1) if isinstance(loss_proprio_per_frame, torch.Tensor) else 0 # (B,)

        loss = loss_visual + alpha * loss_proprio
        return loss

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError

# --- 새로운 함수 추가 (수정됨) ---
def create_trajectory_objective_fn(alpha, goal_indices, goal_weights=None, comparison_mode="indexed_predicted", frameskip=1):
    """
    Creates a flexible objective function for image/trajectory goals.
    Compares predicted latent states against pre-encoded goal latent states.

    Args:
        alpha (float): Weight for the proprioceptive state difference.
        goal_indices (list[int]): List of indices specifying which goal frames from
                                   z_obs_g_traj to use. Can be either:
                                   - Environment step indices (if frameskip > 1 and input is in env steps)
                                   - Action step indices (default, frameskip=1 or already converted)
                                   Negative indices count from the end of the *prediction* horizon.
        goal_weights (list[float], optional): List of weights corresponding to goal_indices.
                                              If None, uniform weights are used. Defaults to None.
        comparison_mode (str): Specifies how to align predicted frames with target frames.
                               - "last_predicted": Compares the *last* predicted frame to the
                                                   *last* specified goal frame in goal_indices.
                               - "indexed_predicted": Compares predicted_frame[i] with goal_frame[i]
                                                      for each absolute index i derived from goal_indices.
                                                      Skips if index is out of prediction bounds.
        frameskip (int): Number of environment steps per action step. Used to convert
                         environment step indices to action step indices. Defaults to 1.

    Returns:
        function: The objective function.
    """
    # Convert environment step indices to action step indices if frameskip > 1
    # This assumes goal_indices are provided in environment step units
    # If already in action step units, frameskip=1 keeps them unchanged
    goal_indices_action = [idx if idx < 0 else idx // frameskip for idx in goal_indices]
    
    # Adjust goal_weights if provided (should match original goal_indices length)
    if goal_weights is None:
        goal_weights = [1.0 / len(goal_indices_action)] * len(goal_indices_action)
    if len(goal_weights) != len(goal_indices_action):
        raise ValueError(f"Length of goal_weights ({len(goal_weights)}) must match goal_indices ({len(goal_indices_action)}).")

    goal_weights = torch.tensor(goal_weights, dtype=torch.float32)
    # Use converted indices from now on
    goal_indices = goal_indices_action

    def objective_fn(z_obs_pred, z_obs_g_traj):
        """
        Calculates the weighted distance between predicted and goal latent states.

        Args:
            z_obs_pred (dict): Predicted latent states from the world model rollout.
                               {'visual': (B, T_pred, P, D), 'proprio': (B, T_pred, D_p)}
            z_obs_g_traj (dict): Pre-encoded goal latent states for the specified indices.
                                 {'visual': (B, N_goals, P, D), 'proprio': (B, N_goals, D_p)}
                                 where N_goals = len(goal_indices).

        Returns:
            torch.Tensor: The calculated loss per batch element (B,).
        """

        pred_visual = z_obs_pred["visual"]
        pred_proprio = z_obs_pred.get("proprio") # May not exist
        goal_visual = z_obs_g_traj["visual"]
        goal_proprio = z_obs_g_traj.get("proprio") # May not exist

        device = pred_visual.device
        _goal_weights = goal_weights.to(device)

        B, T_pred, P, D = pred_visual.shape
        N_goals = goal_visual.shape[1]

        total_loss = torch.zeros(B, device=device)

        # Convert potentially negative goal_indices to absolute indices based on prediction length T_pred
        absolute_pred_indices = [(idx if idx >= 0 else T_pred + idx) for idx in goal_indices]

        if comparison_mode == "last_predicted":
            # Compare only the last predicted frame with the last goal frame specified
            last_pred_idx_abs = T_pred - 1
            last_goal_idx_in_traj = N_goals - 1 # Index within z_obs_g_traj

            if last_pred_idx_abs >= 0:
                pred_visual_frame = pred_visual[:, last_pred_idx_abs:last_pred_idx_abs+1] # (B, 1, P, D)
                goal_visual_frame = goal_visual[:, last_goal_idx_in_traj:last_goal_idx_in_traj+1] # (B, 1, P, D)

                loss_visual = F.mse_loss(pred_visual_frame, goal_visual_frame, reduction='none').mean(dim=(1, 2, 3)) # (B,)

                loss_proprio = 0
                if pred_proprio is not None and goal_proprio is not None:
                    pred_proprio_frame = pred_proprio[:, last_pred_idx_abs:last_pred_idx_abs+1] # (B, 1, D_p)
                    goal_proprio_frame = goal_proprio[:, last_goal_idx_in_traj:last_goal_idx_in_traj+1] # (B, 1, D_p)
                    loss_proprio = F.mse_loss(pred_proprio_frame, goal_proprio_frame, reduction='none').mean(dim=(1, 2)) # (B,)

                total_loss = loss_visual + alpha * loss_proprio # Only one weight matters (implicitly 1.0)

        elif comparison_mode == "indexed_predicted":
            # Compare predicted_frame[i] with goal_frame[i] for each specified index i
            for i, pred_idx_abs in enumerate(absolute_pred_indices):
                goal_idx_in_traj = i # Index within z_obs_g_traj

                # Ensure the predicted index is valid
                if 0 <= pred_idx_abs < T_pred:
                    pred_visual_frame = pred_visual[:, pred_idx_abs:pred_idx_abs+1]
                    goal_visual_frame = goal_visual[:, goal_idx_in_traj:goal_idx_in_traj+1]

                    loss_visual = F.mse_loss(pred_visual_frame, goal_visual_frame, reduction='none').mean(dim=(1, 2, 3))

                    loss_proprio = 0
                    if pred_proprio is not None and goal_proprio is not None:
                        pred_proprio_frame = pred_proprio[:, pred_idx_abs:pred_idx_abs+1]
                        goal_proprio_frame = goal_proprio[:, goal_idx_in_traj:goal_idx_in_traj+1]
                        loss_proprio = F.mse_loss(pred_proprio_frame, goal_proprio_frame, reduction='none').mean(dim=(1, 2))

                    total_loss += _goal_weights[i] * (loss_visual + alpha * loss_proprio)

        # Removed the "all_goals_vs_last_predicted" elif block
        
        else:
            raise ValueError(f"Unknown comparison_mode: {comparison_mode}. Supported modes are 'last_predicted', 'indexed_predicted'.")

        return total_loss

    return objective_fn