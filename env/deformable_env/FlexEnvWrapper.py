from .src.sim.sim_env.flex_env import FlexEnv

import os
import numpy as np
import gym
import torch
import math
from .src.sim.utils import load_yaml

ENV_ACTION_DIM = 4
BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../"))


def aggregate_dct(dcts):
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct

def chamfer_distance(x, y):
    # x: [B, N, D]
    # y: [B, M, D]
    # NOTE: only the first 3 dim is taken!
    x = x[:, :, None, :3].repeat(1, 1, y.size(1), 1) # x: [B, N, M, D]
    y = y[:, None, :, :3].repeat(1, x.size(1), 1, 1) # y: [B, N, M, D]
    dis = torch.norm(torch.add(x, -y), 2, dim=3)    # dis: [B, N, M]
    dis_xy = torch.mean(torch.min(dis, dim=2)[0])   # dis_xy: mean over N
    dis_yx = torch.mean(torch.min(dis, dim=1)[0])   # dis_yx: mean over M
    return dis_xy + dis_yx

class FlexEnvWrapper(FlexEnv):
    def __init__(
        self,
        object_name,
        *,
        table_color=None,
        camera_view=None,
        granular_radius=None,
    ):
        config = load_yaml(os.path.join(BASE_DIR, f"conf/env/{object_name}.yaml"))

        dataset_cfg = config.get("dataset", {})

        if camera_view is not None:
            dataset_cfg["camera_view"] = int(camera_view)

        if table_color is not None:
            dataset_cfg["table_color"] = str(table_color)

        if granular_radius is not None:
            dataset_cfg.setdefault("obj_params", {})
            dataset_cfg["obj_params"]["radius"] = float(granular_radius)

        config["dataset"] = dataset_cfg

        super().__init__(config=config)
        self.action_dim = 4
        self.proprio_start_idx = 0
        self.proprio_end_idx = 2
        self.success_threshold = 0
        # Goal pattern selection: 0=random(default), 1=cross-void 4 squares, 2=top-bottom split, 3=central square
        # You can override by exporting environment variable: GOAL_MODE=1|2|3
        try:
            self.goal_mode = int(os.getenv("GOAL_MODE", "0"))
        except Exception:
            self.goal_mode = 0
        # Flags for measuring Initial CD after set_states
        self._measure_initial_cd = False
        self._goal_state_for_cd = None
        # Flag for forcing recentering after set_states (for final output)
        self._force_recenter_after_set_states = False
    
    def set_measure_initial_cd(self, measure, goal_state):
        """Set flag to measure Initial CD after next set_states call"""
        self._measure_initial_cd = measure
        self._goal_state_for_cd = goal_state

    def eval_state(self, goal_state, cur_state):
        CD = chamfer_distance(torch.tensor([goal_state]), torch.tensor([cur_state]))
        print("CD: ", CD.item())
        success, chamfer_dist = CD.item() < 0.4, CD.item() # CD 최저 임계값 0.4
        metrics = {
            "success": success,
            "chamfer_distance": chamfer_dist,
        }
        return metrics

    def update_env(self, env_info): 
        pass

    def sample_random_init_goal_states(self, seed):
        """
        Return a random state
        """
        self.seed(seed)
        imgs_list, particle_pos_list, eef_states_list = self.reset(save_data=True)

        def transfer_state(state, scale, theta, delta):
            assert state.shape[1] == 4
            state = state.clone()
            state[:, 0] *= scale
            state[:, 2] *= scale
            theta = math.radians(theta)
            rotation_matrix_y = torch.tensor(
                [
                    [math.cos(theta), 0, math.sin(theta)],
                    [0, 1, 0],
                    [-math.sin(theta), 0, math.cos(theta)],
                ]
            )
            rotated_state = torch.matmul(state[:, :3], rotation_matrix_y.T)
            state[:, :3] = rotated_state

            # randomly select negative or positive delta
            delta_x = delta * np.random.choice([-1, 1])
            delta_z = delta * np.random.choice([-1, 1])

            state[:, 0] += delta_x
            state[:, 2] += delta_z
            return state

        def make_grid_centers(n_particles, center_x, center_z, half_span, n_side=None):
            # Create approximately square grid of positions within a square centered at (center_x, center_z)
            if n_side is None:
                n_side = int(max(1, round(n_particles ** 0.5)))
            # ensure enough cells
            while n_side * n_side < n_particles:
                n_side += 1
            xs = torch.linspace(-half_span, half_span, steps=n_side) + center_x
            zs = torch.linspace(-half_span, half_span, steps=n_side) + center_z
            X, Z = torch.meshgrid(xs, zs, indexing='ij')
            positions = torch.stack([X.reshape(-1), Z.reshape(-1)], dim=-1)
            return positions[:n_particles]
        
        def make_rectangular_grid(n_particles, center_x, center_z, half_span_x, half_span_z):
            # Create rectangular grid with width:height = 1:2 ratio (height:width = 2:1)
            # half_span_z should be twice half_span_x
            # Calculate grid dimensions to fit n_particles in 1:2 rectangle
            aspect_ratio = 0.5  # width:height = 1:2 (or height:width = 2:1)
            # Estimate grid size: n_side_x * n_side_z >= n_particles, with n_side_z / n_side_x ≈ 2
            n_side_x = int(max(1, round((n_particles * aspect_ratio) ** 0.5)))
            n_side_z = int(max(1, round(n_side_x / aspect_ratio)))
            # Ensure enough cells
            while n_side_x * n_side_z < n_particles:
                if (n_side_x + 1) * n_side_z >= n_particles:
                    n_side_x += 1
                else:
                    n_side_z += 1
            xs = torch.linspace(-half_span_x, half_span_x, steps=n_side_x) + center_x
            zs = torch.linspace(-half_span_z, half_span_z, steps=n_side_z) + center_z
            X, Z = torch.meshgrid(xs, zs, indexing='ij')
            positions = torch.stack([X.reshape(-1), Z.reshape(-1)], dim=-1)
            return positions[:n_particles]

        def layout_goal_by_mode(state, mode):
            # state: (N,4) tensor
            pos = state[:, :3].clone()
            N = pos.shape[0]
            # compute current center and span to scale target nicely
            x_min, x_max = pos[:, 0].min().item(), pos[:, 0].max().item()
            z_min, z_max = pos[:, 2].min().item(), pos[:, 2].max().item()
            cx = 0.5 * (x_min + x_max)
            cz = 0.5 * (z_min + z_max)
            span = max(x_max - x_min, z_max - z_min)
            # base grid side guess
            n_side = int(max(1, round(N ** 0.5)))
            # reasonable half-span for clusters
            half_span = 0.25 * span

            if mode == 1:
                # Cross-void in center with four diagonal quadrants: split particles into 4 equal clusters
                # Place clusters at diagonal corners (quadrants) leaving center cross-shaped void
                counts = [N // 4, N // 4, N // 4, N - 3 * (N // 4)]
                d = 0.35 * span  # distance from center to cluster centers (diagonal)
                centers = [
                    (cx - d, cz - d),  # bottom-left (quadrant 3)
                    (cx + d, cz - d),  # bottom-right (quadrant 4)
                    (cx - d, cz + d),  # top-left (quadrant 2)
                    (cx + d, cz + d),  # top-right (quadrant 1)
                ]
                targets = []
                for k, (cxk, czk) in enumerate(centers):
                    pts = make_grid_centers(counts[k], cxk, czk, half_span * 0.6)
                    targets.append(pts)
                target_xz = torch.cat(targets, dim=0)
            elif mode == 2:
                # Split horizontally: left and right rectangles separated by a gap
                # Each rectangle has width:height = 1:2 ratio (height:width = 2:1)
                n_left = N // 2
                n_right = N - n_left
                try:
                    gap_scale = float(os.getenv("GOAL_GAP", "0.60"))  # Default 0.50
                except Exception:
                    gap_scale = 0.60
                gap = gap_scale * span
                # For 1:2 rectangles, half_span_z should be twice half_span_x
                # Use half_span for x (width), and 2*half_span for z (height)
                rect_half_span_x = half_span * 0.6  # width
                rect_half_span_z = rect_half_span_x * 2.0  # height = 2 * width
                left_center = (cx - gap, cz)
                right_center = (cx + gap, cz)
                left_pts = make_rectangular_grid(n_left, left_center[0], left_center[1], rect_half_span_x, rect_half_span_z)
                right_pts = make_rectangular_grid(n_right, right_center[0], right_center[1], rect_half_span_x, rect_half_span_z)
                target_xz = torch.cat([left_pts, right_pts], dim=0)
            elif mode == 3:
                # Cross shape centered at (cx, cz): vertical line + horizontal line
                # 십자가 모양: 중앙에 세로선과 가로선이 교차
                try:
                    cross_length_scale = float(os.getenv("GOAL_CROSS_LENGTH", "0.35"))  # 십자가 길이 (span 비율)
                except Exception:
                    cross_length_scale = 0.4
                cross_length = cross_length_scale * span
                
                # 입자 수 분배: 세로선과 가로선으로 나눔 (교차점은 세로선에 포함)
                n_vertical = (N + 1) // 2  # 세로선 입자 수 (교차점 포함)
                n_horizontal = N - n_vertical  # 가로선 입자 수 (교차점 제외)
                
                # 세로선: x = cx, z는 [cz - cross_length, cz + cross_length]
                z_vertical = torch.linspace(cz - cross_length, cz + cross_length, steps=n_vertical)
                x_vertical = torch.full((n_vertical,), cx)
                vertical_line = torch.stack([x_vertical, z_vertical], dim=-1)
                
                # 가로선: z = cz, x는 [cx - cross_length, cx + cross_length] (교차점 제외)
                # 교차점 (cx, cz)를 제외하기 위해 양쪽으로 분할
                if n_horizontal > 0:
                    n_left = n_horizontal // 2
                    n_right = n_horizontal - n_left
                    
                    # 왼쪽: x < cx
                    if n_left > 0:
                        x_left = torch.linspace(cx - cross_length, cx - cross_length * 0.1, steps=n_left)
                        z_left = torch.full((n_left,), cz)
                        left_line = torch.stack([x_left, z_left], dim=-1)
                    else:
                        left_line = torch.empty((0, 2))
                    
                    # 오른쪽: x > cx
                    if n_right > 0:
                        x_right = torch.linspace(cx + cross_length * 0.1, cx + cross_length, steps=n_right)
                        z_right = torch.full((n_right,), cz)
                        right_line = torch.stack([x_right, z_right], dim=-1)
                    else:
                        right_line = torch.empty((0, 2))
                    
                    # 가로선 결합
                    horizontal_line = torch.cat([left_line, right_line], dim=0)
                else:
                    horizontal_line = torch.empty((0, 2))
                
                # 세로선과 가로선 결합
                target_xz = torch.cat([vertical_line, horizontal_line], dim=0)
                
                # 입자 수가 정확히 맞도록 조정 (필요시)
                if target_xz.shape[0] > N:
                    target_xz = target_xz[:N]
                elif target_xz.shape[0] < N:
                    # 부족한 경우 마지막 점을 복제
                    n_missing = N - target_xz.shape[0]
                    target_xz = torch.cat([target_xz, target_xz[-n_missing:]], dim=0)
                
                print(f"[DEBUG] Goal mode 3 (Cross): cx={cx:.3f}, cz={cz:.3f}, span={span:.3f}, cross_length={cross_length:.3f}, n_vertical={n_vertical}, n_horizontal={n_horizontal}")
            else:
                # Fallback to random-like transformation
                goal_scale, goal_theta, goal_delta = (
                    np.random.uniform(0.6, 0.9),
                    0,
                    np.random.uniform(-1, 1),
                )
                return transfer_state(state.clone(), goal_scale, goal_theta, goal_delta)

            goal = state.clone()
            goal[:, 0] = target_xz[:, 0]
            goal[:, 2] = target_xz[:, 1]
            # keep y and w the same
            return goal

        if self.obj == 'granular':
            state = torch.tensor(particle_pos_list[0])
            if getattr(self, 'goal_mode', 0) in (1, 2, 3):
                goal_state = layout_goal_by_mode(state, self.goal_mode)
            else:
                goal_scale, goal_theta, goal_delta = (
                    np.random.uniform(0.6, 0.9),
                    0,
                    np.random.uniform(-1, 1),
                )
                goal_state = transfer_state(state, goal_scale, goal_theta, goal_delta)
            return state, goal_state
        
        elif self.obj == 'rope':
            state = torch.tensor(particle_pos_list[0])
            goal_scale, goal_theta, goal_delta = (
                1,
                np.random.uniform(0, 90),
                np.random.uniform(1, -1),
            )
            goal_state = transfer_state(state, goal_scale, goal_theta, goal_delta)
            return state, goal_state


    def prepare(self, seed, init_state, force_recenter=None):
        """
        Reset with controlled init_state
        Args:
            force_recenter: If provided, use this value. Otherwise, check flag.
        """
        self.seed(seed)
        # 🔧 final output 계산 시 reset()과 set_states() 모두에서 재센터링 적용
        # force_recenter 파라미터가 제공되면 사용, 없으면 플래그 확인
        if force_recenter is None:
            force_recenter = getattr(self, '_force_recenter_after_set_states', False)
        if force_recenter:
            imgs_list, particle_pos_list, eef_states_list = self.reset(save_data=True, force_recenter=True)
        else:
            imgs_list, particle_pos_list, eef_states_list = self.reset(save_data=True)
        # set_states()에도 force_recenter 전달 (reset()에서 플래그를 확인했지만, set_states()에서도 확인하도록)
        self.set_states(init_state, force_recenter=force_recenter)
        if force_recenter:
            # 플래그는 한 번만 사용하므로 리셋하지 않음 (다음 final output을 위해 유지)
            pass
        # Measure Initial CD after set_states (first time only)
        # This will be called by evaluator to measure CD after first set_states
        if hasattr(self, '_measure_initial_cd') and self._measure_initial_cd:
            if hasattr(self, '_goal_state_for_cd') and self._goal_state_for_cd is not None:
                goal_state = self._goal_state_for_cd
                current_state = self.get_positions().reshape(-1, 4)
                if isinstance(current_state, np.ndarray):
                    current_state = torch.tensor(current_state)
                if isinstance(goal_state, np.ndarray):
                    goal_state = torch.tensor(goal_state)
                if len(current_state.shape) == 2:
                    current_state = current_state.unsqueeze(0)
                if len(goal_state.shape) == 2:
                    goal_state = goal_state.unsqueeze(0)
                initial_cd = chamfer_distance(current_state, goal_state)
                print(f"[CEM INIT] Initial CD (after first set_states): {initial_cd.item():.6f}")
                self._measure_initial_cd = False  # Only measure once
        img = self.get_one_view_img()
        obs = {
            "visual": img[..., :3],
            "proprio": np.zeros(1).astype(np.float32),
        }

        state_dct = {"state": particle_pos_list[-1], "proprio": eef_states_list[-1]}
        return obs, state_dct

    def step_multiple(self, actions):
        obses = []
        infos = []
        for action in actions:
            step_data = [], [], []
            obs, out_data = self.step(
                action, save_data=True, data=step_data
            )  # o: (H, W, 5)
            imgs_list, particle_pos_list, eef_states_list = (
                out_data  # imgs_list: (num_cameras, H, W, 5); particle_pos_list: (n_particles, 4); eef_states_list: (1,14)
            )

            obs = {
                "visual": imgs_list[-1][self.camera_view][..., :3],
                "proprio": np.zeros(1).astype(np.float32), # dummy proprio
            }
            info = {"pos_agent": eef_states_list[-1], "state": particle_pos_list[-1]} # dummy 
            obses.append(obs)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = 0
        dones = False
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions, force_recenter=None):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        force_recenter: If provided, use this value. Otherwise, check flag.
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        # 🔧 final output 계산 시 플래그 확인 및 prepare()에 전달
        if force_recenter is None:
            force_recenter = getattr(self, '_force_recenter_after_set_states', False)
        if force_recenter:
            # 🔧 플래그를 명시적으로 설정 (확실히 전달되도록)
            self._force_recenter_after_set_states = True
        obs, state_dct = self.prepare(seed, init_state, force_recenter=force_recenter)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state_dct["state"], 0), infos["state"]])
        states = np.stack(states)
        return obses, states
