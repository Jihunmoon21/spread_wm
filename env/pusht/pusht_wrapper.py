import os
import numpy as np
import gym
from env.pusht.pusht_env import PushTEnv
from utils import aggregate_dct

class PushTWrapper(PushTEnv):
    def __init__(
            self, 
            with_velocity=True,
            with_target=True,
            shape="I",
            color="LightSlateGray",
            background_color="White",
        ):
        super().__init__(
            with_velocity=with_velocity,
            with_target=with_target,
            shape=shape,
            color=color,
            background_color=background_color,
        )
        self.action_dim = self.action_space.shape[0]
    
    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)
        
        def generate_state():
            if self.with_velocity:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                        0,
                        0,  # agent velocities default 0
                    ]
                )
            else:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                    ]
                )
        
        init_state = generate_state()
        goal_state = generate_state()
        
        return init_state, goal_state
    
    def update_env(self, env_info):
        if 'shape' in env_info:
            self.shape = env_info['shape']
        if 'color' in env_info:
            self.color = env_info['color']
        if 'background_color' in env_info:
            self.background_color = env_info['background_color']
            
    def eval_state(self, goal_state, cur_state):
        """
        Return True if the goal is reached
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        """
        # 상태 차원 안전 처리
        goal_state = np.array(goal_state)
        cur_state = np.array(cur_state)
        
        # 최소 공통 차원으로 맞춤
        min_dim = min(len(goal_state), len(cur_state))
        goal_state_safe = goal_state[:min_dim]
        cur_state_safe = cur_state[:min_dim]
        
        # 위치 비교 (최소 4차원 필요)
        if min_dim >= 4:
            pos_diff = np.linalg.norm(goal_state_safe[:4] - cur_state_safe[:4])
        else:
            # fallback: 사용 가능한 차원만 사용
            pos_diff = np.linalg.norm(goal_state_safe[:min_dim] - cur_state_safe[:min_dim])
        
        # 각도 비교 (5차원 이상일 때만)
        if min_dim >= 5:
            angle_diff = np.abs(goal_state_safe[4] - cur_state_safe[4])
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            success = pos_diff < 20 and angle_diff < np.pi / 9
        else:
            # 각도 정보가 없으면 위치만으로 판단
            success = pos_diff < 20
        
        # 전체 상태 거리 계산 (안전한 차원으로)
        state_dist = np.linalg.norm(goal_state_safe - cur_state_safe)

        return {
            'success': success,
            'state_dist': state_dist,
        }

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.seed(seed)
        self.reset_to_state = init_state
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
