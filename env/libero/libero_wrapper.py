import os
import numpy as np
import gym
import bddl
import robosuite as suite
import libero.libero.utils.utils as libero_utils
from libero.libero import get_libero_path, benchmark
import mujoco  # MuJoCo 3.x API
import pickle  # LBP 방식(.pkl)을 위해 추가
from utils import aggregate_dct

# LIBERO 경로 설정
os.environ["LIBERO_ROOT"] = "/home/jihun/LIBERO/libero"

# robosuite OSC_POSE 컨트롤러 기본 설정
DEFAULT_CONTROLLER_CONFIG = {
    "type": "OSC_POSE",
    "input_max": 1,
    "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150,
    "damping": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300],
    "damping_limits": [0, 10],
    "position_limits": None,
    "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
    "ramp_ratio": 0.2,
}


class LiberoWrapper(gym.Env):
    """
    Libero(robosuite) 환경을 spread_wm 플래너 호환 Gym(0.25.x) 스타일로 래핑.
    
    [LBP 방식 수정]
    - '.npy' 및 '.bddl' 파일을 직접 로드하는 대신,
      LBP의 'assets/libero.pkl' 파일에서 메타데이터를 로드합니다.
    - reset() 시그니처를 spread_wm 프로젝트 스타일에 맞춤.
    """
    """
    Args:
        task_name (str): 실행할 태스크의 전체 이름 (예: "libero_spatial.move_the_block_to_the_corner_of_the_table")
        task_suite_name (str): 태스크가 속한 스위트 이름 (예: "libero_spatial", "libero_10")
        libero_pkl_path (str): LBP의 'assets/libero.pkl' 파일 경로
        camera_name (str, optional): 카메라 뷰.
        img_size (int, optional): 이미지 해상도.
        controller_config (dict, optional): Robosuite 컨트롤러 설정.
    """
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        task_name: str,
        task_suite_name: str,
        libero_pkl_path: str,
        camera_name="agentview",
        img_size=256,
        controller_config=None,
        **kwargs, # plan_libero.yaml에 혹시 남아있을 libero_data_path 등을 무시
    ):

        super().__init__()

        # 1) LBP의 .pkl 파일에서 벤치마크 메타데이터 로드
        try:
            with open(libero_pkl_path, "rb") as f:
                self.benchmark_obj = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Libero pkl 파일({libero_pkl_path})을 찾을 수 없습니다.")
        
        benchmark_dict = benchmark.get_benchmark_dict()

        # 2) 태스크 스위트(suite) 가져오기
        if task_suite_name not in benchmark_dict:
            raise ValueError(
                f"Task suite '{task_suite_name}'가 벤치마크에 없습니다. "
                f"사용 가능: {list(benchmark_dict.keys())}"
            )
        self.task_suite = benchmark_dict[task_suite_name]()

        # 3) task_name을 기반으로 task index와 초기 상태(init_states) 탐색
        found_task = False
        task_i = None
        for i in range(self.task_suite.n_tasks):
            task_obj = self.task_suite.get_task(i)
            
            if task_obj.name == task_name:
                task_i = i
                self.all_init_states = self.task_suite.get_task_init_states(i)
                found_task = True
                break
        
        if not found_task:
            raise ValueError(f"Task '{task_name}'를 suite '{task_suite_name}'에서 찾을 수 없습니다.")

        self.current_init_state_idx = 0

        # 4) LIBERO 환경 생성
        if controller_config is None:
            controller_config = DEFAULT_CONTROLLER_CONFIG

        # LIBERO 환경 클래스 사용 (TASK_MAPPING을 통해)
        from libero.libero.envs.bddl_base_domain import TASK_MAPPING
        
        # BDDL 파일 경로 가져오기
        bddl_file_path = self.task_suite.get_task_bddl_file_path(task_i)
        
        # BDDL 파일에서 problem_name을 확인하여 적절한 환경 클래스 선택
        with open(bddl_file_path, 'r') as f:
            bddl_content = f.read()
        
        # problem_name 추출 (예: "LIBERO_Floor_Manipulation", "LIBERO_Tabletop_Manipulation")
        import re
        problem_match = re.search(r'\(define \(problem (\w+)\)', bddl_content)
        if problem_match:
            problem_name = problem_match.group(1).lower()
            
            # problem_name에 맞는 환경 클래스 찾기
            env_class = None
            for key, cls in TASK_MAPPING.items():
                if problem_name in key or key in problem_name:
                    env_class = cls
                    break
            
            if env_class is None:
                # 기본값으로 tabletop_manipulation 사용 (대부분의 태스크가 이 클래스 사용)
                env_class = TASK_MAPPING.get("libero_tabletop_manipulation")
        else:
            # BDDL 파싱 실패시 기본값 사용
            env_class = TASK_MAPPING.get("libero_tabletop_manipulation")
        
        if env_class is None:
            raise ValueError("No suitable environment class found in TASK_MAPPING")
        
        # 4-1) 기본 kwargs 구성 후, 사용자가 전달한 **kwargs(YAML)를 병합하여 우선 적용
        base_kwargs = {
            "robots": ["Panda"],  # 리스트 형태로 수정하여 단일 로봇 명시
            "controller_configs": controller_config,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_camera_obs": True,
            "camera_names": camera_name,
            "camera_heights": img_size,
            "camera_widths": img_size,
            "control_freq": 20,
            "reward_shaping": False,
            "hard_reset": False,
        }
        # 사용자가 넘긴 추가 인자(**kwargs)를 우선순위 높게 병합
        if isinstance(kwargs, dict) and len(kwargs) > 0:
            base_kwargs.update(kwargs)

        # 안전장치: 일부 LIBERO 클래스에서 명시적 인자를 요구
        # 없으면 기본값을 채워 크래시 방지
        table_full_size = kwargs.pop("table_full_size", (1.0, 1.2, 0.05))
        # 타입 정규화: YAML/Config에서 문자열로 들어오는 경우가 있어 float 튜플로 변환
        try:
            if isinstance(table_full_size, (list, tuple)):
                table_full_size = tuple(float(x) for x in table_full_size)
            elif isinstance(table_full_size, str):
                # "0.8,0.8,0.05" 형태 지원
                parts = [p.strip() for p in table_full_size.split(",") if p.strip()]
                if len(parts) == 3:
                    table_full_size = tuple(float(p) for p in parts)
        except Exception:
            # 문제가 있으면 안전한 기본값 사용
            table_full_size = (1.0, 1.2, 0.05)

        # LIBERO 버그들을 우회하는 수정사항 적용 (병합된 kwargs 기반)
        kwargs = self._apply_libero_fixes(env_class, base_kwargs)

        # 래퍼/메타 키 제거: 환경 생성자에 전달되면 오류 발생 가능
        for k in [
            "_target_",
            "task_name",
            "task_suite_name",
            "libero_pkl_path",
            "camera_name",
            "img_size",
            "controller_config",
        ]:
            kwargs.pop(k, None)

        # 일부 클래스는 table_full_size를 명시적 인자로만 받음
        table_full_size = kwargs.pop("table_full_size", (1.0, 1.2, 0.05))
        
        self.env = env_class(bddl_file_name=bddl_file_path, table_full_size=table_full_size, **kwargs)

        # 내부 상태
        self.camera_name = camera_name
        self.img_size = img_size
        self.raw_obs = None

        # 5) action_space: env 정의에 동기화
        self.action_dim = getattr(self.env, "action_dim", 7)
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # 6) observation_space / proprio_dim: reset 이후 지연 초기화
        self._observation_space = None
        self.proprio_dim = None

    # ---------- LIBERO 버그 우회 메서드들 ----------
    def _fix_libero_robot_mapping(self):
        """LIBERO의 ROBOT_CLASS_MAPPING 이슈를 수정"""
        from robosuite.environments.manipulation.manipulation_env import ROBOT_CLASS_MAPPING
        from robosuite.robots.single_arm import SingleArm
        
        # LIBERO 로봇들을 ROBOT_CLASS_MAPPING에 추가
        ROBOT_CLASS_MAPPING.update({
            "MountedPanda": SingleArm,
            "OnTheGroundPanda": SingleArm,
            "OnTheGroundP": SingleArm,
            "OnTheGrounda": SingleArm,
            "OnTheGroundn": SingleArm,  # 잘린 이름도 추가
            "OnTheGroundd": SingleArm,
            "OnTheGroundOnTheGroundPanda": SingleArm,  # 중복 변환된 이름도 추가
        })

    def _fix_robot_name_conversion(self, env_class, kwargs):
        """로봇 이름 변환 이슈를 우회 - LIBERO의 변환 로직을 완전히 우회"""
        # robots 파라미터가 있으면 올바른 형태로 보장
        if "robots" in kwargs:
            robots = kwargs["robots"]
            if isinstance(robots, str):
                # 문자열인 경우 리스트로 변환
                kwargs["robots"] = [robots]
            elif isinstance(robots, list) and len(robots) > 1:
                # 여러 로봇이 있는 경우 첫 번째만 사용
                kwargs["robots"] = [robots[0]]
        
        # floor_manipulation의 경우 원본 Panda 이름을 그대로 사용
        # (libero_floor_manipulation.py에서 변환 로직을 비활성화했으므로)
        return kwargs

    def _fix_tabletop_manipulation_bug(self, env_class, kwargs):
        """tabletop_manipulation의 table_full_size 버그를 우회"""
        if "tabletop_manipulation" in str(env_class):
            # table_full_size가 kwargs에 있으면 올바르게 처리
            if "table_full_size" in kwargs:
                # 이미 올바른 값이므로 그대로 사용
                pass
            else:
                # 기본값 설정
                kwargs["table_full_size"] = (1.0, 1.2, 0.05)
        return kwargs

    def _apply_libero_fixes(self, env_class, kwargs):
        """LIBERO의 알려진 이슈들을 모두 수정"""
        # 1. 로봇 매핑 수정
        self._fix_libero_robot_mapping()
        
        # 2. 로봇 이름 변환 수정
        kwargs = self._fix_robot_name_conversion(env_class, kwargs)
        
        # 3. tabletop_manipulation 버그 수정
        kwargs = self._fix_tabletop_manipulation_bug(env_class, kwargs)
        
        return kwargs

    # ---------- 내부 유틸 ----------
    def _force_obs(self):
        """robosuite 최신 관측 강제 갱신."""
        return self.env._get_observations(force_update=True)

    def _to_spread_obs(self, obs_dict):
        """robosuite obs_dict -> spread_wm 형식으로 변환."""
        img = obs_dict[f"{self.camera_name}_image"]  # (H,W,3) uint8
        # Robosuite 환경의 카메라 출력은 상하 반전되어 있음
        # 학습 데이터(HDF5)는 정상 방향이므로, 환경 이미지를 flip하여 일치시킴
        img = np.flipud(img)
        img = np.fliplr(img)
        # HDF5 형식과 일치시키기 위해 (H, W, C) 유지, uint8 유지
        full_proprio = obs_dict["robot0_proprio-state"].astype(np.float32)
        
        # LIBERO HDF5와 Robosuite의 proprio 구조:
        # - LIBERO HDF5: [joint_pos(7), gripper_qpos(2)] = 9차원
        # - Robosuite: [joint_pos(7), joint_vel(7), gripper_qpos(2), ...] = 39차원
        # 
        # Robosuite proprio_state 구조 (Panda 로봇):
        # [0:7]   - joint positions
        # [7:14]  - joint velocities
        # [14:16] - gripper qpos
        # [16:...]- 추가 정보 (eef, objects 등)
        
        joint_pos = full_proprio[:7]        # 7차원
        gripper_qpos = full_proprio[14:16]  # 2차원
        state = np.concatenate([joint_pos, gripper_qpos])  # 9차원
        
        # 키 이름을 spread_wm 표준에 맞게 수정
        return {"visual": img, "proprio": state}

    def _ensure_spaces(self, obs_dict):
        """첫 reset 이후 관측 기반으로 observation_space / proprio_dim 설정."""
        if self._observation_space is not None:
            return
        # LIBERO HDF5 데이터셋과 일치 (joint_pos 7 + gripper_qpos 2 = 9차원)
        self.proprio_dim = 9
        
        # 실제 이미지 shape 확인
        img = obs_dict[f"{self.camera_name}_image"]
        img_h, img_w = img.shape[:2]
        
        self._observation_space = gym.spaces.Dict(
            {
                "visual": gym.spaces.Box(
                    low=0, high=255, shape=(img_h, img_w, 3), dtype=np.uint8
                ),
                "proprio": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.proprio_dim,), dtype=np.float32
                ),
            }
        )

    def _apply_state(self, state_like):
        """
        초기/임의 상태를 MuJoCo 표준(qpos/qvel)로 적용.
          - dict: {'qpos': ..., 'qvel': ...}
          - 1D array: 길이 >= nq+nv 이면 qpos/qvel로 분해
        """
        sim = self.env.sim
        nq = sim.model.nq
        nv = sim.model.nv

        if isinstance(state_like, dict) and "qpos" in state_like and "qvel" in state_like:
            qpos = np.asarray(state_like["qpos"], dtype=np.float64).reshape(nq)
            qvel = np.asarray(state_like["qvel"], dtype=np.float64).reshape(nv)
        else:
            flat = np.asarray(state_like, dtype=np.float64).ravel()
            if flat.size < nq + nv:
                raise ValueError(
                    f"State vector too short: got {flat.size}, need at least {nq+nv}"
                )
            qpos = flat[:nq]
            qvel = flat[nq : nq + nv]

        sim.data.qpos[:] = qpos
        sim.data.qvel[:] = qvel
        sim.forward()  # robosuite의 sim.forward() 사용

    # ---------- Gym 0.25 API (spread_wm 호환) ----------
    def reset(self):
        super().reset()
        _ = self.env.reset()  # robosuite도 obs를 반환하지만 이후 강제 관측으로 덮음

        # 데모 초기 상태 적용
        try:
            init_state_info = self.all_init_states[self.current_init_state_idx]
        except IndexError:
            if len(self.all_init_states) == 0:
                print("경고: 'all_init_states'가 비어있습니다. MuJoCo 기본 리셋 상태를 사용합니다.")
                self.raw_obs = self._force_obs()
                self._ensure_spaces(self.raw_obs)
                return self.get_obs()
                
            self.current_init_state_idx = 0
            init_state_info = self.all_init_states[0]

        if isinstance(init_state_info, dict):
            if "states" in init_state_info and len(init_state_info["states"]) > 0:
                init_state = init_state_info["states"][0]
            else:
                init_state = init_state_info  # {'qpos','qvel'} 가능
        else:
            init_state = init_state_info

        self._apply_state(init_state)

        # 다음 인덱스 순환
        self.current_init_state_idx = (self.current_init_state_idx + 1) % len(self.all_init_states)

        # 최신 관측 반영 및 스페이스 확정
        self.raw_obs = self._force_obs()
        self._ensure_spaces(self.raw_obs)

        return self.get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.raw_obs, reward, done, info = self.env.step(action)

        # 성공 판정(있으면 사용)
        success = False
        check = getattr(self.env, "_check_success", None)
        if callable(check):
            try:
                success = bool(check())
            except Exception:
                success = bool(info.get("success", False))
        info["success"] = success

        return self.get_obs(), float(reward), bool(done), info

    def render(self, mode="rgb_array"):
        # 러너가 인자 없이 호출하더라도 동작하도록 완화
        if mode not in (None, "rgb_array"):
            raise NotImplementedError("Only 'rgb_array' mode is supported.")
        obs = self._force_obs()
        img = obs[f"{self.camera_name}_image"]  # (H,W,3) uint8
        return img

    def seed(self, seed=None):
        """환경의 랜덤 시드를 설정합니다."""
        if seed is not None:
            np.random.seed(seed)
            # robosuite 환경에도 시드 전파 (가능한 경우)
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)
        return [seed]

    # ---------- Spread_wm 플래너 API ----------
    def get_obs(self):
        if self.raw_obs is None:
            self.raw_obs = self._force_obs()
        return self._to_spread_obs(self.raw_obs)

    def get_state(self):
        qpos = self.env.sim.data.qpos.copy()
        qvel = self.env.sim.data.qvel.copy()
        return np.concatenate([qpos, qvel])

    def set_state(self, state):
        self._apply_state(state)
        self.raw_obs = self._force_obs()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        # reset 이전 접근을 위한 임시 폴백(실제 reset 후 덮어씀)
        if self._observation_space is None:
            self._observation_space = gym.spaces.Dict(
                {
                    "img": gym.spaces.Box(
                        low=0.0, high=1.0, shape=(3, self.img_size, self.img_size), dtype=np.float32
                    ),
                    "state": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                    ),
                }
            )
        return self._observation_space

    # ---------- SubprocVectorEnv 호환 메서드 ----------
    def sample_random_init_goal_states(self, seed):
        """
        all_init_states에서 랜덤하게 초기 상태와 목표 상태를 선택합니다.
        
        Args:
            seed (int): 랜덤 시드
            
        Returns:
            init_state, goal_state: 두 개의 상태 (qpos + qvel)
        """
        rs = np.random.RandomState(seed)
        
        if len(self.all_init_states) == 0:
            # 초기 상태가 없으면 현재 상태를 사용
            print("경고: all_init_states가 비어있습니다. 현재 상태를 사용합니다.")
            current_state = self.get_state()
            return current_state.copy(), current_state.copy()
        
        # all_init_states에서 랜덤하게 두 개 선택
        n_states = len(self.all_init_states)
        init_idx = rs.randint(0, n_states)
        goal_idx = rs.randint(0, n_states)
        
        # 초기 상태 추출
        init_state_info = self.all_init_states[init_idx]
        if isinstance(init_state_info, dict):
            if "states" in init_state_info and len(init_state_info["states"]) > 0:
                init_state = init_state_info["states"][0]
            else:
                init_state = init_state_info
        else:
            init_state = init_state_info
        
        # 목표 상태 추출
        goal_state_info = self.all_init_states[goal_idx]
        if isinstance(goal_state_info, dict):
            if "states" in goal_state_info and len(goal_state_info["states"]) > 0:
                goal_state = goal_state_info["states"][0]
            else:
                goal_state = goal_state_info
        else:
            goal_state = goal_state_info
        
        # dict 형태면 qpos/qvel을 concat하여 반환
        def state_to_array(state):
            if isinstance(state, dict) and "qpos" in state and "qvel" in state:
                qpos = np.asarray(state["qpos"], dtype=np.float64).ravel()
                qvel = np.asarray(state["qvel"], dtype=np.float64).ravel()
                return np.concatenate([qpos, qvel])
            else:
                return np.asarray(state, dtype=np.float64).ravel()
        
        init_state_array = state_to_array(init_state)
        goal_state_array = state_to_array(goal_state)
        
        return init_state_array, goal_state_array

    def update_env(self, env_info):
        """
        환경 설정을 업데이트합니다.
        
        Args:
            env_info (dict): 환경 정보 딕셔너리
        """
        # Libero의 경우 task를 변경하는 것은 복잡하므로
        # 일단 단순하게 구현 (필요시 확장 가능)
        if 'task_name' in env_info:
            print(f"경고: task 변경은 현재 미지원입니다. (요청: {env_info['task_name']})")
        # 다른 설정들도 필요시 추가 가능
        pass

    def prepare(self, seed, init_state):
        """
        주어진 초기 상태로 환경을 리셋합니다.
        
        Args:
            seed (int): 랜덤 시드
            init_state: 초기 상태 (qpos + qvel 형태)
            
        Returns:
            obs (dict): 관측 {'img': (C,H,W), 'state': (proprio_dim,)}
            state (np.ndarray): 전체 상태 (qpos + qvel)
        """
        # 시드 설정
        if hasattr(self, 'seed') and callable(self.seed):
            self.seed(seed)
        
        # robosuite 환경 리셋
        self.env.reset()
        
        # 주어진 상태로 설정 (init_state=None이면 기본 reset 상태 사용)
        if init_state is not None:
            self._apply_state(init_state)
        
        # 최신 관측 반영
        self.raw_obs = self._force_obs()
        self._ensure_spaces(self.raw_obs)
        
        obs = self.get_obs()
        state = self.get_state()
        
        return obs, state

    def step_multiple(self, actions):
        """
        여러 액션을 순차적으로 실행합니다.
        
        Args:
            actions: (T, action_dim) 형태의 액션 시퀀스
            
        Returns:
            obses (dict): 각 키가 (T, ...) 형태의 관측 딕셔너리
            rewards (np.ndarray): (T,) 형태의 보상 배열
            dones (np.ndarray): (T,) 형태의 종료 플래그 배열
            infos (dict): 각 키가 (T, ...) 형태의 정보 딕셔너리
        """
        # 0-step 가드: 빈 시퀀스 처리
        if actions is None or len(actions) == 0:
            cur_obs = self.get_obs()
            cur_state = self.get_state()
            # (0, ...) 길이의 빈 배열 생성
            empty_obs = {
                k: np.empty((0,) + np.asarray(v).shape, dtype=np.asarray(v).dtype)
                for k, v in cur_obs.items()
            }
            empty_rewards = np.empty((0,), dtype=np.float32)
            empty_dones = np.empty((0,), dtype=bool)
            empty_infos = {
                "state": np.empty((0, np.asarray(cur_state).shape[0]), dtype=np.asarray(cur_state).dtype)
            }
            return empty_obs, empty_rewards, empty_dones, empty_infos

        obses = []
        rewards = []
        dones = []
        infos = []
        
        for action in actions:
            obs, reward, done, info = self.step(action)
            obses.append(obs)
            rewards.append(reward)
            dones.append(done)
            # state를 info에 추가 (rollout에서 사용)
            info['state'] = self.get_state()
            infos.append(info)
        
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        주어진 초기 상태에서 액션 시퀀스를 실행하고 결과를 반환합니다.
        
        Args:
            seed (int): 랜덤 시드
            init_state: 초기 상태 (qpos + qvel)
            actions: (T, action_dim) 형태의 액션 시퀀스
            
        Returns:
            obses (dict): 각 키가 (T+1, ...) 형태의 관측 딕셔너리
            states (np.ndarray): (T+1, state_dim) 형태의 상태 배열
            infos (dict): 각 step의 정보 (success 포함)
        """
        # 초기 상태로 리셋
        obs, state = self.prepare(seed, init_state)
        
        # 액션 시퀀스 실행
        obses, rewards, dones, infos = self.step_multiple(actions)
        
        # 초기 관측을 앞에 추가
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        
        # 초기 상태를 앞에 추가
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        
        # infos도 반환 (LIBERO task completion check 포함)
        return obses, states, infos

    def eval_state(self, goal_state, cur_state):
        """
        현재 상태와 목표 상태를 비교하여 성공 여부를 판단합니다.
        
        Args:
            goal_state: 목표 상태
            cur_state: 현재 상태
            
        Returns:
            dict: {'success': bool, 'state_dist': float}
        """
        goal_state = np.array(goal_state)
        cur_state = np.array(cur_state)
        
        # 상태 거리 계산
        state_dist = np.linalg.norm(goal_state - cur_state)
        
        # Libero의 경우 실제 task 성공을 체크하려면 환경의 _check_success를 사용
        # 하지만 상태 비교만으로는 판단하기 어려우므로 거리 기반으로 판단
        # (임계값은 태스크에 따라 조정 필요)
        success = state_dist < 0.1  # 임계값은 조정 가능
        
        return {
            'success': success,
            'state_dist': state_dist,
        }