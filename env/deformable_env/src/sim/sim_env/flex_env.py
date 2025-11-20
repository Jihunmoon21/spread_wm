import os
import numpy as np
import pyflex
import gym
from gym import spaces
import math
from scipy.spatial.distance import cdist

import pybullet as p
import pybullet_data

from .robot_env import FlexRobotHelper

pyflex.loadURDF = FlexRobotHelper.loadURDF
pyflex.resetJointState = FlexRobotHelper.resetJointState
pyflex.getRobotShapeStates = FlexRobotHelper.getRobotShapeStates

from .flex_scene import FlexScene
from .cameras import Camera
from ..utils import fps_with_idx, quatFromAxisAngle, find_min_distance, rand_float

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../../../../"))

TABLE_COLOR_MAP = {
    "default": np.ones(3, dtype=np.float32) * (160.0 / 255.0),
    "brown": np.array([0.6, 0.4, 0.2], dtype=np.float32),
    "purple": np.array([0.75, 0.6, 0.95], dtype=np.float32),
}
DEFAULT_TABLE_COLOR_NAME = "brown"

class FlexEnv(gym.Env):
    def __init__(self, config=None) -> None:
        super().__init__()

        self.dataset_config = config["dataset"]
        raw_table_color = self.dataset_config.get("table_color", DEFAULT_TABLE_COLOR_NAME)
        self.table_color = str(raw_table_color).lower()
        if self.table_color not in TABLE_COLOR_MAP:
            print(f"[WARN] Unknown table_color '{self.table_color}', falling back to '{DEFAULT_TABLE_COLOR_NAME}'.")
            self.table_color = DEFAULT_TABLE_COLOR_NAME
        # Flag to track if set_states has been called for the first time
        self._first_set_states = True

        # env component
        self.obj = self.dataset_config["obj"]
        self.obj_params = self.dataset_config["obj_params"]
        self.scene = FlexScene()

        # set up pybullet
        physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")

        # set up robot arm
        # xarm6
        self.flex_robot_helper = FlexRobotHelper()
        self.end_idx = self.dataset_config["robot_end_idx"]
        self.num_dofs = self.dataset_config["robot_num_dofs"]
        self.robot_speed_inv = self.dataset_config["robot_speed_inv"]

        # set up pyflex
        self.screenWidth = self.dataset_config["screenWidth"]
        self.screenHeight = self.dataset_config["screenHeight"]
        self.camera = Camera(self.screenWidth, self.screenHeight)

        # Handle different PyFleX bindings: some expose set_screenWidth/Height, others set_screen_size
        if hasattr(pyflex, "set_screenWidth") and hasattr(pyflex, "set_screenHeight"):
            pyflex.set_screenWidth(self.screenWidth)
            pyflex.set_screenHeight(self.screenHeight)
        elif hasattr(pyflex, "set_screen_size"):
            pyflex.set_screen_size(self.screenWidth, self.screenHeight)
        else:
            # Fallback: proceed without explicit screen size setup
            pass
        # Light settings (optional in some bindings)
        if hasattr(pyflex, "set_light_dir"):
            pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
        if hasattr(pyflex, "set_light_fov"):
            pyflex.set_light_fov(70.0)
        # Ensure previous PyFlex context is cleaned before re-initializing (important when creating multiple envs sequentially)
        if hasattr(pyflex, "is_initialized") and pyflex.is_initialized():
            try:
                pyflex.clean()
            except Exception:
                pass

        # Initialize PyFleX (new version requires 4 args: headless, render, width, height)
        headless = bool(self.dataset_config["headless"])
        # Ensure EGL is used in headless mode
        if headless:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            # Optionally select device 0; harmless if ignored
            os.environ.setdefault("EGL_DEVICE_ID", "0")
        # Enable EGL offscreen rendering even in headless mode
        # Many clusters support EGL; set render=True to request offscreen context
        render = True
        try:
            pyflex.init(headless, render, int(self.screenWidth), int(self.screenHeight))
        except TypeError:
            # fallback to old signature
            try:
                pyflex.init(headless)
            except TypeError:
                # another fallback
                pyflex.init()
        
        # Critical: Allow PyFlex to fully initialize before setting scene
        # This is especially important for newer GPU architectures
        # Try a dummy step to ensure GPU context is ready
        try:
            # This may fail if no scene is set, but it forces GPU initialization
            pass  # Skip dummy step - will be done after set_scene
        except:
            pass

        # set up camera
        self.camera_view = self.dataset_config["camera_view"]

        # define action space
        self.action_dim = self.dataset_config["action_dim"]
        action_space_value = self.dataset_config["action_space"]  # This is the range value
        # Create proper gym Box action space
        self.action_space = spaces.Box(
            low=-action_space_value,
            high=action_space_value,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        
        # define observation space (RGB + depth channels)
        # Observation is rendered image: (H, W, 5) where 5 = RGB + depth_x + depth_y
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screenHeight, self.screenWidth, 5),
            dtype=np.uint8
        )

        # stat
        self.count = 0
        self.imgs_list = []
        self.particle_pos_list = []
        self.eef_states_list = []

        # Limit recentering: allow only first three calls per planning session
        # This counter persists across resets
        if not hasattr(self, "_recentering_calls"):
            self._recentering_calls = 0
        # Global guard across processes/instances within same interpreter
        if not hasattr(FlexEnv, "_global_recentering_calls"):
            FlexEnv._global_recentering_calls = 0
        # 전역 카운터: 전체 planning 세션에서 초기 상태 설정 시 최대 3번까지 재센터링
        if not hasattr(FlexEnv, "_global_first_set_states_count"):
            FlexEnv._global_first_set_states_count = 0
        # Flag to allow unlimited recentering when forced externally
        self._force_recenter_unlimited = False

        self.fps = self.dataset_config["fps"]
        self.fps_number = self.dataset_config["fps_number"]

        # others
        self.gripper = self.dataset_config["gripper"]
        self.stick_len = self.dataset_config["pusher_len"]
        # Disable rendering until environment verification completes
        self.disable_render = True
    
        self.scene.set_scene(self.obj, self.obj_params)
        
        # For granular scenes, add extra verification and gentle start
        if self.obj == "granular":
            try:
                pyflex.get_positions()
            except Exception:
                pass
            
            # Add a small delay to ensure GPU context is fully ready
            import time as time_module
            time_module.sleep(0.1)
            
            # Start with very gentle steps for granular scenes
            # First few steps with minimal updates to let particles settle
            for i in range(5):
                try:
                    pyflex.step(update_params=None, capture=0, path=None, render=0)
                except Exception as e:
                    print(f"Error during gentle step {i+1}: {e}")
                    # If gentle steps fail, try to continue with normal steps
                    break
            
            # Continue with normal stabilization steps
            stabilize_steps = 30  # Increased from 20
            for i in range(stabilize_steps):
                try:
                    pyflex.step()
                except Exception as e:
                    print(f"Error during stabilization step {i+1}: {e}")
                    raise
        else:
            # Normal stabilization for non-granular scenes
            stabilize_steps = 10
            for _ in range(stabilize_steps):
                pyflex.step()
        
        # set camera
        try:
            self.camera.set_init_camera(self.camera_view)
        except Exception as e:
            print(f"Error setting camera: {e}")
            raise
        
        save_data = False # disable initial rendering to avoid crashes during verification
        if save_data:
            (
                self.camPos_list,
                self.camAngle_list,
                self.cam_intrinsic_params,
                self.cam_extrinsic_matrix,
            ) = self.camera.init_multiview_cameras()
        
        # add table - extra caution for granular scenes
        try:
            self.add_table()
        except Exception as e:
            print(f"Error adding table: {e}")
            raise
        
        # Stabilize after adding table - more steps for granular
        stabilize_after_table = 10 if self.obj == "granular" else 5
        for i in range(stabilize_after_table):
            try:
                pyflex.step()
            except Exception as e:
                print(f"Error during post-table stabilization step {i+1}: {e}")
                raise
        
        ## add robot
        try:
            # Additional stabilization before adding robot meshes
            if self.obj == "granular":
                for _ in range(5):
                    pyflex.step()
            
            self.add_robot()
        except Exception as e:
            print(f"Error adding robot: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Stabilize after adding robot (critical for preventing segfaults)
        # Robot mesh loading can cause issues if PyFlex isn't ready
        stabilize_after_robot = 15 if self.obj == "granular" else 10
        for step_idx in range(stabilize_after_robot):
            try:
                pyflex.step()
            except Exception as e:
                print(f"Error during post-robot stabilization step {step_idx + 1}: {e}")
                raise
        
        # Final stabilization before enabling rendering - more for granular
        final_stabilize_steps = 30 if self.obj == "granular" else 20
        for step_idx in range(final_stabilize_steps):
            try:
                pyflex.step()
            except Exception as e:
                print(f"Error during final stabilization step {step_idx + 1}: {e}")
                raise
        
        # Ensure camera is properly set
        try:
            self.camera.set_init_camera(self.camera_view)
        except Exception as e:
            print(f"Error re-setting camera: {e}")
            raise
        
        # Re-enable rendering now that initialization is complete
        self.disable_render = False
        
        # Additional stabilization after enabling rendering - more for granular
        post_render_steps = 15 if self.obj == "granular" else 10
        for step_idx in range(post_render_steps):
            try:
                pyflex.step()
            except Exception as e:
                print(f"Error during post-render stabilization step {step_idx + 1}: {e}")
                raise
        
        # Verify rendering works
        headless = bool(self.dataset_config["headless"])
        try:
            # Check particle positions to ensure scene has content
            try:
                particle_pos = pyflex.get_positions()
                num_particles = len(particle_pos) // 4
                if num_particles > 0:
                    pos_reshaped = particle_pos.reshape(-1, 4)
            except Exception as e:
                print(f"Warning: Could not get particle positions: {e}")
            
            # Force camera update
            pyflex.set_camPos(self.camera.camPos)
            pyflex.set_camAngle(self.camera.camAngle)
            
            # Multiple render attempts to ensure OpenGL context is ready
            for render_attempt in range(3):
                # Additional steps before rendering
                for _ in range(5):
                    pyflex.step()
                
                # Test render
                test_img = self.render()
                if test_img is not None and test_img.shape[0] > 0:
                    # Check if image is all zeros (black screen)
                    img_mean = test_img.mean()
                    img_std = test_img.std()
                    img_max = test_img.max()
                    img_min = test_img.min()
                    
                    if np.allclose(test_img, 0):
                        print("Warning: Rendering returned black screen - check EGL/OpenGL setup")
                        if render_attempt < 2:
                            continue
                    else:
                        break
                else:
                    print(f"Warning: Render attempt {render_attempt + 1} returned invalid image")
        except Exception as e:
            print(f"Warning: Rendering test failed: {e}")
            if not headless:
                import traceback
                traceback.print_exc()

    ### shape states
    def robot_to_shape_states(self, robot_states):
        n_robot_links = robot_states.shape[0]
        n_table = self.table_shape_states.shape[0]

        shape_states = np.zeros((n_table + n_robot_links, 14))
        shape_states[:n_table] = self.table_shape_states  # set shape states for table
        shape_states[n_table:] = robot_states  # set shape states for robot

        return shape_states

    def reset_robot(self, jointPositions=np.zeros(13).tolist()):
        index = 0
        for j in range(7):
            p.changeDynamics(self.robotId, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.robotId, j)

            jointType = info[2]
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                pyflex.resetJointState(self.flex_robot_helper, j, jointPositions[index])
                index = index + 1

        pyflex.set_shape_states(
            self.robot_to_shape_states(
                pyflex.getRobotShapeStates(self.flex_robot_helper)
            )
        )

    def add_table(self):
        ## add table board
        self.table_shape_states = np.zeros((2, 14))
        # table for workspace
        self.wkspace_height = 0.5
        self.wkspace_width = 3.5  # 3.5*2=7 grid = 700mm
        self.wkspace_length = 4.5  # 4.5*2=9 grid = 900mm
        halfEdge = np.array(
            [self.wkspace_width, self.wkspace_height, self.wkspace_length]
        )
        center = np.array([0.0, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0.0, 1.0, 0.0]), angle=0.0)
        hideShape = 0
        table_color = TABLE_COLOR_MAP.get(self.table_color, TABLE_COLOR_MAP[DEFAULT_TABLE_COLOR_NAME])
        pyflex.add_box(halfEdge, center, quats, hideShape, table_color)
        self.table_shape_states[0] = np.concatenate([center, center, quats, quats])
        
        # table for robot
        if self.obj in ["cloth"]:
            robot_table_height = 0.5 + 1.0
        else:
            robot_table_height = 0.5 + 0.3
        robot_table_width = 126 / 200  # 126mm
        robot_table_length = 126 / 200  # 126mm
        halfEdge = np.array([robot_table_width, robot_table_height, robot_table_length])
        center = np.array([-self.wkspace_width - robot_table_width, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0.0, 1.0, 0.0]), angle=0.0)
        hideShape = 0
        pyflex.add_box(halfEdge, center, quats, hideShape, table_color)
        self.table_shape_states[1] = np.concatenate([center, center, quats, quats])
        
    def add_robot(self):
        if self.obj in ["granular"]:
            # flat board pusher
            robot_base_pos = [-self.wkspace_width - 0.6, 0.0, self.wkspace_height + 0.3]
            robot_base_orn = [0, 0, 0, 1]
            self.robotId = pyflex.loadURDF(
                self.flex_robot_helper,
                os.path.join(BASE_DIR, "env/deformable_env/src/sim/assets/xarm/xarm6_with_gripper_board.urdf"),
                robot_base_pos,
                robot_base_orn,
                globalScaling=10.0,
            )
            self.rest_joints = np.zeros(8)
        elif self.obj in ["rope"]:
            # stick pusher
            robot_base_pos = [-self.wkspace_width - 0.6, 0.0, self.wkspace_height + 0.3]
            robot_base_orn = [0, 0, 0, 1]
            self.robotId = pyflex.loadURDF(
                self.flex_robot_helper,
                os.path.join(BASE_DIR, "env/deformable_env/src/sim/assets/xarm/xarm6_with_gripper.urdf"),
                robot_base_pos,
                robot_base_orn,
                globalScaling=10.0,
            )
            self.rest_joints = np.zeros(8)
        elif self.obj in ["cloth"]:
            # gripper
            robot_base_pos = [-self.wkspace_width - 0.6, 0.0, self.wkspace_height + 1.0]
            robot_base_orn = [0, 0, 0, 1]
            self.robotId = pyflex.loadURDF(
                self.flex_robot_helper,
                os.path.join(BASE_DIR, "env/deformable_env/src/sim/assets/xarm/xarm6_with_gripper_grasp.urdf"),
                robot_base_pos,
                robot_base_orn,
                globalScaling=10.0,
            )
            self.rest_joints = np.zeros(13)

    def store_data(self, store_cam_param=False, init_fps=False):
        # Initialize camera lists if not already initialized
        if not hasattr(self, 'camPos_list') or self.camPos_list is None:
            (
                self.camPos_list,
                self.camAngle_list,
                self.cam_intrinsic_params,
                self.cam_extrinsic_matrix,
            ) = self.camera.init_multiview_cameras()
        
        saved_particles = False
        img_list = []
        for j in range(len(self.camPos_list)):
            pyflex.set_camPos(self.camPos_list[j])
            pyflex.set_camAngle(self.camAngle_list[j])

            if store_cam_param:
                # Ensure cam_intrinsic_params and cam_extrinsic_matrix are initialized
                if not hasattr(self, 'cam_intrinsic_params') or self.cam_intrinsic_params is None:
                    self.cam_intrinsic_params = {}
                    self.cam_extrinsic_matrix = {}
                self.cam_intrinsic_params[j], self.cam_extrinsic_matrix[j] = (
                    self.camera.get_cam_params()
                )

            # save images
            img = self.render()
            img_list.append(img)

            # save particles
            if not saved_particles:
                # save particle pos
                particles = self.get_positions().reshape(-1, 4)
                particles_pos = particles
                # particles_pos = particles[:, :3] #!!! this is changed
                if self.fps:
                    if init_fps:
                        _, self.sampled_idx = fps_with_idx(
                            particles_pos, self.fps_number
                        )
                    particles_pos = particles_pos[self.sampled_idx]
                self.particle_pos_list.append(particles_pos)
                # save eef pos
                robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper)
                if self.gripper:
                    eef_states = np.zeros((2, 14))
                    eef_states[0] = robot_shape_states[9]  # left finger
                    eef_states[1] = robot_shape_states[12]  # right finger
                else:
                    eef_states = np.zeros((1, 14))
                    eef_states[0] = robot_shape_states[-1]  # pusher
                self.eef_states_list.append(eef_states)

                saved_particles = True

        img_list_np = np.array(img_list)
        self.imgs_list.append(img_list_np)
        self.count += 1

    ### setup gripper
    def _set_pos(self, picker_pos, particle_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]  # picker_pos
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)

    def _reset_pos(self, particle_pos):
        pyflex.set_positions(particle_pos)

    def robot_close_gripper(self, close, jointPoses=None):
        for j in range(8, self.num_joints):
            pyflex.resetJointState(self.flex_robot_helper, j, close)
        pyflex.set_shape_states(
            self.robot_to_shape_states(
                pyflex.getRobotShapeStates(self.flex_robot_helper)
            )
        )

    def robot_open_gripper(self):
        for j in range(8, self.num_joints):
            pyflex.resetJointState(self.flex_robot_helper, j, 0.0)
    
    def reset_partial(self, save_data=False):
        pyflex.set_shape_states(
            self.robot_to_shape_states(
                pyflex.getRobotShapeStates(self.flex_robot_helper)
            )
        )

        ## update robot shape states
        for idx, joint in enumerate(self.rest_joints):
            pyflex.set_shape_states(
                self.robot_to_shape_states(
                    pyflex.resetJointState(self.flex_robot_helper, idx, joint)
                )
            )

        self.num_joints = p.getNumJoints(self.robotId)
        self.joints_lower = np.zeros(self.num_dofs)
        self.joints_upper = np.zeros(self.num_dofs)
        dof_idx = 0
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robotId, i)
            jointType = info[2]
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                self.joints_lower[dof_idx] = info[8]
                self.joints_upper[dof_idx] = info[9]
                dof_idx += 1
        self.reset_robot()

        # initial render
        for _ in range(10):
            pyflex.step()

        # save initial rendering
        if save_data:
            self.store_data(store_cam_param=True, init_fps=True)

        # output
        out_data = self.imgs_list, self.particle_pos_list, self.eef_states_list

        return out_data

    ### reset env
    def reset(self, save_data=False, force_recenter=False):
        # Reset the flag so that set_states will recenter on first call after reset
        self._first_set_states = True
        pyflex.set_shape_states(
            self.robot_to_shape_states(
                pyflex.getRobotShapeStates(self.flex_robot_helper)
            )
        )

        ## update robot shape states
        for idx, joint in enumerate(self.rest_joints):
            pyflex.set_shape_states(
                self.robot_to_shape_states(
                    pyflex.resetJointState(self.flex_robot_helper, idx, joint)
                )
            )

        self.num_joints = p.getNumJoints(self.robotId)
        self.joints_lower = np.zeros(self.num_dofs)
        self.joints_upper = np.zeros(self.num_dofs)
        dof_idx = 0
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robotId, i)
            jointType = info[2]
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                self.joints_lower[dof_idx] = info[8]
                self.joints_upper[dof_idx] = info[9]
                dof_idx += 1
        self.reset_robot()

        # initial render / stabilization
        for _ in range(50):
            pyflex.step()
        # 🔧 final output 계산 시에만 재센터링 (force_recenter 파라미터 또는 플래그 확인)
        # 🔧 모든 경우에 재센터링 적용
        self._recenter_particles_xz()

        # save initial rendering
        if save_data:
            self.store_data(store_cam_param=True, init_fps=True)

        # output
        out_data = self.imgs_list, self.particle_pos_list, self.eef_states_list

        return out_data

    def step(self, action, save_data=False, data=None):
        """
        action: [start_x, start_z, end_x, end_z]
        """
        self.count = 0
        self.imgs_list, self.particle_pos_list, self.eef_states_list = data

        # set up action
        h = 0.5 + self.stick_len
        s_2d = np.concatenate([action[:2], [h]])
        e_2d = np.concatenate([action[2:], [h]])

        # pusher angle depending on x-axis
        if (s_2d - e_2d)[0] == 0:
            pusher_angle = np.pi / 2
        else:
            pusher_angle = np.arctan((s_2d - e_2d)[1] / (s_2d - e_2d)[0])
        # robot orientation
        orn = np.array([0.0, np.pi, pusher_angle + np.pi / 2])

        # create way points
        if self.gripper:
            way_points = [
                s_2d + [0.0, 0.0, 0.5],
                s_2d,
                s_2d,
                e_2d + [0.0, 0.0, 0.5],
                e_2d,
            ]
        else:
            way_points = [s_2d + [0.0, 0.0, 0.2], s_2d, e_2d, e_2d + [0.0, 0.0, 0.2]]
        self.reset_robot(self.rest_joints)

        # set robot speed
        speed = 1.0 / self.robot_speed_inv

        # step
        for i_p in range(len(way_points) - 1):
            s = way_points[i_p]
            e = way_points[i_p + 1]
            steps = int(np.linalg.norm(e - s) / speed) + 1

            for i in range(steps):
                end_effector_pos = s + (e - s) * i / steps  # expected eef position
                end_effector_orn = p.getQuaternionFromEuler(orn)
                jointPoses = p.calculateInverseKinematics(
                    self.robotId,
                    self.end_idx,
                    end_effector_pos,
                    end_effector_orn,
                    self.joints_lower.tolist(),
                    self.joints_upper.tolist(),
                    (self.joints_upper - self.joints_lower).tolist(),
                    self.rest_joints,
                )
                # print('jointPoses:', jointPoses)
                self.reset_robot(jointPoses)
                pyflex.step()

                ## ================================================================
                ## gripper control
                if self.gripper and i_p >= 1:
                    grasp_thresd = 0.1
                    obj_pos = self.get_positions().reshape(-1, 4)[:, :3]
                    new_particle_pos = self.get_positions().reshape(-1, 4).copy()

                    ### grasping
                    if i_p == 1:
                        close = 0
                        start = 0
                        end = 0.7
                        close_steps = 50  # 500
                        finger_y = 0.5
                        for j in range(close_steps):
                            robot_shape_states = pyflex.getRobotShapeStates(
                                self.flex_robot_helper
                            )  # 9: left finger; 12: right finger
                            left_finger_pos, right_finger_pos = (
                                robot_shape_states[9][:3],
                                robot_shape_states[12][:3],
                            )
                            left_finger_pos[1], right_finger_pos[1] = (
                                left_finger_pos[1] - finger_y,
                                right_finger_pos[1] - finger_y,
                            )
                            new_finger_pos = (left_finger_pos + right_finger_pos) / 2

                            if j == 0:
                                # fine the k pick point
                                pick_k = 5
                                left_min_dist, left_pick_index = find_min_distance(
                                    left_finger_pos, obj_pos, pick_k
                                )
                                right_min_dist, right_pick_index = find_min_distance(
                                    right_finger_pos, obj_pos, pick_k
                                )
                                min_dist, pick_index = find_min_distance(
                                    new_finger_pos, obj_pos, pick_k
                                )
                                # save the original setting for restoring
                                pick_origin = new_particle_pos[pick_index]

                            if (
                                left_min_dist <= grasp_thresd
                                or right_min_dist <= grasp_thresd
                            ):
                                new_particle_pos[left_pick_index, :3] = left_finger_pos
                                new_particle_pos[left_pick_index, 3] = 0
                                new_particle_pos[right_pick_index, :3] = (
                                    right_finger_pos
                                )
                                new_particle_pos[right_pick_index, 3] = 0
                            self._set_pos(new_finger_pos, new_particle_pos)

                            # close the gripper
                            close += (end - start) / close_steps
                            self.robot_close_gripper(close)
                            pyflex.step()

                    # find finger positions
                    robot_shape_states = pyflex.getRobotShapeStates(
                        self.flex_robot_helper
                    )  # 9: left finger; 12: right finger
                    left_finger_pos, right_finger_pos = (
                        robot_shape_states[9][:3],
                        robot_shape_states[12][:3],
                    )
                    left_finger_pos[1], right_finger_pos[1] = (
                        left_finger_pos[1] - finger_y,
                        right_finger_pos[1] - finger_y,
                    )
                    new_finger_pos = (left_finger_pos + right_finger_pos) / 2
                    # connect pick pick point to the finger
                    new_particle_pos[pick_index, :3] = new_finger_pos
                    new_particle_pos[pick_index, 3] = 0
                    self._set_pos(new_finger_pos, new_particle_pos)

                    self.reset_robot(jointPoses)
                    pyflex.step()

                ## ================================================================

                # save img in each step
                obj_pos = self.get_positions().reshape(-1, 4)[:, [0, 2]]
                obj_pos[:, 1] *= -1
                robot_obj_dist = np.min(
                    cdist(end_effector_pos[:2].reshape(1, 2), obj_pos)
                )
                if save_data:
                    rob_obj_dist_thresh = self.dataset_config["rob_obj_dist_thresh"]
                    contact_interval = self.dataset_config["contact_interval"]
                    non_contact_interval = self.dataset_config["non_contact_interval"]
                    if (
                        robot_obj_dist < rob_obj_dist_thresh
                        and i % contact_interval == 0
                    ):  # robot-object contact
                        self.store_data()
                    elif i % non_contact_interval == 0:  # not contact
                        self.store_data()

                self.reset_robot()
                if math.isnan(self.get_positions().reshape(-1, 4)[:, 0].max()):
                    print("simulator exploded when action is", action)
                    return None

        # set up gripper
        if self.gripper:
            self.robot_open_gripper()
            # reset the mass for the pick points
            new_particle_pos[pick_index, 3] = pick_origin[:, 3]
            self._reset_pos(new_particle_pos)

        self.reset_robot()

        for i in range(200):
            pyflex.step()

        # save final rendering
        if save_data:
            self.store_data()

        obs = self.render()
        out_data = self.imgs_list, self.particle_pos_list, self.eef_states_list

        return obs, out_data

    def render(self, no_return=False):
        # Proceed with rendering both in headless (EGL) and non-headless modes
        headless = bool(self.dataset_config.get("headless", True))
        try:
            pyflex.step()
        except Exception as e:
            print(f"Error in pyflex.step(): {e}")
        if no_return:
            return
        # If rendering is disabled, return a dummy frame to keep the pipeline running
        if getattr(self, "disable_render", False):
            return np.zeros((self.screenHeight, self.screenWidth, 5), dtype=np.float32)
        
        # Ensure camera is set before rendering
        if hasattr(self.camera, 'camPos') and hasattr(self.camera, 'camAngle'):
            try:
                pyflex.set_camPos(self.camera.camPos)
                pyflex.set_camAngle(self.camera.camAngle)
            except Exception as e:
                # Silent failure is OK - camera may already be set
                pass
        
        # Render and reshape
        try:
            # Debug: Check particle positions before rendering
            if hasattr(self, '_render_debug_count'):
                self._render_debug_count += 1
            else:
                self._render_debug_count = 1
            
            # Try rendering - pyflex.render() returns a tuple (image, depth)
            
            render_result = None
            try:
                render_result = pyflex.render()
            except Exception as render_err:
                # In headless without proper EGL, try a fallback if available
                print(f"Render method failed: {render_err}")
                return np.zeros((self.screenHeight, self.screenWidth, 5), dtype=np.float32)
            
            # Handle tuple return (image, depth)
            if render_result is None:
                return np.zeros((self.screenHeight, self.screenWidth, 5), dtype=np.float32)
            
            # Unpack tuple if needed
            if isinstance(render_result, tuple):
                if len(render_result) >= 2:
                    image, depth = render_result[0], render_result[1]
                    # Combine into single array
                    if image is not None and depth is not None:
                        img_array = np.array(image)
                        depth_array = np.array(depth)
                        # Stack RGB + depth
                        render_result = np.concatenate([img_array, depth_array], axis=-1) if len(img_array.shape) == 3 else img_array
                    elif image is not None:
                        render_result = np.array(image)
                    else:
                        return np.zeros((self.screenHeight, self.screenWidth, 5), dtype=np.float32)
                else:
                    render_result = np.array(render_result[0]) if len(render_result) > 0 else None
            
            if render_result is None or (hasattr(render_result, 'size') and render_result.size == 0):
                # Return black screen if rendering fails
                return np.zeros((self.screenHeight, self.screenWidth, 5), dtype=np.float32)
            
            # Reshape result - handle different output formats and vertically flip output
            try:
                reshaped = render_result.reshape(self.screenHeight, self.screenWidth, -1)
                if reshaped.shape[2] < 5:
                    out = np.zeros((self.screenHeight, self.screenWidth, 5), dtype=np.float32)
                    out[:, :, :reshaped.shape[2]] = reshaped
                elif reshaped.shape[2] > 5:
                    out = reshaped[:, :, :5]
                else:
                    out = reshaped

                # Always flip vertically so that origin is at the top-left for image consumers
                out = np.flip(out, axis=0).copy()
                out = np.flip(out, axis=1).copy()
                return out
            except ValueError:
                # Reshape failed - return what we can and still flip vertically
                flat_size = self.screenHeight * self.screenWidth * 5
                if render_result.size >= flat_size:
                    out = render_result[:flat_size].reshape(self.screenHeight, self.screenWidth, 5)
                else:
                    out = np.zeros((self.screenHeight, self.screenWidth, 5), dtype=np.float32)
                    out.flat[:render_result.size] = render_result.flatten()
                out = np.flip(out, axis=0).copy()
                out = np.flip(out, axis=1).copy()
                return out
                    
        except Exception as e:
            print(f"Warning: Rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((self.screenHeight, self.screenWidth, 5), dtype=np.float32)

    def close(self):
        pyflex.clean()

    def sample_action(self, init=False, boundary_points=None, boundary=None):
        if self.obj in ["rope", "granular"]:
            action = self.sample_deform_actions()
            return action
        elif self.obj in ["cloth"]:
            action, boundary_points, boundary = self.sample_grasp_actions_corner(
                init, boundary_points, boundary
            )
            return action, boundary_points, boundary
        else:
            raise ValueError("action not defined")

    def sample_deform_actions(self):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1  # align with the coordinates
        num_points = positions.shape[0]
        pos_xz = positions[:, [0, 2]]

        pos_x, pos_z = positions[:, 0], positions[:, 2]
        center_x, center_z = np.median(pos_x), np.median(pos_z)
        chosen_points = []
        for idx, (x, z) in enumerate(zip(pos_x, pos_z)):
            if np.sqrt((x - center_x) ** 2 + (z - center_z) ** 2) < 2.0:
                chosen_points.append(idx)
        # print(f'chosen points {len(chosen_points)} out of {num_points}.')
        if len(chosen_points) == 0:
            print("no chosen points")
            chosen_points = np.arange(num_points)

        # random choose a start point which can not be overlapped with the object
        valid = False
        for _ in range(1000):
            startpoint_pos_origin = np.random.uniform(
                -self.action_space, self.action_space, size=(1, 2)
            )
            startpoint_pos = startpoint_pos_origin.copy()
            startpoint_pos = startpoint_pos.reshape(-1)

            # choose end points which is the expolation of the start point and obj point
            pickpoint = np.random.choice(chosen_points)
            obj_pos = positions[pickpoint, [0, 2]]
            slope = (obj_pos[1] - startpoint_pos[1]) / (obj_pos[0] - startpoint_pos[0])
            if obj_pos[0] < startpoint_pos[0]:
                # 1.0 for planning
                # (1.5, 2.0) for data collection
                x_end = obj_pos[0] - 1.0  # rand_float(1.5, 2.0)
            else:
                x_end = obj_pos[0] + 1.0  # rand_float(1.5, 2.0)
            y_end = slope * (x_end - startpoint_pos[0]) + startpoint_pos[1]
            endpoint_pos = np.array([x_end, y_end])
            if (
                obj_pos[0] != startpoint_pos[0]
                and np.abs(x_end) < 1.5
                and np.abs(y_end) < 1.5
                and np.min(cdist(startpoint_pos_origin, pos_xz)) > 0.2
            ):
                valid = True
                break

        if valid:
            action = np.concatenate(
                [startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0
            )
        else:
            action = None

        return action

    def sample_grasp_actions_corner(
        self, init=False, boundary_points=None, boundary=None
    ):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1
        particle_x, particle_y, particle_z = (
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
        )
        x_min, y_min, z_min = np.min(particle_x), np.min(particle_y), np.min(particle_z)
        x_max, y_max, z_max = np.max(particle_x), np.max(particle_y), np.max(particle_z)

        # choose the starting point at the boundary of the object
        if init:  # record boundary points
            boundary_points = []
            boundary = []
            for idx, point in enumerate(positions):
                if point[0] == x_max:
                    boundary_points.append(idx)
                    boundary.append(1)
                elif point[0] == x_min:
                    boundary_points.append(idx)
                    boundary.append(2)
                elif point[2] == z_max:
                    boundary_points.append(idx)
                    boundary.append(3)
                elif point[2] == z_min:
                    boundary_points.append(idx)
                    boundary.append(4)
        assert len(boundary_points) == len(boundary)

        # random pick a point as start point
        valid = False
        for _ in range(1000):
            pick_idx = np.random.choice(len(boundary_points))
            startpoint_pos = positions[boundary_points[pick_idx], [0, 2]]
            endpoint_pos = startpoint_pos.copy()
            # choose end points which is outside the obj
            move_distance = rand_float(1.0, 1.5)

            if boundary[pick_idx] == 1:
                endpoint_pos[0] += move_distance
            elif boundary[pick_idx] == 2:
                endpoint_pos[0] -= move_distance
            elif boundary[pick_idx] == 3:
                endpoint_pos[1] += move_distance
            elif boundary[pick_idx] == 4:
                endpoint_pos[1] -= move_distance

            if np.abs(endpoint_pos[0]) < 3.5 and np.abs(endpoint_pos[1]) < 2.5:
                valid = True
                break

        if valid:
            action = np.concatenate(
                [startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0
            )
        else:
            action = None

        return action, boundary_points, boundary

    def get_positions(self):
        return pyflex.get_positions()

    def set_positions(self, positions):
        pyflex.set_positions(positions)

    def _recenter_particles_xz(self):
        """Translate particles so that their xz center is at (0, 0)."""
        try:
            # 🔧 모든 경우에 재센터링 적용 (제한 제거)
            pos = pyflex.get_positions()
            if pos is None or len(pos) == 0:
                return
            pos4 = pos.reshape(-1, 4)
            x_min, x_max = pos4[:, 0].min(), pos4[:, 0].max()
            z_min, z_max = pos4[:, 2].min(), pos4[:, 2].max()
            cx = 0.5 * (x_min + x_max)
            cz = 0.5 * (z_min + z_max)
            if abs(cx) <= 1e-6 and abs(cz) <= 1e-6:
                return
            pos4[:, 0] -= cx
            pos4[:, 2] -= cz
            pyflex.set_positions(pos4.reshape(-1))
            # Count successful recentering executions (참고용, 제한 없음)
            if hasattr(self, "_recentering_calls"):
                self._recentering_calls += 1
            if hasattr(FlexEnv, "_global_recentering_calls"):
                FlexEnv._global_recentering_calls += 1
        except Exception as e:
            print(f"[DEBUG] Recentering failed: {e}")

    def reset_recentering_counters(self):
        """Reset recentering counters so that forced recentering can run again."""
        self._recentering_calls = 0
        FlexEnv._global_recentering_calls = 0
        self._force_recenter_unlimited = True


    def get_num_particles(self):
        return self.get_positions().reshape(-1, 4).shape[0]

    def get_property_params(self):
        return self.scene.get_property_params()

    def get_states(self):
        return self.get_positions().reshape(-1, 4)

    def set_states(self, states, force_recenter=False):
        if states is not None:
            # self.scene.set_scene(self.obj)
            pyflex.set_positions(states)
            # 🔧 모든 경우에 재센터링 적용
            self._recenter_particles_xz()
            # _first_set_states 플래그 리셋 (한 번만 실행되도록)
            if self._first_set_states:
                self._first_set_states = False

    def seed(self, seed):
        np.random.seed(seed)

    def get_one_view_img(self, cam_id=None):
        # Initialize camera lists if not already initialized
        if not hasattr(self, 'camPos_list') or self.camPos_list is None:
            (
                self.camPos_list,
                self.camAngle_list,
                self.cam_intrinsic_params,
                self.cam_extrinsic_matrix,
            ) = self.camera.init_multiview_cameras()
        
        cam_id = cam_id or self.camera_view
        pyflex.set_camPos(self.camPos_list[cam_id])
        pyflex.set_camAngle(self.camAngle_list[cam_id])
        return self.render()
