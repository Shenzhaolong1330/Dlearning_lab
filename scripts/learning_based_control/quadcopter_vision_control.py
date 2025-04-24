"""
shenzhaolong 2025.4.24
This script demonstrates how to create a simple quadcopter environment 
-with an attitude controller
-with an acceleration controller (functional now 4.12)
-with a velocity controller (functional now 4.12)
-with a position controller (functional now 4.23)
-with a camera attached on the front
-with randomly generated obstacles

observation including:
    root_quat_w = obs["policy"]['root_quat_w']
    root_ang_vel_b = obs["policy"]['root_ang_vel_b']
    root_lin_vel_b = obs["policy"]['root_lin_vel_b']
    root_pos_w = obs["policy"]['root_pos_w']
    projected_gravity_b = obs["policy"]['projected_gravity_b']
    imu_lin_acc_b = obs["policy"]['imu_lin_acc_b']
    depth_image = obs["policy"]['camera_depth']

vision control:
    1. use the camera to detect the obstacle
    2. get depth image
    3. gussian blur the depth image
    4. culculate gradient of the depth image
    5. rotate the gradient to the world frame

.. code-block:: bash

    # Usage
    python scripts/learning_based_control/quadcopter_vision_control.py --enable_camera

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a quadcopter stabilization environment.")
parser.add_argument("--num_envs", type=int, default = 2, help="Number of environments to spawn.")
parser.add_argument("--env_spacing", type=float, default = 10.0, help="Space between each environment.")
parser.add_argument("--num_obstacles", type=int, default = 32, help="Number of obstacles to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from torchvision.transforms.functional import gaussian_blur

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import CameraCfg, ImuCfg, save_images_to_file, FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import cv2
##
# Pre-defined configs
##
from isaaclab_assets.robots.quadcopter import CRAZYFLIE_CFG# isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from quadcopter_controller import QuadcopterController

class QuadcopterActionTerm(ActionTerm):
    """Action term for stablizing the quadcopter."""
    
    _asset: Articulation

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 4, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 4, device=env.device)
        self._thrust = torch.zeros(env.num_envs, 1, 3, device=env.device)
        self._moment = torch.zeros(env.num_envs, 1, 3, device=env.device)
        self.thrust_to_weight = cfg.thrust_to_weight
        self.moment_scale = cfg.moment_scale

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # no-processing of actions
        # self._processed_actions = self._raw_actions.clone().clamp(-1.0, 1.0) # 神经网络action -1 ~ 1
        self._processed_actions = self._raw_actions.clone()

    def apply_actions(self):
        # 输入的actions是总推力+三轴力矩
        mass = self._asset.root_physx_view.get_masses()[0].sum()
        g = 9.81

        # 计算神经网络控制器输出到推力和力矩
        # self._thrust[:, 0, 2] = self.thrust_to_weight * mass * g * (self._processed_actions[:, 0] + 1.0) / 2.0
        # self._moment[:, 0, :] = self.moment_scale * self._processed_actions[:, 1:]
        
        # 直接输出力 N和力矩 Nm给动力学系统
        self._thrust[:, 0, 2] = self._processed_actions[:, 0]
        self._moment[:, 0, :] = self._processed_actions[:, 1:]
        body_id = self._asset.find_bodies("body")[0]
        self._asset.set_external_force_and_torque(self._thrust, self._moment, body_id)


@configclass
class QuadcopterActionTermCfg(ActionTermCfg):
    """Configuration for the quadcopter action term."""
    class_type: type = QuadcopterActionTerm
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01


def generate_random_position(index: int, region: dict = {'x':10.0, 'y':10.0, 'z':0.0}, min_spacing=1.0):
    positions = []
    while len(positions) <= index:
        x = random.uniform(-region['x'], region['x'])
        y = random.uniform(-region['y'], region['y'])
        z = random.uniform(0.0, region['z'])
        # 确保与已有障碍物的间隔足够大
        if all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 > min_spacing for px, py, _ in positions):
            positions.append((x, y, z))
        positions.append((x, y, z))
    return positions[index]


@configclass
class QuadcopterSceneCfg(InteractiveSceneCfg):
    
    # terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/front_cam",
        update_period=0.01,
        height=720,
        width=1280,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.03, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body", # imu的prim path只用绑定在目标地址,不用绑定在传感器地址
        gravity_bias=(0, 0, 0), 
        debug_vis=True
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    obstacles = RigidObjectCollectionCfg(
        rigid_objects={
            f"obstacle_{i}": RigidObjectCfg(
                # prim_path=f"/World/Obstacles/Obstacle_{i}",
                prim_path=f"/World/envs/env_.*/Obstacle_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(random.uniform(0.2, 0.3), random.uniform(0.2, 0.3), random.uniform(3.0, 4.0)), 
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(random.random(), random.random(), random.random()),  
                        metallic=0.2,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, 
                        solver_velocity_iteration_count=0,
                        kinematic_enabled = True,
                        disable_gravity = True,
                        rigid_body_enabled = True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    # DONE: add mass, rigid and collision properties when evaluate the algorithm
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=generate_random_position(i, min_spacing=1.5), 
                ),
            )
            for i in range(args_cli.num_obstacles)
        }
    )


@configclass
class ActionsCfg:
    """Action specifications for the quadcopter environment."""

    external_force_and_torque = QuadcopterActionTermCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    """Observation specifications for the quadcopter environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        root_lin_vel_b = ObsTerm(func=mdp.base_lin_vel)
        root_ang_vel_b = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity_b = ObsTerm(func=mdp.projected_gravity)
        # DONE: desired_pos_b 放在观测类的外面计算
        root_pos_w = ObsTerm(func=mdp.root_pos_w)
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        imu_lin_acc_b = ObsTerm(func=mdp.imu_lin_acc)

        camera_rgb = ObsTerm(
            func=lambda env, asset_cfg: env.scene[asset_cfg.name].data.output["rgb"],
            params={"asset_cfg": SceneEntityCfg("camera")},
        )
        camera_depth = ObsTerm(
            func=lambda env, asset_cfg: env.scene[asset_cfg.name].data.output["depth"],
            params={"asset_cfg": SceneEntityCfg("camera")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_drone = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-15.0,-16.0),
                "y": (-2.0,2.0),
                "z": (0.5,1.0),
                "roll": (0.0,0.0), 
                "pitch": (0.0,0.0), 
                "yaw": (0.0,0.0),
                },
            "velocity_range": {
                # "x": (-1.0,1.0),
                # "y": (-1.0,1.0),
                # "z": (-1.0,1.0)
                "x": (0.0,0.0),
                "y": (0.0,0.0),
                "z": (0.0,0.0)
                },
        },
    )


@configclass
class QuadcopterEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the quadcopter environment."""

    # Scene settings
    scene = QuadcopterSceneCfg(num_envs=1024, env_spacing=0.0)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (6.0, 4.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 2.0)
        # step settings
        self.decimation = 2  # env step every 2 sim steps: 100Hz / 2 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 10ms: 100Hz



class QuadcopterEnv(ManagerBasedEnv):
    """Quadcopter environment."""
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg):
        super().__init__(cfg)
        # self.desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # self.desired_pos_w[:, 2] = 1.0 



def plot(data_log, plot_type='thrust'):
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'thrust':
        plt.plot(data_log['time'], data_log['thrust'], label='Thrust')
        plt.title('Thrust Output')
        plt.ylabel('Thrust (N)')
        
    elif plot_type == 'torque':
        torque = np.array(data_log['torque'])
        plt.plot(data_log['time'], torque[:, 0], label='X Torque')
        plt.plot(data_log['time'], torque[:, 1], label='Y Torque') 
        plt.plot(data_log['time'], torque[:, 2], label='Z Torque')
        plt.title('Torque Output')
        plt.ylabel('Torque (Nm)')
        
    elif plot_type == 'attitude':
        rpy = np.array(data_log['attitude'])
        desired_att = np.array(data_log['desired_att'])
        plt.plot(data_log['time'], rpy[:, 0], label='Roll')
        plt.plot(data_log['time'], desired_att[:, 0], '--', label='Desired Roll')
        plt.plot(data_log['time'], rpy[:, 1], label='Pitch')
        plt.plot(data_log['time'], desired_att[:, 1], '--', label='Desired Pitch')
        plt.plot(data_log['time'], rpy[:, 2], label='Yaw')
        plt.plot(data_log['time'], desired_att[:, 2], '--', label='Desired Yaw')
        plt.title('Attitude Tracking')
        plt.ylabel('Attitude (rad)')
        
    elif plot_type == 'position':
        pos = np.array(data_log['position'])
        desired_pos = np.array(data_log['desired_pos'])
        plt.plot(data_log['time'], pos[:, 0], label='X Position')
        plt.plot(data_log['time'], desired_pos[:, 0], '--', label='Desired X')
        plt.plot(data_log['time'], pos[:, 1], label='Y Position')
        plt.plot(data_log['time'], desired_pos[:, 1], '--', label='Desired Y')
        plt.plot(data_log['time'], pos[:, 2], label='Z Position')
        plt.plot(data_log['time'], desired_pos[:, 2], '--', label='Desired Z')
        plt.title('Position Tracking')
        plt.ylabel('Position (m)')
        
    elif plot_type == 'velocity':
        vel = np.array(data_log['velocity'])
        desired_vel = np.array(data_log['desired_vel'])
        plt.plot(data_log['time'], vel[:, 0], label='Actual X')
        plt.plot(data_log['time'], desired_vel[:, 0], '--', label='Desired X')
        plt.plot(data_log['time'], vel[:, 1], label='Actual Y')
        plt.plot(data_log['time'], desired_vel[:, 1], '--', label='Desired Y')
        plt.plot(data_log['time'], vel[:, 2], label='Actual Z') 
        plt.plot(data_log['time'], desired_vel[:, 2], '--', label='Desired Z')
        plt.title('Velocity Tracking')
        plt.ylabel('Velocity (m/s)')
        
    elif plot_type == 'acceleration':
        acc = np.array(data_log['acceleration'])
        desired_acc = np.array(data_log['desired_acc'])
        plt.plot(data_log['time'], acc[:, 0], label='Actual X')
        plt.plot(data_log['time'], desired_acc[:, 0], '--', label='Desired X')
        plt.plot(data_log['time'], acc[:, 1], label='Actual Y')
        plt.plot(data_log['time'], desired_acc[:, 1], '--', label='Desired Y')
        plt.plot(data_log['time'], acc[:, 2], label='Actual Z')
        plt.plot(data_log['time'], desired_acc[:, 2], '--', label='Desired Z')
        plt.title('Acceleration Tracking')
        plt.ylabel('Acceleration (m/s²)')
        
    else:
        raise ValueError("Unsupported plot type. Valid options: 'thrust', 'torque', 'attitude', 'position', 'velocity', 'acceleration'")

    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



def main():
    """Main function."""
    # parse the arguments
    env_cfg = QuadcopterEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing

    # setup base environment
    env = QuadcopterEnv(cfg=env_cfg)
    robot_mass =env.scene.__getitem__("robot").root_physx_view.get_masses()[0].sum()
    dt=env_cfg.sim.dt
    num_envs=env_cfg.scene.num_envs
    print("-"*10,"config","-"*10)
    print('robot_mass',robot_mass,'kg')
    print('num_envs',num_envs)
    print('dt',dt,'s')

    # create a folder to save images
    depth_image_dir = "scripts/depth_images"
    os.makedirs(depth_image_dir, exist_ok=True)

    # initialize pid controller
    controller = QuadcopterController(
        num_envs=num_envs,
        mass = robot_mass,
        gravity=9.81,
        dt=env_cfg.sim.dt,
        device=env.device
    )

    # 在初始化控制器后添加数据记录容器
    data_log = {
        'time': [],
        'thrust': [],
        'torque': [],
        'attitude': [],
        'position': [],
        'velocity': [],
        'acceleration': [],
        'desired_acc': [],
        'desired_vel': [],
        'desired_att': [],
        'desired_pos': [],
    }

    # initialize control target
    desired_pos=torch.zeros(env_cfg.scene.num_envs, 3, device=env.device)
    desired_pos[:, 0] = -11.0
    desired_pos[:, 2] = 2.0
    desired_vel=torch.zeros(env_cfg.scene.num_envs, 3, device=env.device)
    desired_vel[:, 0] = 1.0
    # desired_vel[:, 1] = 2.0
    desired_vel[:, 2] = 0.1
    desired_acc=torch.zeros(env_cfg.scene.num_envs, 3, device=env.device)
    desired_acc[:, 0] = 1.0
    desired_acc[:, 1] = 1.0
    desired_acc[:, 2] = 5.0
    desired_attitude=torch.zeros(env_cfg.scene.num_envs, 3, device=env.device)
    desired_attitude[:, 1] = 0.5
    desired_attitude[:, 1] = 0.01
    desired_attitude[:, 2] = -0.4

    # simulate physics
    count = 0
    obs, _ = env.reset()
    controller.reset_controller()
    while count < 300:

        root_quat_w = obs["policy"]['root_quat_w']
        root_ang_vel_b = obs["policy"]['root_ang_vel_b']
        root_lin_vel_b = obs["policy"]['root_lin_vel_b']
        root_pos_w = obs["policy"]['root_pos_w']
        projected_gravity_b = obs["policy"]['projected_gravity_b']
        imu_lin_acc_b = obs["policy"]['imu_lin_acc_b']
        depth_image = obs["policy"]['camera_depth']

        # preprocess depth image
        for env_idx in range(depth_image.shape[0]):
            env_depth = depth_image[env_idx]
            if torch.any(~torch.isinf(env_depth)):
                max_valid_depth = torch.max(env_depth[~torch.isinf(env_depth)])
            else:
                max_valid_depth = torch.tensor(1.0, device=env_depth.device)
            env_depth[torch.isinf(env_depth)] = max_valid_depth
            env_depth_normalized = (env_depth - env_depth.min()) / (env_depth.max() - env_depth.min())
            depth_image[env_idx] = env_depth_normalized
            
        # gaussian blur depth image
        for env_idx in range(depth_image.shape[0]):
            env_depth = depth_image[env_idx]
            env_depth = env_depth.permute(2, 0, 1)
            env_depth_smoothed = gaussian_blur(env_depth, kernel_size=[51, 51], sigma=[1000.0, 1000.0])
            env_depth_smoothed = env_depth_smoothed.permute(1, 2, 0)
            depth_image[env_idx] = env_depth_smoothed

        sensor_data = {
            'root_quat_w': root_quat_w,
            'base_ang_vel': root_ang_vel_b,
            'base_lin_vel': root_lin_vel_b,
            'projected_gravity_b': projected_gravity_b,
            'root_pos_w': root_pos_w,
            'imu_lin_acc_b': imu_lin_acc_b,
            'camera_depth': depth_image
        }
        depth_image_path = os.path.join(depth_image_dir, f"depth_{count:04d}.png")
        test_depth_image = np.squeeze(depth_image[0].cpu().numpy())
        plt.imsave(depth_image_path, test_depth_image, cmap="viridis")
        print(f"Saved depth image to {depth_image_path}")

        # heading to 0.0.0
        # desired_attitude[:, 2] = torch.atan2(-root_pos_w[:, 1], -root_pos_w[:, 0])
        if count < 20:
            # test attitude_controller
            thrust, torque = controller.attitude_controller(sensor_data=sensor_data, desired_attitude=torch.zeros(env_cfg.scene.num_envs, 3, device=env.device))
            desired_attitude_controller = torch.zeros(env_cfg.scene.num_envs, 3, device=env.device)
            desired_acceleration_controller = torch.zeros(env_cfg.scene.num_envs, 3, device=env.device)
        else:
            # test attitude_controller
            # thrust, torque = controller.attitude_controller(sensor_data=sensor_data, desired_attitude=desired_attitude)
            # desired_attitude_controller = desired_attitude.clone()

            # test acceleration_controller
            # thrust, torque, desired_attitude_controller = controller.acceleration_controller(sensor_data=sensor_data, desired_acc=desired_acc, desired_yaw=desired_attitude[:, 2])

            # test velocity_controller
            # thrust, torque, desired_attitude_controller, desired_acceleration_controller = controller.velocity_controller(sensor_data=sensor_data, desired_vel=desired_vel, desired_yaw=desired_attitude[:, 2])

            # test position_controller
            thrust, torque, desired_attitude_controller, desired_acceleration_controller = controller.position_controller(sensor_data=sensor_data, desired_pos=desired_pos, desired_yaw=desired_attitude[:, 2])
            
        # 打印当前状态
        print('-'*20)
        print('力矩torque:', np.round(torque[0,:].cpu().numpy(), 5)) 
        print('推力thrust:', np.round(thrust[0].cpu().numpy(), 5)) 
        # test attitude_controlle
        print('姿态角_w:', np.round(controller.quat_to_euler(root_quat_w)[0,:].cpu().numpy(), 5))
        print('位置_w:', np.round(root_pos_w[0,:].cpu().numpy(), 5))
        rotation_matrix = controller.euler_to_rotation_matrix(controller.quat_to_euler(root_quat_w))
        print('加速度_b:', np.round(imu_lin_acc_b[0,:].cpu().numpy(), 5))
        print('加速度_w:', np.round(torch.bmm(rotation_matrix, imu_lin_acc_b.unsqueeze(-1)).squeeze(-1)[0,:].cpu().numpy(), 5))
        print('速度_b:', np.round(root_lin_vel_b[0,:].cpu().numpy(), 5))
        print('速度_w:', np.round(torch.bmm(rotation_matrix, root_lin_vel_b.unsqueeze(-1)).squeeze(-1)[0,:].cpu().numpy(), 5)) 
        print('-'*20)
        actions = torch.cat([thrust, torque], dim=1)
        obs, _ = env.step(actions)
        count += 1

        # 添加数据记录（示例记录第一个环境的）
        current_time = count * env_cfg.sim.dt
        data_log['time'].append(current_time)
        data_log['thrust'].append(thrust[0].cpu().numpy())
        data_log['torque'].append(torque[0].cpu().numpy())
        data_log['attitude'].append(controller.quat_to_euler(root_quat_w)[0].cpu().numpy())
        data_log['position'].append(root_pos_w[0].cpu().numpy())
        data_log['velocity'].append(torch.bmm(rotation_matrix, root_lin_vel_b.unsqueeze(-1)).squeeze(-1)[0].cpu().numpy())
        data_log['acceleration'].append(torch.bmm(rotation_matrix, imu_lin_acc_b.unsqueeze(-1)).squeeze(-1)[0].cpu().numpy())
        data_log['desired_acc'].append(desired_acceleration_controller[0].cpu().numpy())
        data_log['desired_vel'].append(desired_vel[0].cpu().numpy())
        data_log['desired_att'].append(desired_attitude_controller[0].cpu().numpy())
        data_log['desired_pos'].append(desired_pos[0].cpu().numpy())

    env.close()
    return data_log

if __name__ == "__main__":
    data_log = main()
    plot(data_log,'position')
    plot(data_log,'velocity')
    plot(data_log,'acceleration')
    plot(data_log,'attitude')
    simulation_app.close()