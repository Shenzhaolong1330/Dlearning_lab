"""
shenzhaolong 2025.3.27
This script demonstrates how to create a simple environment to evaluate the trained policy of stabilizing a quadcopter. 

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/learning_based_control/creat_quadcopter_env.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a quadcopter stabilization environment.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument("--env_spacing", type=float, default=10.0, help="Space between each environment.")
# parser.add_argument("--checkpoint", type=str, default='logs/rsl_rl/quadcopter_direct/2025-03-29_00-21-07/exported/policy.pt', help="Path to model checkpoint exported as jit.")
parser.add_argument("--num_obstacles", type=int, default= 4, help="Number of obstacles to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import io
import os
import torch
import omni
import torch.nn as nn
import torch.nn.functional as F
from rl_games.algos_torch import players

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
from isaaclab.sensors import CameraCfg, ImuCfg, save_images_to_file
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import random
##
# Pre-defined configs
##
from isaaclab_assets.robots.quadcopter import CRAZYFLIE_CFG# isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


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


def generate_random_position(index: int, min_spacing=0.5):
    positions = []
    while len(positions) <= index:
        x = random.uniform(-2.5, 2.5)
        y = random.uniform(-2.5, 2.5)
        z = random.uniform(0.0, 2.5)
        # 确保与已有障碍物的间隔足够大
        # if all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 > min_spacing for px, py, _ in positions):
            # positions.append((x, y, z))
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
                    size=(random.uniform(0.3, 0.5), random.uniform(0.3, 0.5), random.uniform(0.3, 0.5)), 
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

    reset_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (2.0,3.0),
                "y": (2.0,3.0),
                "z": (0.0,2.0),
                "roll": (-0.1,0.1), 
                "pitch": (-0.1,0.1), 
                "yaw": (-10.0,10.0),},
            "velocity_range": {
                "x": (0.0,0.0),
                "y": (0.0,0.0),
                "z": (0.0,0.0)},
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
        self.viewer.eye = (6.0, 0.0, 8.0)
        self.viewer.lookat = (0.0, 0.0, 2.0)
        # step settings
        self.decimation = 2  # env step every 2 sim steps: 100Hz / 2 = 50Hz
        # simulation settings
        self.sim.dt = 0.01  # sim step every 10ms: 100Hz



class QuadcopterEnv(ManagerBasedEnv):
    """Quadcopter environment."""
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg):
        super().__init__(cfg)
        # self.desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # self.desired_pos_w[:, 2] = 1.0



def get_desired_pos_b(robot_pos_w: torch.Tensor, robot_alt_w: torch.Tensor, desired_pos_w: torch.Tensor)-> torch.Tensor:
    desired_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_alt_w, desired_pos_w)
    return desired_pos_b



class QuadrotorController:
    def __init__(self, num_envs, mass, gravity=9.81, dt=0.01, device="cuda"):
        self.device = device
        self.num_envs = num_envs
        
        # ================= 物理参数 =================
        self.mass = mass
        self.gravity = gravity
        self.base_thrust = self.mass * self.gravity * torch.ones((num_envs,1), device=device)
        self.dt = dt
        
        # ================= 独立控制参数 =================
        # --- 位置控制 ---
        self.position_p_gain = torch.ones((num_envs, 3), device=device) * 0.5    # XYZ P增益[3](@ref)
        
        # --- 速度控制 ---
        self.velocity_p_gain = torch.ones((num_envs, 3), device=device) * 1.2     # P增益[3](@ref)
        self.velocity_i_gain = torch.ones((num_envs, 3), device=device) * 0.05    # I增益[3](@ref)
        self.velocity_integral = torch.zeros(num_envs, 3, device=device)
        self.velocity_integral_limit = 2.0
        
        # --- 加速度控制 ---
        self.acceleration_p_gain = torch.ones((num_envs, 2), device=device) * 0.15  # XY轴增益[6](@ref)
        self.max_thrust_ratio = 2.0
        
        # --- 姿态控制 ---
        self.attitude_p_gain = torch.tensor([[0.03, 0.03]], device=device).repeat(num_envs, 1)  # Roll/Pitch P[3](@ref)
        self.attitude_d_gain = torch.tensor([[0.0005, 0.0005]], device=device).repeat(num_envs, 1) # D项[3](@ref)
        self.yaw_rate_gain = torch.full((num_envs,), 0.003, device=device)          # Yaw速率增益[6](@ref)
        
        # --- 安全限制 ---
        self.max_roll_pitch = torch.tensor(0.3, device=device)                    # 最大滚转俯仰角[6](@ref)
        self.eps = torch.finfo(torch.float32).eps

    # ================= 独立控制接口 =================
    def position_control(self, sensor_data, desired_pos):
        """位置控制层：直接返回推力和力矩"""
        # 计算速度设定值
        pos_error = desired_pos - sensor_data['root_pos_w']
        vel_sp = pos_error * self.position_p_gain
        
        # 调用速度控制
        return self.velocity_control(sensor_data, vel_sp)

    def velocity_control(self, sensor_data, vel_sp):
        """速度控制层：直接返回推力和力矩"""
        # 计算加速度设定值
        vel_error = vel_sp - sensor_data['base_lin_vel']
        
        # 积分项更新（带限幅）
        self.velocity_integral += torch.clamp(
            vel_error * self.velocity_i_gain * self.dt,
            -self.velocity_integral_limit,
            self.velocity_integral_limit
        )
        accel_sp = vel_error * self.velocity_p_gain + self.velocity_integral
        
        # 调用加速度控制
        return self.acceleration_control(sensor_data, accel_sp)

    def acceleration_control(self, sensor_data, accel_sp, yaw_sp=0.0):
        """加速度控制层：直接返回推力和力矩"""
        # 坐标系转换
        R = self._get_rotation_matrix(sensor_data['root_quat_w'])
        force_body = torch.bmm(R.transpose(1,2), (self.mass * accel_sp).unsqueeze(-1)).squeeze(-1)
        
        # 生成姿态设定值（独立计算）
        thrust_dir = force_body / (torch.norm(force_body, dim=1, keepdim=True) + self.eps)
        attitude_sp = torch.zeros((self.num_envs, 3), device=self.device)
        attitude_sp[:, 0] = torch.atan2(-thrust_dir[:,1], thrust_dir[:,2])  # Roll[4](@ref)
        attitude_sp[:, 1] = torch.asin(torch.clamp(thrust_dir[:,0], -0.9, 0.9))  # Pitch[4](@ref)
        attitude_sp[:, 2] = yaw_sp
        attitude_sp[:, :2] = torch.clamp(attitude_sp[:, :2], -self.max_roll_pitch, self.max_roll_pitch)
        
        # 调用姿态控制
        return self.attitude_control(sensor_data, attitude_sp)

    def attitude_control(self, sensor_data, attitude_sp):
        """姿态控制层：直接返回推力和力矩"""
        current_euler = self.quat_to_euler(sensor_data['root_quat_w'])
        angle_error = attitude_sp[:, :2] - current_euler[:, :2]
        ang_vel = sensor_data['base_ang_vel'][:, :2]
        
        # PD控制（独立计算）[3](@ref)
        torque = (angle_error * self.attitude_p_gain - 
                 ang_vel * self.attitude_d_gain)
        
        # Yaw控制（独立通道）[6](@ref)
        yaw_rate_error = attitude_sp[:, 2] - current_euler[:, 2]
        torque_z = yaw_rate_error * self.yaw_rate_gain
        
        # 推力补偿（基于重力投影）[4](@ref)
        g_z = torch.clamp(sensor_data['projected_gravity_b'][:, 2], min=-1.0, max=-0.1)
        thrust = self.base_thrust / (-g_z.unsqueeze(1))
        
        # 推力限幅（独立处理）
        thrust_magnitude = torch.norm(thrust, dim=1, keepdim=True)
        thrust = torch.clamp(thrust_magnitude, 
                           min=0.1*self.base_thrust, 
                           max=self.max_thrust_ratio*self.base_thrust)
        
        return thrust, torch.cat([torque, torque_z.unsqueeze(1)], dim=1)

    # ================= 工具函数 =================
    def quat_to_euler(self, quat):
        """四元数转欧拉角（批量处理）"""
        w, x, y, z = quat.unbind(dim=1)
        roll = torch.atan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        pitch = torch.asin(2*(w*y - z*x))
        yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        return torch.stack([roll, pitch, yaw], dim=1)  # [num_envs, 3][4](@ref)

    def _get_rotation_matrix(self, quat):
        """从四元数获取旋转矩阵（严格转换）"""
        w, x, y, z = quat.unbind(dim=1)
        R = torch.zeros((self.num_envs, 3, 3), device=self.device)
        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - x*w)
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)
        return R  # [num_envs, 3, 3][6](@ref)



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

    # initialize pid controller
    controller = QuadrotorController(
        num_envs=num_envs,
        mass = robot_mass,
        gravity=9.81,
        dt=env_cfg.sim.dt,
        device=env.device
    )

    # initialize control target
    desired_pos=torch.zeros(env_cfg.scene.num_envs, 3, device=env.device)
    desired_pos[:, 2] = 5.0
    desired_vel=torch.zeros(env_cfg.scene.num_envs, 3, device=env.device)
    desired_vel[:, 0] = 0.5
    desired_attitude=torch.zeros(env_cfg.scene.num_envs, 3, device=env.device)
    desired_attitude[:, 2] = 0.2
    # simulate physics
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 300 == 0:
                count = 0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            
            root_quat_w = obs["policy"]['root_quat_w']
            root_ang_vel_b = obs["policy"]['root_ang_vel_b']
            root_lin_vel_b = obs["policy"]['root_lin_vel_b']
            root_pos_w = obs["policy"]['root_pos_w']
            projected_gravity_b = obs["policy"]['projected_gravity_b']
            imu_lin_acc_b = obs["policy"]['imu_lin_acc_b']
            sensor_data = {
                'root_quat_w': root_quat_w,
                'base_ang_vel': root_ang_vel_b,
                'base_lin_vel': root_lin_vel_b,
                'projected_gravity_b': projected_gravity_b,
                'root_pos_w': root_pos_w,
                'imu_lin_acc_b': imu_lin_acc_b
            }

            # actions = torch.zeros(env.num_envs, 4, device=env.device)
            thrust, torque = controller.attitude_control(sensor_data=sensor_data, attitude_sp=desired_attitude)
            # thrust, torque = controller.velocity_controller(sensor_data=sensor_data, desired_vel=desired_vel)
            # thrust, torque = controller.position_control(sensor_data, desired_pos)
            # TODO: 检查控制器结构
            # TODO：调参
            print('-'*20)
            # print('attitude',controller.gravity_to_attitude(projected_gravity_b)[:,0])
            print('torque',torque[:,0])
            print('base_ang_vel',root_ang_vel_b[:,0])
            print('base_lin_vel',root_lin_vel_b[:,0])
            print('-'*20)
            actions = torch.cat([thrust, torque], dim=1)
            obs, _ = env.step(actions)
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()