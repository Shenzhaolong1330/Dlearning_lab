"""
shenzhaolong 2025.4.10
This script demonstrates how to create a simple environment to evaluate the trained policy of stabilizing a quadcopter. 
With a camera attached on the front

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/learning_based_control/quadcopter_vision_env.py --enable_camera

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a quadcopter stabilization environment.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to spawn.")
parser.add_argument("--checkpoint", type=str, default='logs/rsl_rl/quadcopter_direct/2025-03-29_00-21-07/exported/policy.pt', help="Path to model checkpoint exported as jit.")

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
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

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
        self._processed_actions = self._raw_actions.clone().clamp(-1.0, 1.0)

    def apply_actions(self):
        # 输入的actions是总推力+三轴力矩
        mass = self._asset.root_physx_view.get_masses()[0].sum()
        g = 9.81
        self._thrust[:, 0, 2] = self.thrust_to_weight * mass * g * (self._processed_actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.moment_scale * self._processed_actions[:, 1:]
        # print(self._thrust.shape)
        # print(self._moment.shape)
        body_id = self._asset.find_bodies("body")[0]
        self._asset.set_external_force_and_torque(self._thrust, self._moment, body_id)


@configclass
class QuadcopterActionTermCfg(ActionTermCfg):
    """Configuration for the quadcopter action term."""
    class_type: type = QuadcopterActionTerm
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01



@configclass
class QuadcopterSceneCfg(InteractiveSceneCfg):
    
    # terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/front_cam",
        update_period=0.05,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.03, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
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
        # desired_pos_b = ObsTerm(func=get_desired_pos_b, params={"robot_pos_w":mdp.root_pos_w, "robot_alt_w":mdp.root_quat_w, "desired_pos_w":torch.zeros(3)})
        # DONE: desired_pos_b 放在观测类的外面计算
        root_pos_w = ObsTerm(func=mdp.root_pos_w)
        root_quat_w = ObsTerm(func=mdp.root_quat_w)

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
                "x": (-3.0,3.0),
                "y": (-3.0,3.0),
                "z": (0.0,2.0)}, # dict[str, tuple[float, float]],
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
        self.decimation = 2  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.01  # sim step every 5ms: 200Hz



class QuadcopterEnv(ManagerBasedEnv):
    """Quadcopter environment."""
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg):
        super().__init__(cfg)
        self.desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.desired_pos_w[:, 2] = 1.5


def get_desired_pos_b(robot_pos_w: torch.Tensor, robot_alt_w: torch.Tensor, desired_pos_w: torch.Tensor)-> torch.Tensor:
    desired_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_alt_w, desired_pos_w)
    return desired_pos_b



def main():
    """Main function."""
    # parse the arguments
    env_cfg = QuadcopterEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = QuadcopterEnv(cfg=env_cfg)
    # policy_path = os.path.abspath(args_cli.checkpoint)
    print('----------------',args_cli.checkpoint,'----------------')
    trained_model = torch.load(args_cli.checkpoint).to(env.device)

    # simulate physics
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 100 == 0:
                count = 0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            
            # obs fusion
            desired_pos_b = get_desired_pos_b(obs["policy"]['root_pos_w'], obs["policy"]['root_quat_w'], env.desired_pos_w)
            obs = torch.cat(
                [
                    obs["policy"]['root_lin_vel_b'],
                    obs["policy"]['root_ang_vel_b'], 
                    obs["policy"]['projected_gravity_b'], 
                    desired_pos_b
                ],
                dim=-1,
            )
            # actions = torch.zeros(env.num_envs, 4, device=env.device)
            # print(obs.shape)
            actions = trained_model.actor(obs)
            # step the environment
            obs, _ = env.step(actions)
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()