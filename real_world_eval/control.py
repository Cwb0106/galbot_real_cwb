import sys
import numpy as np

# TODO 
sys.path.append("/home/ubuntu/cwb_works/project/galbot_real_world_cwb/galbotsdk-new_pb/src")

from robot_executor import RobotExecutor
from _impl.robot_sensor_embosa import RobotSensorEmbosa
from sdk_utils.galbot_logger import (
    INFO,
    DEBUG,
    WARN,
    ERROR,


    EXCEPTION,
    create_local_logger,
    list_local_loggers,
)

"""
rgb: head, wrist
depth -> point cloud
imu
joint
"""

# "/raid/wenbo/project/mshab/action_space.npz"

class GalbotControl:
    """
    RobotExecutorEmbosa(): 
    RobotSensorEmbosa(): get img/imu/depth data
    RobotMotion(): motion planningf + IK + FK

    """
    def __init__(self, cfgs):
        self.arm_executor = RobotExecutor(robot_cfg="robot_config.yaml")
        INFO("RobotExecutorEmbosa initialized successfully.")
        self.sensor = RobotSensorEmbosa()
        INFO("RobotSensorEmbosa initialized successfully.")
        # self.motion_planning = RobotMotion()
        # TODO
        self.odom_executor = 111
        self.camera_list = cfgs.sensor.camera_list #  # ["head_left", "head_right", "left_wrist", "right_wrist"]
        # 0-2: odom, 3-10: left_arm, 11-18: right arm
        self.enabled_groups = cfgs.control.enabled_groups
        self.enabled_modalities = cfgs.snesor.enabled_modalities

        self.stats = np.load("/raid/wenbo/project/mshab/action_space.npz")


    def reset(self):
        obs = self._get_observation_from_galbot()
        # norm state
        obs['state'] = self._normalize_data(obs['state'], self.stats['state'])
        return obs

    def step(self, action):
        processed_action = self._post_processing_action(action)
        info = self._galbot_execution(processed_action)
        obs = self._get_observation_from_galbot()
        obs['state'] = self._normalize_data(obs['state'], self.stats['state'])
        return obs

    def _normalize_data(self, data, stats):
        """归一化到 [-1, 1]"""
        denom = stats['max'] - stats['min']
        denom[denom == 0] = 1.0 
        normalized = (data - stats['min']) / denom
        normalized = normalized * 2.0 - 1.0
        return normalized

    def _unnormalize_data(self, normalized_data, stats):
        """反归一化"""
        denom = stats['max'] - stats['min']
        denom[denom == 0] = 1.0 
        data = (normalized_data + 1.0) / 2.0
        data = data * denom + stats['min']
        return data

    def _get_observation_from_galbot(self):
        """
        Returns:
            dict:
                {
                    "observations": "head_left_rgb", "left_wrist_rgb", "left_wrist_depth" ...,
                    "state": state
                }
        """
        observations = dict()
        for camera_name in self.camera_list:
            (rgb, depth) = self.sensor.get_image(camera_name, "align", False)
            if "rgb" in self.enabled_modalities:
                observations["observations"][camera_name + "_rgb"] = rgb
            if "depth" in self.enabled_modalities:
                observations["observations"][camera_name + "_depth"] = depth
            if "state" in self.enabled_modalities:
                # TODO get current arm joint position, note: need to select enabled joint from it !!!!
                temp_state = self.arm_executor.get_whole_body_joint_position()
                observations["state"]
            
            # if "pc" in self.enabled_modalities:
            # if "imu" in self.enabled_modalities:
        return observations
    
    # 默认一定会有odom这个space，然后-> left -> right
    def _post_processing_action(self, action):
        action = self._unnormalize_data(action, self.stats['action'])
        processed_action = dict()
        if "odom" in self.enabled_groups:
            processed_action['odom'] = action[:3]
        if "left_arm" in self.enabled_groups:
            processed_action['left_arm'] = action[3:10]
        if "right_arm" in self.enabled_groups:
            processed_action['right_arm'] = action[10:17]

        return processed_action


    def _galbot_execution(self, unnormalize_action):
        is_success = True
        for joint_name in self.enabled_groups:
            if joint_name == "left_arm":
                self.arm_executor.set_arm_joint_angles(
                    unnormalize_action["odom"].tolist(),
                    speed=0.5,
                    arm="left_arm",
                    asynchronous=True,
                )
            elif joint_name == "right_arm":
                self.arm_executor.set_arm_joint_angles(
                    unnormalize_action["right_arm"].tolist(),
                    speed=0.5,
                    arm="right_arm",
                    asynchronous=True,
                )
            elif joint_name == "odom":
                # TODO
                print("缺少odom控制函数")
                pass
            else:
                ERROR(f"{joint_name}, illegal joint")
                is_success = False
        return is_success
        

