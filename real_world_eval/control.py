import sys


# TODO 
sys.path.append("/home/ubuntu/cwb_works/project/galbot_real_world_cwb/galbotsdk-new_pb/src")

from robot_executor import RobotExecutorEmbosa
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



class GalbotControl:
    """
    RobotExecutorEmbosa(): 
    RobotSensorEmbosa(): get img/imu/depth data
    RobotMotion(): motion planningf + IK + FK
    
    """
    def __init__(self, cfgs):
        self.arm_executor = RobotExecutorEmbosa()
        INFO("RobotExecutorEmbosa initialized successfully.")
        self.sensor = RobotSensorEmbosa()
        INFO("RobotSensorEmbosa initialized successfully.")
        # self.motion_planning = RobotMotion()
        # TODO
        self.odom_executor = 111
        self.camera_list = cfgs.sensor.camera_list #  # ["head_left", "head_right", "left_wrist", "right_wrist"]
        self.enabled_groups = cfgs.control.enabled_groups
        self.enabled_modalities = cfgs.snesor.enabled_modalities


    def reset(self):
        obs = self._get_observation_from_galbot()
        return obs

    def step(self, action):
        unnormalize_action = self._unnormalization_action(action)
        info = self._galbot_execution(unnormalize_action)
        obs = self._get_observation_from_galbot()
        return obs


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
                observations["state"] = self.arm_executor.get_whole_body_joint_position()
            # if "pc" in self.enabled_modalities:
            # if "imu" in self.enabled_modalities:
        return observations
    

    def _unnormalization_action(self):
        print("TODO")
        exit(1)


    def _galbot_execution(self, unnormalize_action):
        is_success = True
        for joint_name in self.enabled_groups:
            if joint_name == "left_arm":
                self.arm_executor.set_arm_joint_angles(
                    # unnormalize_action.tolist(),
                    speed=0.5,
                    arm="left_arm",
                    asynchronous=True,
                )
            elif joint_name == "right_arm":
                self.arm_executor.set_arm_joint_angles(
                    # unnormalize_action.tolist(),
                    speed=0.5,
                    arm="right_arm",
                    asynchronous=True,
                )
            elif joint_name == "odom":
                # TODO
                pass
            else:
                ERROR(f"{joint_name}, illegal joint")
                is_success = False
        return is_success
        