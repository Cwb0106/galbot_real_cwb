# config.py
import numpy as np

DEMO_PATH = "/home/ubuntu/cwb_works/project/galbot_real_world_cwb/data_example/20251127_105419_record0.SYNC.mcap"
OUTPUT_PATH = "/home/ubuntu/cwb_works/project/galbot_real_world_cwb/output"

# ================= 基础配置 =================
IMG_WIDTH = 640
IMG_HEIGHT = 360
STATE_DIM = 29 

# ================= Topic 映射 (请检查这里！) =================
TOPIC_CONFIG = {
    "camera_map": {
        # --- RGB 相机 ---
        "camera_front_head_left": "/front_head_camera/left_color/image_raw",
        "camera_front_head_right": "/front_head_camera/right_color/image_raw",
        "camera_left_wrist": "/left_arm_camera/color/image_raw",
        "camera_right_wrist": "/right_arm_camera/color/image_raw",
        
        # --- [新增] Depth 相机 (必须加在这里才能被提取为视频) ---
        "depth_left_wrist": "/left_arm_camera/depth/image_raw",
        "depth_right_wrist": "/right_arm_camera/depth/image_raw",
    },
    "state_map": {
        "odom": "/odom/base_link",
        "wbcs": "singorix/wbcs/sensor"
    }
}

# 自动生成 VIDEO_KEYS，这样 data_utils 就会把 depth 也当作视频流处理
VIDEO_KEYS = list(TOPIC_CONFIG["camera_map"].keys())

# ================= 其他参数 =================
DATA_ARGS = {
    "fps": 15,
    "video_codec": "mp4v", 
    "repo_name": "galbot_tele_new",
    "joint_map": {
        'state_right_arm_joint_position': 'right_arm',
        'state_right_arm_gripper_width': 'right_gripper',
        'state_front_head_joint': 'head',
        'state_body_joint_position': 'leg',
        'state_left_arm_joint_position': 'left_arm',
        'state_left_arm_gripper_width': 'left_gripper'
    }
}

# ... FEATURES 定义 (验证脚本不需要) ...
FEATURES = {}