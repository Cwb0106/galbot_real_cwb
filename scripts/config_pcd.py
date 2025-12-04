# config_pcd.py

# ================= 配置区域 =================
PCD_FORMAT = 'ply' 
DEPTH_SCALE = 1000.0 
MAX_DEPTH_METERS = 2.0
TARGET_FRAME = "base_link" 

# ================= Topic 定义 =================
# config_pcd.py
TOPIC_CONFIG = {
    "sensors": {
        "left_wrist": {
            "rgb": "/left_arm_camera/color/image_raw",
            "depth": "/left_arm_camera/depth/image_raw",
            "info": "/left_arm_camera/depth/camera_info",
            "frame_id": "left_arm_camera_color_optical_frame" # <--- 确认无误
        },
        "right_wrist": {
            "rgb": "/right_arm_camera/color/image_raw",
            "depth": "/right_arm_camera/depth/image_raw",
            "info": "/right_arm_camera/depth/camera_info",
            "frame_id": "right_arm_camera_color_optical_frame" # <--- 确认无误
        },
        # 头部的名字需要稍微改一下，因为 TF 里叫 head_left... 而不是 front_head...
        "front_head_left": {
            "rgb": "/front_head_camera/left_color/image_raw",
            "depth": None,
            "info": "/front_head_camera/left_color/camera_info",
            "frame_id": "head_left_camera_color_optical_frame" # <--- [修改] 匹配 TF 树
        },
        "front_head_right": {
            "rgb": "/front_head_camera/right_color/image_raw",
            "depth": None,
            "info": "/front_head_camera/right_color/camera_info",
            "frame_id": "head_right_camera_color_optical_frame" # <--- [修改] 匹配 TF 树
        }
    },
    "tf_topics": ["/embosa_tf", "/embosa_tf_static"]
}