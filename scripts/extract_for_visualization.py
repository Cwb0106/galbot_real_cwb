# import os
# import cv2
# import numpy as np
# import argparse
# import collections
# import shutil
# from tqdm import tqdm
# from scipy.spatial.transform import Rotation as R
# from mcap.reader import make_reader
# from mcap_protobuf.decoder import DecoderFactory
# from google.protobuf.json_format import MessageToDict

# # 引入配置
# from config import DATA_ARGS, TOPIC_CONFIG as STATE_CONFIG, DEMO_PATH, OUTPUT_PATH
# from config_pcd import TOPIC_CONFIG as PCD_CONFIG, TARGET_FRAME
# from data_utils import parse_image_msg, get_timestamp_from_header

# # ==============================================================================
# # 1. 强力 TF 树 (保持你提供的版本不变)
# # ==============================================================================
# class RobustTfTree:
#     def __init__(self):
#         self.static_tfs = {}
#         self.dynamic_tfs = collections.defaultdict(list)
#         self.graph = collections.defaultdict(dict)
#         self.all_frames = set() 

#     def _pose_to_mat(self, trans, rot):
#         mat = np.eye(4)
#         mat[:3, 3] = [trans.x, trans.y, trans.z]
#         mat[:3, :3] = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
#         return mat

#     def load_tf_message(self, proto_msg, is_static=False):
#         for t in proto_msg.transforms:
#             parent = t.header.frame_id
#             child = t.child_frame_id
#             self.all_frames.add(parent)
#             self.all_frames.add(child)
#             self.graph[parent][child] = True
#             self.graph[child][parent] = True
#             mat = self._pose_to_mat(t.transform.translation, t.transform.rotation)
#             if is_static:
#                 self.static_tfs[(parent, child)] = mat
#             else:
#                 ts = get_timestamp_from_header(t.header)
#                 self.dynamic_tfs[(parent, child)].append((ts, mat))

#     def _find_nearest_tf(self, tf_list, query_time):
#         if not tf_list: return np.eye(4)
#         best_mat = tf_list[0][1]
#         min_dt = abs(tf_list[0][0] - query_time)
#         for ts, mat in tf_list:
#             dt = abs(ts - query_time)
#             if dt < min_dt:
#                 min_dt = dt
#                 best_mat = mat
#         return best_mat

#     def _get_edge_transform(self, u, v, query_time):
#         if (u, v) in self.static_tfs:
#             return np.linalg.inv(self.static_tfs[(u, v)])
#         if (u, v) in self.dynamic_tfs:
#             mat = self._find_nearest_tf(self.dynamic_tfs[(u, v)], query_time)
#             return np.linalg.inv(mat)
#         if (v, u) in self.static_tfs:
#             return self.static_tfs[(v, u)]
#         if (v, u) in self.dynamic_tfs:
#             return self._find_nearest_tf(self.dynamic_tfs[(v, u)], query_time)
#         return None

#     def lookup_transform(self, target_frame, source_frame, time):
#         if target_frame not in self.all_frames or source_frame not in self.all_frames:
#             return None
#         if target_frame == source_frame:
#             return np.eye(4)
#         queue = collections.deque([[source_frame]])
#         visited = {source_frame}
#         path = None
#         while queue:
#             curr_path = queue.popleft()
#             node = curr_path[-1]
#             if node == target_frame:
#                 path = curr_path
#                 break
#             for neighbor in self.graph[node]:
#                 if neighbor not in visited:
#                     visited.add(neighbor)
#                     new_path = list(curr_path)
#                     new_path.append(neighbor)
#                     queue.append(new_path)
#         if not path:
#             return None
#         T_final = np.eye(4)
#         for i in range(len(path) - 1):
#             u = path[i]
#             v = path[i+1]
#             T_uv = self._get_edge_transform(u, v, time)
#             if T_uv is None: return None
#             T_final = T_uv @ T_final
#         return T_final

# # ==============================================================================
# # 2. 辅助函数
# # ==============================================================================
# def get_intrinsic_matrix(camera_info_msg):
#     k_list = camera_info_msg.k
#     K = np.array(k_list).reshape(3, 3)
#     return K

# def find_nearest_data(target_ts, data_list, start_idx=0, max_delta=0.1):
#     if not data_list: return None, 0
#     best_data, min_dt, best_idx = None, max_delta, start_idx
#     for i in range(start_idx, len(data_list)):
#         ts, val = data_list[i]
#         dt = abs(ts - target_ts)
#         if dt < min_dt:
#             min_dt = dt
#             best_data = val
#             best_idx = i
#         elif ts > target_ts + max_delta:
#             break
#     return best_data, best_idx

# def get_delta_trans_and_rot(src, dst):
#     if len(src.shape) == 1: src = src[None, :]
#     if len(dst.shape) == 1: dst = dst[None, :]
#     def quat_to_yaw(qx, qy, qz, qw):
#         t3 = 2.0 * (qw * qz + qx * qy)
#         t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
#         return np.arctan2(t3, t4)
#     src_trans = src[:, 0:2]
#     dst_trans = dst[:, 0:2]
#     src_yaw = quat_to_yaw(src[:, 2], src[:, 3], src[:, 4], src[:, 5])
#     dst_yaw = quat_to_yaw(dst[:, 2], dst[:, 3], dst[:, 4], dst[:, 5])
#     cos_y = np.cos(src_yaw)
#     sin_y = np.sin(src_yaw)
#     dx_world = dst_trans[:, 0] - src_trans[:, 0]
#     dy_world = dst_trans[:, 1] - src_trans[:, 1]
#     dx_body = cos_y * dx_world + sin_y * dy_world
#     dy_body = -sin_y * dx_world + cos_y * dy_world
#     dtheta = (dst_yaw - src_yaw + np.pi) % (2 * np.pi) - np.pi
#     return np.stack([dx_body, dy_body, dtheta], axis=1)

# # ==============================================================================
# # 3. 主程序
# # ==============================================================================
# def main(data_dir, output_root):
#     # 1. 路径准备
#     mcap_file = data_dir
#     task_name = os.path.basename(mcap_file).replace('.mcap', '')
    
#     # 创建任务主目录
#     task_dir = os.path.join(output_root, task_name)
#     if os.path.exists(task_dir):
#         print(f"Directory {task_dir} exists, removing...")
#         shutil.rmtree(task_dir)
#     os.makedirs(task_dir)

#     # 创建视频目录
#     video_dir = os.path.join(task_dir, "video")
#     os.makedirs(video_dir, exist_ok=True)

#     # 初始化 TF 树
#     tf_tree = RobustTfTree()
    
#     # 初始化数据 Buffer
#     cam_buffer = { name: {'rgb': [], 'depth': [], 'info': []} for name in PCD_CONFIG['sensors'].keys() }
#     state_buffer = { 'odom': [], 'joints': {} }
#     for joint_key in DATA_ARGS['joint_map'].keys(): state_buffer['joints'][joint_key] = []

#     # Topic 映射
#     topic_map = {}
#     for name, conf in PCD_CONFIG['sensors'].items():
#         if conf['rgb']: topic_map[conf['rgb']] = (name, 'rgb')
#         if conf['depth']: topic_map[conf['depth']] = (name, 'depth')
#         if conf['info']: topic_map[conf['info']] = (name, 'info')
    
#     odom_topic = STATE_CONFIG['state_map']['odom']
#     wbcs_topic = STATE_CONFIG['state_map']['wbcs']
#     tf_topics = PCD_CONFIG['tf_topics']
#     all_topics = list(topic_map.keys()) + tf_topics + [odom_topic, wbcs_topic]

#     # --- 第一遍：读取 MCAP 到内存 ---
#     print(f"Reading MCAP: {mcap_file}...")
#     with open(mcap_file, "rb") as f:
#         reader = make_reader(f, decoder_factories=[DecoderFactory()])
#         for schema, channel, message, proto_msg in tqdm(reader.iter_decoded_messages(topics=all_topics), desc="Parsing"):
#             topic = channel.topic
#             if topic in tf_topics:
#                 is_static = (topic == "/embosa_tf_static")
#                 tf_tree.load_tf_message(proto_msg, is_static)
#                 continue
            
#             ts = get_timestamp_from_header(proto_msg.header)
            
#             if topic in topic_map:
#                 sensor_name, data_type = topic_map[topic]
#                 if data_type == 'rgb':
#                     img = parse_image_msg(proto_msg)
#                     if img is not None: cam_buffer[sensor_name]['rgb'].append((ts, img))
#                 elif data_type == 'depth':
#                     img = parse_image_msg(proto_msg)
#                     if img is not None: cam_buffer[sensor_name]['depth'].append((ts, img))
#                 elif data_type == 'info':
#                     cam_buffer[sensor_name]['info'].append((ts, get_intrinsic_matrix(proto_msg)))
#             elif topic == odom_topic:
#                 pos = proto_msg.pose.pose.position
#                 quat = proto_msg.pose.pose.orientation
#                 state_buffer['odom'].append((ts, [pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w]))
#             elif topic == wbcs_topic:
#                 msg_dict = MessageToDict(proto_msg, preserving_proto_field_name=True)
#                 sensor_map = msg_dict.get('joint_sensor_map', {})
#                 for internal_key, map_key in DATA_ARGS['joint_map'].items():
#                     if map_key in sensor_map:
#                         pos_list = sensor_map[map_key].get('position', [])
#                         if pos_list: state_buffer['joints'][internal_key].append((ts, pos_list))

#     # --- 第二遍：对齐、保存图片、生成视频 ---
#     primary_sensor = list(PCD_CONFIG['sensors'].keys())[0]
#     base_timeline = cam_buffer[primary_sensor]['rgb']
#     print(f"\nAligning & Saving to: {task_dir}")
#     print(f"Primary sensor: {primary_sensor} ({len(base_timeline)} frames)")
    
#     # 准备存储非图像数据的字典
#     traj_data = { 
#         'timestamps': [], 
#         'odom': [], 
#         # extrinsics/intrinsics 会根据相机名保存，例如 'extrinsic_left_wrist'
#     }
#     # 初始化 joint keys
#     for jk in state_buffer['joints']: traj_data[f'joint_{jk}'] = []
    
#     # 准备相机的目录和 VideoWriter
#     video_writers = {}
    
#     for sn in PCD_CONFIG['sensors']:
#         # 创建图片目录
#         os.makedirs(os.path.join(task_dir, sn, "rgb"), exist_ok=True)
#         if PCD_CONFIG['sensors'][sn]['depth']:
#             os.makedirs(os.path.join(task_dir, sn, "depth"), exist_ok=True)
        
#         # 初始化列表
#         traj_data[f'intrinsic_{sn}'] = []
#         traj_data[f'extrinsic_{sn}'] = []
        
#         # 初始化 Video Writer
#         # 假设分辨率与第一帧相同，帧率 30fps
#         if len(cam_buffer[sn]['rgb']) > 0:
#             h, w, _ = cam_buffer[sn]['rgb'][0][1].shape
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             vid_path = os.path.join(video_dir, f"{sn}.mp4")
#             video_writers[sn] = cv2.VideoWriter(vid_path, fourcc, 30.0, (w, h))

#     # 索引管理
#     indices = {'odom': 0}
#     for jk in state_buffer['joints']: indices[jk] = 0
#     cam_indices = {}
#     for sn in PCD_CONFIG['sensors']: 
#         cam_indices[f'{sn}_depth'] = 0
#         cam_indices[f'{sn}_rgb'] = 0

#     # 主循环
#     for i, (ts_primary, _) in enumerate(tqdm(base_timeline, desc="Processing Frames")):
#         traj_data['timestamps'].append(ts_primary)
        
#         # 1. Action (Odom & Joints)
#         odom_val, new_idx = find_nearest_data(ts_primary, state_buffer['odom'], indices['odom'])
#         indices['odom'] = new_idx
#         traj_data['odom'].append(odom_val if odom_val is not None else np.zeros(7))
        
#         for jk, j_list in state_buffer['joints'].items():
#             j_val, new_idx = find_nearest_data(ts_primary, j_list, indices[jk])
#             indices[jk] = new_idx
#             traj_data[f'joint_{jk}'].append(j_val if j_val is not None else [])

#         # 2. Camera Processing
#         for sensor_name, conf in PCD_CONFIG['sensors'].items():
#             # --- RGB ---
#             if sensor_name == primary_sensor:
#                 rgb_img = base_timeline[i][1]
#             else:
#                 rgb_data, new_idx = find_nearest_data(ts_primary, cam_buffer[sensor_name]['rgb'], cam_indices[f'{sensor_name}_rgb'])
#                 cam_indices[f'{sensor_name}_rgb'] = new_idx
#                 rgb_img = rgb_data

#             if rgb_img is not None:
#                 # RGB -> BGR for OpenCV
#                 bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                
#                 # A. 保存图片
#                 img_name = f"{i:06d}.png"
#                 cv2.imwrite(os.path.join(task_dir, sensor_name, "rgb", img_name), bgr_img)
                
#                 # B. 写入视频
#                 if sensor_name in video_writers:
#                     video_writers[sensor_name].write(bgr_img)

#             # --- Depth ---
#             if conf['depth'] and len(cam_buffer[sensor_name]['depth']) > 0:
#                 depth_data, new_idx = find_nearest_data(ts_primary, cam_buffer[sensor_name]['depth'], cam_indices[f'{sensor_name}_depth'])
#                 cam_indices[f'{sensor_name}_depth'] = new_idx
#                 depth_img = depth_data
                
#                 if depth_img is not None:
#                     # 深度图通常保存为 16位 PNG (单位毫米)
#                     # 假设原始数据已经是 uint16 mm，直接保存
#                     # 如果原始是 float 米，需要: (depth_img * 1000).astype(np.uint16)
#                     img_name = f"{i:06d}.png"
#                     cv2.imwrite(os.path.join(task_dir, sensor_name, "depth", img_name), depth_img)

#             # --- Intrinsics ---
#             K = cam_buffer[sensor_name]['info'][0][1] if cam_buffer[sensor_name]['info'] else np.eye(3)
#             traj_data[f'intrinsic_{sensor_name}'].append(K)
            
#             # --- Extrinsics (TF Lookup) ---
#             camera_frame = conf['frame_id']
#             T_camera_to_base = np.eye(4)
#             try:
#                 # 查找 T_camera_to_base (Target=Base, Source=Camera)
#                 T = tf_tree.lookup_transform(target_frame=TARGET_FRAME, source_frame=camera_frame, time=ts_primary)
#                 if T is not None:
#                     T_camera_to_base = T
#                 else:
#                     # 沿用上一帧
#                     if len(traj_data[f'extrinsic_{sensor_name}']) > 0:
#                         T_camera_to_base = traj_data[f'extrinsic_{sensor_name}'][-1]
#             except:
#                 pass
            
#             traj_data[f'extrinsic_{sensor_name}'].append(T_camera_to_base)

#     # 释放 Video Writer
#     for vw in video_writers.values():
#         vw.release()

#     # === 保存 Trajectory 数据到 NPZ ===
#     print("\nSaving trajectory data...")
    
#     # 计算 Action Delta (基于 Odom)
#     odom_np = np.array(traj_data['odom'])
#     if len(odom_np) > 1:
#         odom_for_calc = odom_np[:, [0, 1, 3, 4, 5, 6]]
#         actions = get_delta_trans_and_rot(odom_for_calc[:-1], odom_for_calc[1:])
#         traj_data['action_delta'] = np.vstack([actions, np.zeros((1, 3))])
#     else:
#         traj_data['action_delta'] = np.zeros((len(odom_np), 3))

#     # 转换为 numpy array 并保存
#     npz_path = os.path.join(task_dir, "trajectory.npz")
#     save_dict = {k: np.array(v) for k, v in traj_data.items()}
#     np.savez_compressed(npz_path, **save_dict)

#     print(f"Done! Data saved to: {task_dir}")
#     print(f"Videos saved to: {video_dir}")

# if __name__ == "__main__":
#     main(DEMO_PATH, OUTPUT_PATH)


import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

# 引入配置 (保持不变)
from config import DEMO_PATH, OUTPUT_PATH
from config_pcd import TOPIC_CONFIG as PCD_CONFIG
from data_utils import parse_image_msg, get_timestamp_from_header

# 辅助函数：查找最近的数据（用于时间对齐）
def find_nearest_data(target_ts, data_list, start_idx=0, max_delta=0.1):
    if not data_list: return None, 0
    best_data, min_dt, best_idx = None, max_delta, start_idx
    for i in range(start_idx, len(data_list)):
        ts, val = data_list[i]
        dt = abs(ts - target_ts)
        if dt < min_dt:
            min_dt = dt
            best_data = val
            best_idx = i
        elif ts > target_ts + max_delta:
            break
    return best_data, best_idx

def main(data_dir, output_root):
    # 1. 路径准备
    mcap_file = data_dir
    task_name = os.path.basename(mcap_file).replace('.mcap', '')
    
    # 创建任务目录
    video_dir = os.path.join(output_root, task_name)
    # video_dir = os.path.join(task_dir, "video")
    
    if os.path.exists(video_dir):
        print(f"Video directory {video_dir} exists, cleaning up...")
        shutil.rmtree(video_dir)
    os.makedirs(video_dir, exist_ok=True)

    # 2. 准备 Topic 映射 (只关注 RGB)
    # 结构: topic_url -> sensor_name
    rgb_topic_map = {}
    for name, conf in PCD_CONFIG['sensors'].items():
        if conf['rgb']: 
            rgb_topic_map[conf['rgb']] = name
            
    target_topics = list(rgb_topic_map.keys())
    
    # 初始化数据 Buffer
    # 结构: cam_buffer['camera_name'] = [(ts, img), (ts, img)...]
    cam_buffer = { name: [] for name in PCD_CONFIG['sensors'].keys() }

    # --- 第一遍：只读取 RGB 到内存 ---
    print(f"Reading MCAP (RGB Only): {mcap_file}...")
    with open(mcap_file, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        # 只解码 RGB 相关的 topic，速度会快很多
        for schema, channel, message, proto_msg in tqdm(reader.iter_decoded_messages(topics=target_topics), desc="Parsing"):
            topic = channel.topic
            sensor_name = rgb_topic_map[topic]
            ts = get_timestamp_from_header(proto_msg.header)
            
            # 解析图像
            img = parse_image_msg(proto_msg)
            if img is not None: 
                cam_buffer[sensor_name].append((ts, img))

    # --- 第二遍：对齐并生成视频 ---
    # 选取主相机作为时间轴基准
    primary_sensor = list(PCD_CONFIG['sensors'].keys())[0]
    base_timeline = cam_buffer[primary_sensor]
    
    if not base_timeline:
        print(f"Error: No data found for primary sensor {primary_sensor}")
        return

    print(f"\nGenerating Videos in: {video_dir}")
    print(f"Primary sensor: {primary_sensor} ({len(base_timeline)} frames)")

    # 初始化 Video Writer
    video_writers = {}
    
    # 遍历每个传感器初始化 Writer
    for sn in PCD_CONFIG['sensors']:
        if len(cam_buffer[sn]) > 0:
            # 获取第一帧的宽高
            h, w, _ = cam_buffer[sn][0][1].shape
            # mp4v 编码兼容性较好，也可以尝试 avc1
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            vid_path = os.path.join(video_dir, f"{sn}.mp4")
            # 假设帧率 30fps
            video_writers[sn] = cv2.VideoWriter(vid_path, fourcc, 30.0, (w, h))
        else:
            print(f"Warning: No images found for sensor {sn}")

    # 索引记录，用于加速查找
    cam_indices = { sn: 0 for sn in PCD_CONFIG['sensors'] }

    # 主循环：基于主相机时间戳对齐
    for i, (ts_primary, _) in enumerate(tqdm(base_timeline, desc="Writing Video")):
        
        for sensor_name in PCD_CONFIG['sensors']:
            if sensor_name not in video_writers:
                continue

            # 获取图像
            rgb_img = None
            
            if sensor_name == primary_sensor:
                # 主相机直接取当前帧
                rgb_img = base_timeline[i][1]
            else:
                # 其他相机根据时间戳查找最近帧
                rgb_data, new_idx = find_nearest_data(ts_primary, cam_buffer[sensor_name], cam_indices[sensor_name])
                cam_indices[sensor_name] = new_idx
                if rgb_data is not None:
                    rgb_img = rgb_data

            # 写入视频
            if rgb_img is not None:
                # RGB -> BGR (OpenCV 需要 BGR)
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                video_writers[sensor_name].write(bgr_img)

    # 释放所有 Video Writer
    for vw in video_writers.values():
        vw.release()

    print(f"Done! Videos saved to: {video_dir}")

if __name__ == "__main__":
    from pathlib import Path
    
    root_path = Path("/home/ubuntu/cwb_works/project/galbot_real_world_cwb/data_example")
    output_path = "/home/ubuntu/cwb_works/project/galbot_real_world_cwb/output_demo"

    # 使用 glob 直接筛选包含 SYNC.mcap 的文件
    for path in root_path.glob("*SYNC.mcap"):
        print(f"Processing: {path.name}")
        # 注意：建议转为 str，防止 main 函数内部不支持 Path 对象
        main(data_dir=str(path), output_root=output_path)

# if __name__ == "__main__":
#     main(DEMO_PATH, OUTPUT_PATH)