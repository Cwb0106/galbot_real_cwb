import os
import numpy as np
import argparse
import h5py
import collections
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from google.protobuf.json_format import MessageToDict

from config import DATA_ARGS, TOPIC_CONFIG as STATE_CONFIG, DEMO_PATH, OUTPUT_PATH
from config_pcd import TOPIC_CONFIG as PCD_CONFIG, TARGET_FRAME
from data_utils import parse_image_msg, get_timestamp_from_header




# ==============================================================================
# 1. 强力 TF 树 (修复了参数名和矩阵求逆逻辑)
# ==============================================================================
class RobustTfTree:
    def __init__(self):
        self.static_tfs = {}
        self.dynamic_tfs = collections.defaultdict(list)
        self.graph = collections.defaultdict(dict)
        self.all_frames = set() 

    def _pose_to_mat(self, trans, rot):
        mat = np.eye(4)
        mat[:3, 3] = [trans.x, trans.y, trans.z]
        mat[:3, :3] = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
        return mat

    def load_tf_message(self, proto_msg, is_static=False):
        for t in proto_msg.transforms:
            parent = t.header.frame_id
            child = t.child_frame_id
            
            self.all_frames.add(parent)
            self.all_frames.add(child)
            
            # 建立双向图连接
            self.graph[parent][child] = True
            self.graph[child][parent] = True
            
            mat = self._pose_to_mat(t.transform.translation, t.transform.rotation)
            
            if is_static:
                self.static_tfs[(parent, child)] = mat
            else:
                ts = get_timestamp_from_header(t.header)
                self.dynamic_tfs[(parent, child)].append((ts, mat))

    def _find_nearest_tf(self, tf_list, query_time):
        if not tf_list: return np.eye(4)
        best_mat = tf_list[0][1]
        min_dt = abs(tf_list[0][0] - query_time)
        # 简单遍历寻找最近时间戳
        for ts, mat in tf_list:
            dt = abs(ts - query_time)
            if dt < min_dt:
                min_dt = dt
                best_mat = mat
        return best_mat

    def _get_edge_transform(self, u, v, query_time):
        """
        获取 T_v_u (即 P_v = T_v_u * P_u)
        """
        # 情况 1: u(Parent) -> v(Child). 存储的是 T_u_v. 我们要 T_v_u = Inv(T_u_v)
        if (u, v) in self.static_tfs:
            return np.linalg.inv(self.static_tfs[(u, v)])
        if (u, v) in self.dynamic_tfs:
            mat = self._find_nearest_tf(self.dynamic_tfs[(u, v)], query_time)
            return np.linalg.inv(mat)
            
        # 情况 2: v(Parent) -> u(Child). 存储的是 T_v_u. 直接返回.
        if (v, u) in self.static_tfs:
            return self.static_tfs[(v, u)]
        if (v, u) in self.dynamic_tfs:
            return self._find_nearest_tf(self.dynamic_tfs[(v, u)], query_time)
            
        return None

    def lookup_transform(self, target_frame, source_frame, time):
        # 1. 基础检查
        if target_frame not in self.all_frames:
            # 只有在非常确信出错时才打印，避免刷屏
            # print(f"[TF Error] Target '{target_frame}' not found.")
            return None
        if source_frame not in self.all_frames:
            # print(f"[TF Error] Source '{source_frame}' not found.")
            return None
        if target_frame == source_frame:
            return np.eye(4)

        # 2. BFS 寻找路径
        queue = collections.deque([[source_frame]])
        visited = {source_frame}
        path = None
        
        while queue:
            curr_path = queue.popleft()
            node = curr_path[-1]
            if node == target_frame:
                path = curr_path
                break
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(curr_path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        
        if not path:
            return None

        # 3. 计算变换矩阵 T_target_source
        T_final = np.eye(4)
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            T_uv = self._get_edge_transform(u, v, time)
            if T_uv is None:
                return None
            T_final = T_uv @ T_final
            
        return T_final

# ==============================================================================
# 2. 辅助函数
# ==============================================================================
def get_intrinsic_matrix(camera_info_msg):
    k_list = camera_info_msg.k
    K = np.array(k_list).reshape(3, 3)
    return K

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

def get_delta_trans_and_rot(src, dst):
    if len(src.shape) == 1: src = src[None, :]
    if len(dst.shape) == 1: dst = dst[None, :]

    def quat_to_yaw(qx, qy, qz, qw):
        t3 = 2.0 * (qw * qz + qx * qy)
        t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
        return np.arctan2(t3, t4)

    src_trans = src[:, 0:2]
    dst_trans = dst[:, 0:2]
    src_yaw = quat_to_yaw(src[:, 2], src[:, 3], src[:, 4], src[:, 5])
    dst_yaw = quat_to_yaw(dst[:, 2], dst[:, 3], dst[:, 4], dst[:, 5])

    cos_y = np.cos(src_yaw)
    sin_y = np.sin(src_yaw)
    dx_world = dst_trans[:, 0] - src_trans[:, 0]
    dy_world = dst_trans[:, 1] - src_trans[:, 1]
    dx_body = cos_y * dx_world + sin_y * dy_world
    dy_body = -sin_y * dx_world + cos_y * dy_world
    dtheta = (dst_yaw - src_yaw + np.pi) % (2 * np.pi) - np.pi
    
    return np.stack([dx_body, dy_body, dtheta], axis=1)

# ==============================================================================
# 3. 主程序
# ==============================================================================
def main(data_dir, output_root):
    mcap_file = data_dir
    file_base_name = os.path.basename(mcap_file).replace('.mcap', '')
    output_h5_path = os.path.join(output_root, f"{file_base_name}.h5")
    os.makedirs(output_root, exist_ok=True)

    tf_tree = RobustTfTree()
    
    cam_buffer = { name: {'rgb': [], 'depth': [], 'info': []} for name in PCD_CONFIG['sensors'].keys() }
    state_buffer = { 'odom': [], 'joints': {} }
    for joint_key in DATA_ARGS['joint_map'].keys(): state_buffer['joints'][joint_key] = []

    topic_map = {}
    for name, conf in PCD_CONFIG['sensors'].items():
        if conf['rgb']: topic_map[conf['rgb']] = (name, 'rgb')
        if conf['depth']: topic_map[conf['depth']] = (name, 'depth')
        if conf['info']: topic_map[conf['info']] = (name, 'info')
    
    odom_topic = STATE_CONFIG['state_map']['odom']
    wbcs_topic = STATE_CONFIG['state_map']['wbcs']
    tf_topics = PCD_CONFIG['tf_topics']
    all_topics = list(topic_map.keys()) + tf_topics + [odom_topic, wbcs_topic]

    print(f"Reading MCAP: {mcap_file}...")
    with open(mcap_file, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, proto_msg in tqdm(reader.iter_decoded_messages(topics=all_topics), desc="Parsing Messages"):
            topic = channel.topic
            
            if topic in tf_topics:
                is_static = (topic == "/embosa_tf_static")
                tf_tree.load_tf_message(proto_msg, is_static)
                continue
            
            ts = get_timestamp_from_header(proto_msg.header)
            
            if topic in topic_map:
                sensor_name, data_type = topic_map[topic]
                if data_type == 'rgb':
                    img = parse_image_msg(proto_msg)
                    if img is not None: cam_buffer[sensor_name]['rgb'].append((ts, img))
                elif data_type == 'depth':
                    img = parse_image_msg(proto_msg)
                    if img is not None: cam_buffer[sensor_name]['depth'].append((ts, img))
                elif data_type == 'info':
                    cam_buffer[sensor_name]['info'].append((ts, get_intrinsic_matrix(proto_msg)))
            
            elif topic == odom_topic:
                pos = proto_msg.pose.pose.position
                quat = proto_msg.pose.pose.orientation
                state_buffer['odom'].append((ts, [pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w]))
            
            elif topic == wbcs_topic:
                msg_dict = MessageToDict(proto_msg, preserving_proto_field_name=True)
                sensor_map = msg_dict.get('joint_sensor_map', {})
                for internal_key, map_key in DATA_ARGS['joint_map'].items():
                    if map_key in sensor_map:
                        pos_list = sensor_map[map_key].get('position', [])
                        if pos_list: state_buffer['joints'][internal_key].append((ts, pos_list))

    primary_sensor = list(PCD_CONFIG['sensors'].keys())[0]
    base_timeline = cam_buffer[primary_sensor]['rgb']
    print(f"\nAligning data to primary sensor: {primary_sensor} ({len(base_timeline)} frames)")
    
    final_data = { 
        'timestamps': [], 
        'action': { 'odom': [] }, 
        'image': {}, 'depth': {}, 'intrinsic': {}, 'extrinsic': {}
    }
    for jk in state_buffer['joints']: final_data['action'][jk] = []
    for sn in PCD_CONFIG['sensors']:
        final_data['image'][sn] = []
        final_data['depth'][sn] = []
        final_data['intrinsic'][sn] = []
        final_data['extrinsic'][sn] = []

    indices = {'odom': 0}
    for jk in state_buffer['joints']: indices[jk] = 0
    cam_indices = {}
    for sn in PCD_CONFIG['sensors']: 
        cam_indices[f'{sn}_depth'] = 0
        cam_indices[f'{sn}_rgb'] = 0

    dummy_cache = {}

    for i, (ts_primary, _) in enumerate(tqdm(base_timeline, desc="Aligning Data")):
        final_data['timestamps'].append(ts_primary)
        
        # Action (Odom & Joints)
        odom_val, new_idx = find_nearest_data(ts_primary, state_buffer['odom'], indices['odom'])
        indices['odom'] = new_idx
        final_data['action']['odom'].append(odom_val if odom_val is not None else np.zeros(7))
        
        for jk, j_list in state_buffer['joints'].items():
            j_val, new_idx = find_nearest_data(ts_primary, j_list, indices[jk])
            indices[jk] = new_idx
            final_data['action'][jk].append(j_val if j_val is not None else [])

        # Camera
        for sensor_name, conf in PCD_CONFIG['sensors'].items():
            if sensor_name == primary_sensor:
                rgb_img = base_timeline[i][1]
            else:
                rgb_data, new_idx = find_nearest_data(ts_primary, cam_buffer[sensor_name]['rgb'], cam_indices[f'{sensor_name}_rgb'])
                cam_indices[f'{sensor_name}_rgb'] = new_idx
                rgb_img = rgb_data
            
            if rgb_img is not None: dummy_cache[f"{sensor_name}_rgb"] = np.zeros_like(rgb_img)
            elif f"{sensor_name}_rgb" in dummy_cache: rgb_img = dummy_cache[f"{sensor_name}_rgb"]
            if rgb_img is not None: final_data['image'][sensor_name].append(rgb_img)

            if conf['depth'] and len(cam_buffer[sensor_name]['depth']) > 0:
                depth_data, new_idx = find_nearest_data(ts_primary, cam_buffer[sensor_name]['depth'], cam_indices[f'{sensor_name}_depth'])
                cam_indices[f'{sensor_name}_depth'] = new_idx
                depth_img = depth_data
                if depth_img is not None: dummy_cache[f"{sensor_name}_depth"] = np.zeros_like(depth_img)
                elif f"{sensor_name}_depth" in dummy_cache: depth_img = dummy_cache[f"{sensor_name}_depth"]
                if depth_img is not None: final_data['depth'][sensor_name].append(depth_img)
            
            K = cam_buffer[sensor_name]['info'][0][1] if cam_buffer[sensor_name]['info'] else np.eye(3)
            final_data['intrinsic'][sensor_name].append(K)
            
            # --- Extrinsic Lookup ---
            camera_frame = conf['frame_id']
            T_camera_to_base = np.eye(4)
            

            # [修复] 参数名对齐: target_frame, source_frame
            T = tf_tree.lookup_transform(target_frame=TARGET_FRAME, source_frame=camera_frame, time=ts_primary)
            if T is not None:
                T_camera_to_base = T
            else:
                if len(final_data['extrinsic'][sensor_name]) > 0:
                    T_camera_to_base = final_data['extrinsic'][sensor_name][-1]
      
            
            if i == 0:
                if np.allclose(T_camera_to_base, np.eye(4)):
                    print(f"\n[⚠️ FATAL] {sensor_name}: Failed to transform {camera_frame} -> {TARGET_FRAME}")
                else:
                    # 简单检查 Z 轴是否为正 (相机通常在地上)
                    z_val = T_camera_to_base[2, 3]
                    status = "✅ OK" if z_val > 0 else "❓ Weird (Underground?)"
                    print(f"\n[{status}] {sensor_name} Extrinsic: Z={z_val:.3f}m")

            final_data['extrinsic'][sensor_name].append(T_camera_to_base)

    # === Action Deltas ===
    print("\nComputing Action Deltas...")
    odom_np = np.array(final_data['action']['odom'])
    if len(odom_np) > 1:
        odom_for_calc = odom_np[:, [0, 1, 3, 4, 5, 6]]
        actions = get_delta_trans_and_rot(odom_for_calc[:-1], odom_for_calc[1:])
        final_data['action']['action_delta'] = np.vstack([actions, np.zeros((1, 3))])
    else:
        final_data['action']['action_delta'] = np.zeros((len(odom_np), 3))

    # === Saving ===
    print(f"Saving HDF5: {output_h5_path}...")
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('timestamps', data=np.array(final_data['timestamps']))
        
        grp_act = f.create_group('action')
        for k, v in final_data['action'].items():
            if len(v) > 0 and not (isinstance(v[0], list) and len(v[0]) == 0):
                grp_act.create_dataset(k, data=np.array(v))

        for gname in ['image', 'depth', 'intrinsic', 'extrinsic']:
            grp = f.create_group(gname)
            for sname in final_data[gname]:
                d = final_data[gname][sname]
                if len(d) > 0:
                    comp = 'gzip' if gname in ['image', 'depth'] else None
                    grp.create_dataset(sname, data=np.stack(d), compression=comp)

    print("Done! All data saved.")

# if __name__ == "__main__":

#     main(DEMO_PATH, OUTPUT_PATH)


# if __name__ == "__main__":
#     from pathlib import Path
#     root_path = Path("/home/ubuntu/cwb_works/project/galbot_real_world_cwb/data/202512_01")
#     for path in root_path.iterdir():

#         main(data_dir=path, output_root=OUTPUT_PATH)


if __name__ == "__main__":
    from pathlib import Path
    
    root_path = Path("/home/ubuntu/cwb_works/project/galbot_real_world_cwb/data_example")
    output_path = "/home/ubuntu/cwb_works/project/galbot_real_world_cwb/output_demo"

    # 使用 glob 直接筛选包含 SYNC.mcap 的文件
    for path in root_path.glob("*SYNC.mcap"):
        print(f"Processing: {path.name}")
        # 注意：建议转为 str，防止 main 函数内部不支持 Path 对象
        main(data_dir=str(path), output_root=output_path)
