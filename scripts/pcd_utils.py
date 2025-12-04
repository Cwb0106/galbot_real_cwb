# pcd_utils.py
import numpy as np
import json
import cv2
from scipy.spatial.transform import Rotation as R

class SimpleTfTree:
    """
    一个不依赖 ROS 的轻量级 TF 树查询器。
    用于解析 /tf 和 /tf_static 并计算 extrinsic 矩阵。
    """
    def __init__(self):
        # 静态变换 (parent -> child -> matrix)
        self.static_transforms = {} 
        # 动态变换 (parent -> child -> list of (time, matrix))
        self.dynamic_transforms = {} 

    def _msg_to_matrix(self, transform):
        """把 TF 消息里的平移和旋转转成 4x4 矩阵"""
        tx = transform.translation.x
        ty = transform.translation.y
        tz = transform.translation.z
        qx = transform.rotation.x
        qy = transform.rotation.y
        qz = transform.rotation.z
        qw = transform.rotation.w
        
        mat = np.eye(4)
        mat[:3, 3] = [tx, ty, tz]
        mat[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        return mat

    def load_tf_message(self, proto_msg, is_static=False):
        """加载一条 TF 消息"""
        for t in proto_msg.transforms:
            parent = t.header.frame_id.strip('/')
            child = t.child_frame_id.strip('/')
            
            # 记录时间戳
            sec = t.header.timestamp.sec
            nanosec = t.header.timestamp.nanosec
            ts = float(sec) + float(nanosec) / 1e9
            
            mat = self._msg_to_matrix(t.transform)
            
            if is_static:
                if parent not in self.static_transforms: self.static_transforms[parent] = {}
                self.static_transforms[parent][child] = mat
            else:
                if parent not in self.dynamic_transforms: self.dynamic_transforms[parent] = {}
                if child not in self.dynamic_transforms[parent]: self.dynamic_transforms[parent][child] = []
                self.dynamic_transforms[parent][child].append((ts, mat))

    def get_transform(self, parent, child, query_time):
        """
        核心函数：查找从 child 到 parent 的变换矩阵 (T_parent_child)。
        简化版：假设是直接连接的，或者只查静态 TF。
        如果需要完整的多级链查找 (Chain Lookup)，逻辑会复杂很多。
        这里优先查静态，再查动态最近邻。
        """
        # 1. 查静态
        if parent in self.static_transforms and child in self.static_transforms[parent]:
            return self.static_transforms[parent][child]
        
        # 2. 查动态 (最近邻查找)
        if parent in self.dynamic_transforms and child in self.dynamic_transforms[parent]:
            transforms = self.dynamic_transforms[parent][child]
            # 找到时间戳最近的一个
            best_mat = None
            min_diff = float('inf')
            
            # 这里可以用二分查找优化，为了代码简单先用遍历
            for ts, mat in transforms:
                diff = abs(ts - query_time)
                if diff < min_diff:
                    min_diff = diff
                    best_mat = mat
                # 如果时间差开始变大，说明已经过了最近点（假设是按时间排序的）
                if diff > min_diff and diff > 1.0: 
                    break
            
            if min_diff > 0.5: # 如果最近的 TF 都在 0.5秒 以外，说明不同步
                print(f"[TF Warning] Time diff too large: {min_diff:.4f}s for {parent}->{child}")
            
            return best_mat

        return np.eye(4) # 没找到，返回单位矩阵 (即不变换)

def generate_point_cloud(depth_img, rgb_img, K, extrinsic_T=None, scale=1000.0, max_dist=2.0):
    """
    生成彩色点云。
    depth_img: HxW uint16
    rgb_img: HxW uint8 (BGR format)
    K: 3x3 内参矩阵
    extrinsic_T: 4x4 外参矩阵 (Camera -> World)
    """
    h, w = depth_img.shape
    
    # 1. 创建像素坐标网格 (u, v)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 2. 提取有效深度 (过滤掉 0 和 过远的点)
    depth_m = depth_img.astype(np.float32) / scale
    valid_mask = (depth_m > 0.1) & (depth_m < max_dist)
    
    z = depth_m[valid_mask]
    u = u[valid_mask]
    v = v[valid_mask]
    
    # 3. 反投影 (2D -> 3D Camera Frame)
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 堆叠成 (N, 3) 矩阵
    points = np.stack([x, y, z], axis=1)
    
    # 4. 获取对应的颜色
    if rgb_img is not None:
        # OpenCV 是 BGR，转 RGB
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        colors = rgb_img[valid_mask] / 255.0 # 归一化到 0-1
    else:
        colors = np.ones_like(points) # 默认白色
        
    # 5. 坐标变换 (Camera Frame -> Base/World Frame)
    if extrinsic_T is not None:
        # 齐次坐标变换: P_world = T * P_cam
        # 旋转 R * P + 平移 T
        R = extrinsic_T[:3, :3]
        t = extrinsic_T[:3, 3]
        points = (R @ points.T).T + t

    return points, colors

def save_ply(path, points, colors):
    """保存为 .ply 格式，可用 MeshLab 或 Open3D 打开"""
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""".format(len(points))

    with open(path, "w") as f:
        f.write(header)
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(np.uint8)
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")