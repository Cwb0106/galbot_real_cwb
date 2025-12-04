import open3d as o3d
import numpy as np
import cv2
import os
import argparse
import sys

# === 配置 ===
DEPTH_SCALE = 1000.0   # 深度图单位换算 (mm -> m)
MAX_DEPTH = 7.0        # 最大显示距离 (米)
MIN_DEPTH = 0.05       # 最小显示距离 (米)

class TaskViewer:
    def __init__(self, task_dir, sensor_name):
        self.task_dir = task_dir
        self.sensor_name = sensor_name
        
        # 路径设置
        self.rgb_dir = os.path.join(task_dir, sensor_name, "rgb")
        self.depth_dir = os.path.join(task_dir, sensor_name, "depth")
        self.npz_path = os.path.join(task_dir, "trajectory.npz")
        
        # 1. 检查路径
        if not os.path.exists(self.rgb_dir) or not os.path.exists(self.depth_dir):
            print(f"[Error] 图片文件夹不存在: {self.rgb_dir}")
            sys.exit(1)
        if not os.path.exists(self.npz_path):
            print(f"[Error] NPZ文件不存在: {self.npz_path}")
            sys.exit(1)
            
        # 2. 加载元数据 (trajectory.npz)
        print(f"[-] Loading metadata from {self.npz_path}...")
        self.traj_data = np.load(self.npz_path)
        
        # 检查 Keys
        self.intr_key = f'intrinsic_{sensor_name}'
        self.extr_key = f'extrinsic_{sensor_name}' # 应该是 T_camera_to_base
        
        if self.intr_key not in self.traj_data:
            print(f"[Error] 找不到相机内参: {self.intr_key}")
            print(f"可用 Keys: {list(self.traj_data.keys())}")
            sys.exit(1)
            
        self.total_frames = len(self.traj_data['timestamps'])
        print(f"[-] Found {self.total_frames} frames.")

    def load_frame(self, index):
        """读取指定帧的图像和矩阵"""
        img_name = f"{index:06d}.png"
        
        # A. 读取图像
        rgb_path = os.path.join(self.rgb_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)
        
        if not os.path.exists(rgb_path): return None
        
        # RGB (OpenCV读取为BGR，需转RGB)
        color = cv2.imread(rgb_path)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        
        # Depth (读取16位原始数据)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # B. 获取矩阵
        K = self.traj_data[self.intr_key][index]
        T = self.traj_data[self.extr_key][index]
        
        return color, depth, K, T

def update_geometry(vis, pcd, color, depth, K, T):
    """生成点云并更新到 Visualizer"""
    h, w, _ = color.shape
    
    # 1. 深度预处理 (实现 MIN_DEPTH 和单位转换的准备)
    # 将深度图转换为 float32 以便处理
    depth_f = depth.astype(np.float32)
    
    # 简单的掩码处理：小于最小距离的点设为0 (Open3D会忽略0)
    # 注意：DEPTH_SCALE 在 create_from_rgbd_image 中使用，这里只需处理阈值
    # MIN_DEPTH * DEPTH_SCALE 把米转回毫米进行比较
    mask_min = depth_f < (MIN_DEPTH * DEPTH_SCALE)
    depth_f[mask_min] = 0
    
    # 重新转回 uint16 或直接使用 float (Open3D Image 支持 float32)
    # 这里我们保持 float32 并创建 Open3D Image
    o3d_depth = o3d.geometry.Image(depth_f)
    o3d_color = o3d.geometry.Image(color)
    
    # 2. 创建 RGBD
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        depth_scale=DEPTH_SCALE, 
        depth_trunc=MAX_DEPTH, 
        convert_rgb_to_intensity=False
    )
    
    # 3. 创建内参对象
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0,0], K[1,1], K[0,2], K[1,2])
    
    # 4. 生成临时点云
    temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    if not temp_pcd.has_points():
        print("[Warn] Empty point cloud.")
        return

    # 5. 坐标变换 (Camera -> Base)
    # 假设 T 是 T_camera_to_base
    temp_pcd.transform(T)
    
    # 6. 更新全局点云数据
    # 注意：必须更新 pcd.points 而不是替换 pcd 对象，否则 visualizer 会丢失引用
    pcd.points = temp_pcd.points
    pcd.colors = temp_pcd.colors
    
    # 7. 通知 Visualizer 更新
    vis.update_geometry(pcd)
    
    # [关键修复] 强制设置 ViewControl 防止远处物体消失
    ctr = vis.get_view_control()
    ctr.set_constant_z_far(10000.0)  # 看得很远
    ctr.set_constant_z_near(0.001)   # 看得很近

def main(task_dir, sensor_name):
    # 初始化数据加载器
    viewer_backend = TaskViewer(task_dir, sensor_name)
    
    # --- Open3D 初始化 ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Viewer: {sensor_name}", width=1280, height=720)
    
    # 创建点云容器
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    # 添加坐标轴 (原点)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(axis)

    # 状态管理
    state = {
        "index": 0,
        "needs_update": True,
        "first_render": True
    }

    # --- 按键回调 ---
    def next_frame(vis):
        if state["index"] < viewer_backend.total_frames - 1:
            state["index"] += 1
            state["needs_update"] = True
            print(f"\r[>] Frame: {state['index']}", end="")
        return False

    def prev_frame(vis):
        if state["index"] > 0:
            state["index"] -= 1
            state["needs_update"] = True
            print(f"\r[<] Frame: {state['index']}", end="")
        return False
    
    def quit_app(vis):
        vis.close()
        return False

    # 注册按键
    # Open3D >= 0.16 建议使用 key_action，旧版本使用 ord
    vis.register_key_callback(ord("D"), next_frame) # D 下一帧
    vis.register_key_callback(ord("A"), prev_frame) # A 上一帧
    vis.register_key_callback(ord("Q"), quit_app)   # Q 退出
    
    # 为了兼容方向键 (Key Codes 可能因系统而异，这里保留 D/A 作为备用)
    vis.register_key_callback(262, next_frame) # Right Arrow
    vis.register_key_callback(263, prev_frame) # Left Arrow

    print("="*40)
    print(f"Task: {os.path.basename(task_dir)}")
    print(f"Sensor: {sensor_name}")
    print("-" * 30)
    print(" [D / →] : 下一帧")
    print(" [A / ←] : 上一帧")
    print(" [Q]     : 退出")
    print("="*40)

    # --- 主循环 ---
    while True:
        if state["needs_update"]:
            # 加载数据
            data = viewer_backend.load_frame(state["index"])
            if data:
                color, depth, K, T = data
                # 更新几何体
                update_geometry(vis, pcd, color, depth, K, T)
                
                # 只有第一帧重置视角，之后保持用户旋转的角度
                if state["first_render"]:
                    vis.reset_view_point(True)
                    state["first_render"] = False
            
            state["needs_update"] = False
        
        # 刷新渲染器
        vis.poll_events()
        vis.update_renderer()
        
        # 如果窗口关闭则退出
        if not vis.poll_events():
            break

    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认值修改为你刚才的路径
    parser.add_argument("--dir", type=str, default="/home/ubuntu/cwb_works/project/galbot_real_world/output/20251127_105419_record0.SYNC", help="Task directory path")
    parser.add_argument("--sensor", type=str, default="right_wrist", help="Sensor name")
    args = parser.parse_args()
    
    main(args.dir, args.sensor)