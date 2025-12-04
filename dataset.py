import h5py
import numpy as np
import open3d as o3d

# === 配置 ===
H5_PATH = "/home/ubuntu/cwb_works/project/galbot_real_world/output/20251127_105419_record0.SYNC.h5"
TARGET_CAMERA = 'right_wrist'  # 只看右手
DEPTH_SCALE = 1000.0           # [关键] 毫米转米
MAX_DEPTH = 7.0                # 裁剪阈值 (米)
MIN_DEPTH = 0               # 最小距离 (米)

def get_frame_pcd(f, idx):
    """读取单帧，处理数据，返回 Open3D 点云对象"""
    
    # 1. 读取原始数据
    depth_raw = f['depth'][TARGET_CAMERA][idx]
    image = f['image'][TARGET_CAMERA][idx]
    K = f['intrinsic'][TARGET_CAMERA][idx]
    T = f['extrinsic'][TARGET_CAMERA][idx] # T_camera_to_base

    # 2. 颜色 BGR -> RGB & 归一化
    image = image[..., ::-1]
    colors = image.reshape(-1, 3).astype(np.float64) / 255.0

    # 3. [单位转换] & 反投影
    # mm -> m
    H, W = depth_raw.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    
    # 这里的除法完成了单位转换
    z = depth_raw.astype(np.float32) / DEPTH_SCALE
    
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    
    # 相机坐标系下的点 (N, 3)
    points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # 4. [裁剪] Crop
    mask = (points_cam[:, 2] > MIN_DEPTH) & (points_cam[:, 2] < MAX_DEPTH)
    points_cam = points_cam[mask]
    colors = colors[mask]

    if len(points_cam) == 0:
        return None

    # 5. 坐标转换 (Camera -> Base)
    ones = np.ones((points_cam.shape[0], 1))
    points_hom = np.hstack([points_cam, ones]) 
    points_base = (T @ points_hom.T).T[:, :3]

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_base)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def main():
    print(f"[-] Loading file: {H5_PATH}")
    
    # 创建一个通用的坐标轴 (原点)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

    with h5py.File(H5_PATH, 'r') as f:
        total_frames = len(f['timestamps'])
        print(f"Total Frames: {total_frames}")
        print("="*50)
        print("操作模式：")
        print("1. 窗口弹出后，你可以旋转/缩放查看点云。")
        print("2. 按键盘上的 'Q' 键，或者用鼠标关闭窗口。")
        print("3. 关闭后，程序会自动加载并弹出【下一帧】。")
        print("4. 在终端按 Ctrl+C 可以强行终止程序。")
        print("="*50)

        # 循环播放每一帧
        for i in range(total_frames):
            print(f"正在显示第 [{i}/{total_frames-1}] 帧... (请关闭窗口以继续)")
            
            pcd = get_frame_pcd(f, i)
            
            if pcd is None:
                print(f"Frame {i} 数据为空 (全被过滤了)，跳过。")
                continue
            
            # --- 核心可视化逻辑 ---
            # draw_geometries 是阻塞的，窗口不关，代码不往下走
            o3d.visualization.draw_geometries(
                [pcd, axis], 
                window_name=f"Frame {i} (Press Q for next)",
                width=1024, 
                height=768,
                left=50, top=50 # 固定窗口出现位置，避免乱跑
            )
            # --------------------

    print("所有帧已播放完毕。")

if __name__ == "__main__":
    main()