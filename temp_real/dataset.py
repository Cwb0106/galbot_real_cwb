import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# =========================================================
# 辅助函数：图像与数据处理
# =========================================================

def normalize_data(data, stats):
    """归一化到 [-1, 1]"""
    denom = stats['max'] - stats['min']
    denom[denom == 0] = 1.0 
    normalized = (data - stats['min']) / denom
    normalized = normalized * 2.0 - 1.0
    return normalized

def unnormalize_data(normalized_data, stats):
    """反归一化"""
    denom = stats['max'] - stats['min']
    denom[denom == 0] = 1.0 
    data = (normalized_data + 1.0) / 2.0
    data = data * denom + stats['min']
    return data

def process_image(img_numpy):
    """
    输入: H5 读取的 (H, W, C) uint8
    输出: (C, H, W) float32, range [-1, 1]
    """
    if img_numpy.dtype == np.uint8:
        img_numpy = img_numpy.astype(np.float32) / 255.0
    
    # 归一化到 [-1, 1]
    img_numpy = img_numpy * 2.0 - 1.0
    
    tensor = torch.from_numpy(img_numpy)
    # (H, W, C) -> (C, H, W)
    return tensor.permute(2, 0, 1)


# =========================================================
# 核心 Dataset 类
# =========================================================

class DPDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        obs_horizon=2,
        pred_horizon=16,
    ):
        self.dataset_path = dataset_path
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        
        # 1. 预扫描元数据和 Action 统计值
        print(f"Loading metadata from {dataset_path}...")
        
        # 我们在这里打开文件只为了读取轻量级数据，读完就关
        # 真正的图像读取留给 __getitem__
        with h5py.File(dataset_path, "r") as f:
            # 读取轨迹索引信息
            # meta/traj_indices: [[start, end], [start, end], ...]
            if 'meta/traj_indices' not in f:
                raise ValueError("Metadata 'meta/traj_indices' not found. Please regenerate H5 with the new script.")
            
            self.traj_indices = f['meta/traj_indices'][:]
            
            # --- 动态计算 Action/State 统计值 ---
            # 为了计算 min/max，我们需要把所有 action 读入内存
            # Action 数据通常很小 (e.g. 50MB)，可以接受
            print("Computing statistics...")
            
            # 这里的键名需要和你 extract_data_single_h5.py 写入的一致
            # 假设我们需要拼装 action = [action_delta, joint_pos, gripper]

            base_act = f['action']['action_delta'][:]
            arm_joint = f['action']['state_left_arm_joint_position'][:]
            gripper = f['action']['state_left_arm_gripper_width'][:]
            
            # 维度对齐
            if base_act.ndim == 1: base_act = base_act[:, None]
            if arm_joint.ndim == 1: arm_joint = arm_joint[:, None]
            if gripper.ndim == 1: gripper = gripper[:, None]
            
            # 拼接完整的 Action 向量
            self.global_actions = np.concatenate([base_act, arm_joint, gripper], axis=-1)
            
            # 假设 State 和 Action 是同样的数据源
            self.global_states = self.global_actions.copy()
            
        # 计算统计值
        self.stats = {
            "action": {
                "min": np.min(self.global_actions, axis=0),
                "max": np.max(self.global_actions, axis=0)
            },
            "state": {
                "min": np.min(self.global_states, axis=0),
                "max": np.max(self.global_states, axis=0)
            }
        }
        
        # 2. 生成采样索引 (Slices)
        # 将每个轨迹拆分为多个 (traj_idx, start_frame_relative_to_traj)
        self.slices = []
        for i, (t_start, t_end) in enumerate(self.traj_indices):
            traj_len = t_end - t_start
            
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            
            # 生成该轨迹下的所有合法切片起始点
            for rel_start in range(-pad_before, traj_len - pred_horizon + pad_after):
                self.slices.append((i, rel_start))

        print(f"Total trajectories: {len(self.traj_indices)}")
        print(f"Total samples: {len(self.slices)}")
        
        # 初始化文件句柄为 None (用于多进程 Lazy Loading)
        self.f = None
        
        # 相机映射 (你可以根据 config_pcd.py 修改这里)
        self.camera_map = {
            "head_rgb": "front_head_left",  # 输出名: H5内部名
            "wrist_rgb": "left_wrist"
        }

    def get_stats(self):
        return self.stats

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        """
        核心懒加载逻辑：
        1. 检查文件是否打开，没打开则打开。
        2. 根据索引计算绝对位置。
        3. 仅从硬盘读取需要的片段。
        """
        if self.f is None:
            # swmr=True 允许在写入时读取，libver='latest' 提高性能
            self.f = h5py.File(self.dataset_path, "r", swmr=True, libver='latest')

        # 1. 获取索引信息
        traj_idx, rel_start = self.slices[idx]
        t_start_abs, t_end_abs = self.traj_indices[traj_idx] # 轨迹在整个大数组中的绝对起止点
        traj_len = t_end_abs - t_start_abs

        # ==========================
        # 2. 读取 Observation (Images)
        # ==========================
        obs_seq = {}
        
        # 计算相对切片范围
        rel_obs_start = rel_start
        rel_obs_end = rel_start + self.obs_horizon
        
        # 计算合法的读取范围 (处理边界 Padding)
        read_rel_start = max(0, rel_obs_start)
        read_rel_end = min(traj_len, rel_obs_end)
        
        # 转换为绝对地址
        read_abs_start = t_start_abs + read_rel_start
        read_abs_end = t_start_abs + read_rel_end
        
        # A. 读取图像
        for out_key, h5_key in self.camera_map.items():
            if h5_key in self.f['image']:
                # 只读取这一小段 [read_abs_start : read_abs_end]
                # 这时候才发生硬盘 IO
                raw_imgs = self.f['image'][h5_key][read_abs_start:read_abs_end]
                
                # 处理图像 (uint8 -> float32 norm)
                processed_list = [process_image(img) for img in raw_imgs]
                
                if len(processed_list) > 0:
                    img_seq = torch.stack(processed_list)
                else:
                    img_seq = torch.zeros((0, 3, 128, 128))
            else:
                # 如果缺少某个相机数据，返回全黑占位符
                valid_len = read_abs_end - read_abs_start
                img_seq = torch.zeros((valid_len, 3, 128, 128))

            # Padding (处理头部越界)
            if rel_obs_start < 0:
                pad_len = abs(rel_obs_start)
                if img_seq.shape[0] > 0:
                    first = img_seq[0].unsqueeze(0)
                    img_seq = torch.cat([first.repeat(pad_len, 1, 1, 1), img_seq], dim=0)
                else:
                    img_seq = torch.zeros((pad_len, 3, 128, 128))

            # Padding (处理尾部越界)
            if img_seq.shape[0] < self.obs_horizon:
                pad_len = self.obs_horizon - img_seq.shape[0]
                if img_seq.shape[0] > 0:
                    last = img_seq[-1].unsqueeze(0)
                    img_seq = torch.cat([img_seq, last.repeat(pad_len, 1, 1, 1)], dim=0)
                else:
                    remaining = self.obs_horizon - img_seq.shape[0]
                    img_seq = torch.cat([img_seq, torch.zeros((remaining, 3, 128, 128))], dim=0)
            
            obs_seq[out_key] = img_seq

        # B. 读取 State (从内存缓存中读，速度快)
        # State 切片逻辑同 Observation
        raw_state_seq = self.global_states[read_abs_start:read_abs_end]
        norm_state_seq = normalize_data(raw_state_seq, self.stats['state'])
        norm_state_seq = torch.from_numpy(norm_state_seq).float()
        
        # State Padding
        if rel_obs_start < 0:
            pad_len = abs(rel_obs_start)
            first = norm_state_seq[0].unsqueeze(0)
            norm_state_seq = torch.cat([first.repeat(pad_len, 1), norm_state_seq], dim=0)
            
        if norm_state_seq.shape[0] < self.obs_horizon:
            pad_len = self.obs_horizon - norm_state_seq.shape[0]
            last = norm_state_seq[-1].unsqueeze(0)
            norm_state_seq = torch.cat([norm_state_seq, last.repeat(pad_len, 1)], dim=0)
            
        obs_seq['state'] = norm_state_seq

        # ==========================
        # 3. 读取 Action (Prediction)
        # ==========================
        rel_act_start = rel_start
        rel_act_end = rel_start + self.pred_horizon
        
        read_act_rel_start = max(0, rel_act_start)
        read_act_rel_end = min(traj_len, rel_act_end)
        
        read_act_abs_start = t_start_abs + read_act_rel_start
        read_act_abs_end = t_start_abs + read_act_rel_end
        
        # 从内存缓存读取
        raw_act_seq = self.global_actions[read_act_abs_start:read_act_abs_end]
        norm_act_seq = normalize_data(raw_act_seq, self.stats['action'])
        norm_act_seq = torch.from_numpy(norm_act_seq).float()
        
        # Action Padding
        if rel_act_start < 0:
            pad_len = abs(rel_act_start)
            first = norm_act_seq[0].unsqueeze(0)
            norm_act_seq = torch.cat([first.repeat(pad_len, 1), norm_act_seq], dim=0)
            
        if norm_act_seq.shape[0] < self.pred_horizon:
            pad_len = self.pred_horizon - norm_act_seq.shape[0]
            last = norm_act_seq[-1].unsqueeze(0)
            norm_act_seq = torch.cat([norm_act_seq, last.repeat(pad_len, 1)], dim=0)

        return {
            "observations": obs_seq,
            "actions": norm_act_seq,
            "state": norm_state_seq
        }

if __name__ == "__main__":
    import time
    
    data_path = "/home/ubuntu/cwb_works/project/galbot_real_world_cwb/output_demo/combined_dataset_20251201.h5"
    
    print("Initializing Dataset...")
    start_t = time.time()
    
    dataset = DPDataset(data_path, obs_horizon=2, pred_horizon=16)
    
    print(f"Init done in {time.time() - start_t:.2f}s")

    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    print("Starting iteration...")
    start_t = time.time()
    
    for i, batch in enumerate(loader):
        if i == 0:
            print("Batch 0 Loaded!")
            print("Action shape:", batch['actions'].shape)
            print("Img shape:", batch['observations']['head_rgb'].shape)
        
        if i >= 10: 
            break
            
    print(f"Read 10 batches in {time.time() - start_t:.2f}s")