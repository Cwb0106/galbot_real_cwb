# data_utils.py
import numpy as np
import cv2
import time
from tqdm import tqdm
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from google.protobuf.json_format import MessageToDict
from config import IMG_WIDTH, IMG_HEIGHT, VIDEO_KEYS

class MultiKeySequenceAligner:
    def __init__(self, base_key, base_timestamps, base_data):
        self.base_key = base_key
        self.base_timestamps = np.array(base_timestamps)
        self.base_data = base_data 
        self.sequences = {}
        self.behind_flags = {}
        self.video_key_set = set(VIDEO_KEYS)

    def add_sequence(self, key, timestamps, data, behind_flag=False):
        self.sequences[key] = (np.array(timestamps), data)
        self.behind_flags[key] = behind_flag

    def align(self):
        # --- 诊断信息 ---
        tqdm.write(f"\n[DEBUG] Video Keys defined in Config: {self.video_key_set}")
        tqdm.write(f"[DEBUG] Base Key is: {self.base_key}")
        
        # 1. 处理 Base Key
        start_time = time.time()
        if self.base_key in self.video_key_set:
            tqdm.write(f"[DEBUG] Base Key '{self.base_key}' matched video keys. Keeping as List.")
            aligned = {self.base_key: self.base_data} 
        else:
            tqdm.write(f"[DEBUG] Base Key '{self.base_key}' NOT matched. Converting to Numpy (Might be slow)...")
            aligned = {self.base_key: np.array(self.base_data)}
        tqdm.write(f"[DEBUG] Base Key processed in {time.time() - start_time:.2f}s")
             
        aligned_timestamps = {self.base_key: self.base_timestamps}
        
        # 2. 处理其他 Topics
        # 使用 tqdm 包装，并确保 leave=True
        for key, (timestamps, data) in tqdm(self.sequences.items(), desc="Aligning topics", leave=True):
            if key == "state_left_arm_gripper_width":
                print(1)
            # --- 诊断打印：开始处理某个 Key ---
            # 只有当数据是图片（3维数组）时，转换 Numpy 才是危险的
            is_image_data = (len(data) > 0 and isinstance(data[0], np.ndarray) and data[0].ndim == 3)
            is_video_key = key in self.video_key_set
            
            status_msg = f"Processing '{key}' | Is Image Data: {is_image_data} | Is Configured Video: {is_video_key}"
            # tqdm.write(status_msg) # 如果不想刷屏可以注释掉这行

            aligned_data_list = []
            aligned_ts_list = []
            ts_len = len(timestamps)
            curr_idx = 0
            
            # 查找对齐
            for base_t in self.base_timestamps:
                while curr_idx < ts_len - 1:
                    diff_curr = abs(timestamps[curr_idx] - base_t)
                    diff_next = abs(timestamps[curr_idx+1] - base_t)
                    if diff_next < diff_curr:
                        curr_idx += 1
                    else:
                        break
                
                aligned_data_list.append(data[curr_idx])
                aligned_ts_list.append(timestamps[curr_idx])
            
            # --- 关键逻辑 ---
            if is_video_key:
                # 这种情况下绝对安全
                aligned[key] = aligned_data_list
            elif is_image_data:
                # [危险] 它是图片数据，但没有在 VIDEO_KEYS 里定义！
                tqdm.write(f"\n[WARNING] Key '{key}' looks like IMAGE data but is NOT in VIDEO_KEYS!")
                tqdm.write(f"[WARNING] Force converting {len(aligned_data_list)} images to Numpy... This causes the FREEZE.")
                aligned[key] = np.array(aligned_data_list) # <--- 这里就是卡死的地方
                tqdm.write(f"[INFO] '{key}' conversion done.")
            else:
                # 普通数据 (Joints, Odom)，转 Numpy 很快
                aligned[key] = np.array(aligned_data_list)
                
            aligned_timestamps[key] = np.array(aligned_ts_list)
            
        return aligned, aligned_timestamps

def parse_image_msg(proto_msg):
    try:
        img = None
        if hasattr(proto_msg, 'format') and proto_msg.format:
            if '16UC1' in proto_msg.format:
                dtype = np.uint16
                temp_img = np.frombuffer(proto_msg.data, dtype=dtype)
                if len(temp_img) == IMG_WIDTH * IMG_HEIGHT:
                    img = temp_img.reshape((IMG_HEIGHT, IMG_WIDTH))
            else:
                np_arr = np.frombuffer(proto_msg.data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        elif hasattr(proto_msg, 'width') and hasattr(proto_msg, 'height'):
            width = proto_msg.width
            height = proto_msg.height
            encoding = proto_msg.encoding
            data = proto_msg.data
            
            dtype = np.uint8
            if '16UC1' in encoding: dtype = np.uint16
            temp_img = np.frombuffer(data, dtype=dtype)
            
            if 'rgb8' in encoding:
                img = temp_img.reshape((height, width, 3))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif 'bgr8' in encoding:
                img = temp_img.reshape((height, width, 3))
            elif 'mono8' in encoding:
                img = temp_img.reshape((height, width))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img is not None:
            h, w = img.shape[:2]
            if h != IMG_HEIGHT or w != IMG_WIDTH:
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        
        return img
    except Exception:
        return None

def get_timestamp_from_header(header):
    sec = header.timestamp.sec
    nanosec = header.timestamp.nanosec
    return float(sec) + float(nanosec) / 1e9

def get_episode_from_mcap_new(mcap_path, config, data_args):
    print(f"Processing MCAP: {mcap_path}")
    
    raw_data = {"timestamps": {}, "data": {}}
    for cam_key in config["camera_map"]:
        raw_data["timestamps"][cam_key] = []
        raw_data["data"][cam_key] = []
    
    raw_data["timestamps"]["odom"] = []
    raw_data["data"]["odom"] = []
    
    wbcs_buffer = { "timestamps": [], "raw_msgs": [] }

    topics_to_read = list(config["camera_map"].values()) + list(config["state_map"].values())
    topic_to_key = {v: k for k, v in config["camera_map"].items()}
    topic_to_key[config["state_map"]["odom"]] = "odom"
    topic_to_key[config["state_map"]["wbcs"]] = "wbcs"

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        
        for schema, channel, message, proto_msg in reader.iter_decoded_messages(topics=topics_to_read):
            topic = channel.topic
            key = topic_to_key.get(topic)
            if not key: continue

            if key in config["camera_map"]:
                ts = get_timestamp_from_header(proto_msg.header)
                img = parse_image_msg(proto_msg)
                if img is not None:
                    raw_data["timestamps"][key].append(ts)
                    raw_data["data"][key].append(img)

            elif key == "odom":
                ts = get_timestamp_from_header(proto_msg.header)
                pos = proto_msg.pose.pose.position
                quat = proto_msg.pose.pose.orientation
                odom_vec = [pos.x, pos.y, quat.x, quat.y, quat.z, quat.w]
                raw_data["timestamps"]["odom"].append(ts)
                raw_data["data"]["odom"].append(odom_vec)

            elif key == "wbcs":
                ts = get_timestamp_from_header(proto_msg.header)
                wbcs_buffer["timestamps"].append(ts)
                wbcs_buffer["raw_msgs"].append(proto_msg)

    print(f"Parsing WBCS joint data ({len(wbcs_buffer['raw_msgs'])} msgs)...")
    for internal_key in data_args["joint_map"].keys():
        raw_data["timestamps"][internal_key] = []
        raw_data["data"][internal_key] = []

    for i, proto_msg in tqdm(enumerate(wbcs_buffer["raw_msgs"]), total=len(wbcs_buffer["raw_msgs"]), desc="Parsing Joints"):
        ts = wbcs_buffer["timestamps"][i]
        msg_dict = MessageToDict(proto_msg, preserving_proto_field_name=True)
        sensor_map = msg_dict.get('joint_sensor_map', {}) 
        
        for internal_key, map_key in data_args["joint_map"].items():
            if map_key in sensor_map:
                pos = sensor_map[map_key].get('position', [])
                raw_data["timestamps"][internal_key].append(ts)
                raw_data["data"][internal_key].append(pos)
    
    base_cam = list(config["camera_map"].keys())[0]
    if len(raw_data['timestamps'][base_cam]) == 0:
        print(f"Warning: No frames found for base camera {base_cam}")
        return None, None

    print(f"Aligning data based on camera: {base_cam} ({len(raw_data['timestamps'][base_cam])} frames)")
    
    aligner = MultiKeySequenceAligner(
        base_key=base_cam,
        base_timestamps=raw_data["timestamps"][base_cam],
        base_data=raw_data["data"][base_cam]
    )
    
    for cam_key in config["camera_map"].keys():
        if cam_key == base_cam: continue
        if len(raw_data["timestamps"][cam_key]) > 0:
            aligner.add_sequence(cam_key, raw_data["timestamps"][cam_key], raw_data["data"][cam_key])
    
    if len(raw_data["timestamps"]["odom"]) > 0:
        aligner.add_sequence("odom", raw_data["timestamps"]["odom"], raw_data["data"]["odom"])
        
    for internal_key in data_args["joint_map"].keys():
        if len(raw_data["timestamps"][internal_key]) > 0:
            aligner.add_sequence(internal_key, raw_data["timestamps"][internal_key], raw_data["data"][internal_key])
        else:
            print(f"Warning: No data found for joint key: {internal_key}")

    # [新增] 手动释放大对象，防止 return 时卡顿
    print("  -> Clearing buffers to free memory...")
    del wbcs_buffer
    del raw_data
    # 强制垃圾回收
    import gc
    gc.collect()
    
    print("  -> Returning aligned data...")
    return aligner.align()