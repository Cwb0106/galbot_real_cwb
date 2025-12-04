import sys
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from tqdm import tqdm

def inspect_tf(mcap_path):
    print(f"正在检查 TF 数据: {mcap_path} ...")
    
    # 集合用于去重，避免重复打印
    known_transforms = set()
    
    # 我们只关心这两个 TF topic
    tf_topics = ["/embosa_tf", "/embosa_tf_static"]

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        
        count = 0
        for schema, channel, message, proto_msg in reader.iter_decoded_messages(topics=tf_topics):
            # TF 消息通常包含一个 transforms 列表
            # 结构通常是: msg.transforms[i].header.frame_id (父) -> msg.transforms[i].child_frame_id (子)
            
            if hasattr(proto_msg, 'transforms'):
                for t in proto_msg.transforms:
                    parent = t.header.frame_id
                    child = t.child_frame_id
                    
                    # 组合成字符串用于去重
                    pair_str = f"{parent} -> {child}"
                    
                    if pair_str not in known_transforms:
                        known_transforms.add(pair_str)
                        # 打印发现的新关系
                        print(f"[TF] 发现变换关系: {parent:<40} --> {child}")
            
            count += 1
            # 如果数据量太大，扫描前 10000 帧通常就足够发现所有静态/动态 frame 了
            if count > 10000:
                print("已扫描部分数据，停止扫描。")
                break

    print("\n" + "="*80)
    print("✅ 分析完成。请在上面的列表中寻找类似 'camera', 'optical', 'wrist', 'sensor' 的名字。")
    print("="*80)

if __name__ == "__main__":
    default_path = "/home/ubuntu/cwb_works/project/galbot/data_original/20251127_105419_record0.SYNC.mcap"
    mcap_file = sys.argv[1] if len(sys.argv) > 1 else default_path
    inspect_tf(mcap_file)