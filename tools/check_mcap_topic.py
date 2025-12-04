import sys
from mcap.reader import make_reader

def list_topics(mcap_path):
    print(f"正在读取文件: {mcap_path} ...")
    
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        
        print("\n" + "="*90)
        print(f"{'Topic Name':<50} | {'Message Encoding/Type'}")
        print("="*90)

        # 策略 1: 尝试从文件摘要中快速获取 (速度极快)
        if summary and summary.channels:
            # summary.channels 是一个字典 {channel_id: Channel}
            # 我们按 Topic 名字排序方便查看
            sorted_channels = sorted(summary.channels.values(), key=lambda x: x.topic)
            
            for channel in sorted_channels:
                print(f"{channel.topic:<50} | {channel.message_encoding}")
                
        # 策略 2: 如果文件没写摘要，则扫描前 5000 条消息来发现 Topic (兜底方案)
        else:
            print("⚠️ 未找到文件摘要，正在扫描前 5000 条消息来发现 Topic...")
            seen_topics = set()
            count = 0
            for schema, channel, message in reader.iter_messages():
                if channel.topic not in seen_topics:
                    print(f"{channel.topic:<50} | {channel.message_encoding}")
                    seen_topics.add(channel.topic)
                count += 1
                if count > 5000:
                    print(f"... (仅扫描了前 {count} 条消息)")
                    break

if __name__ == "__main__":
    # 如果没有命令行参数，请修改这里的默认路径
    default_path = "/home/ubuntu/cwb_works/project/galbot_real_world_cwb/data_original/20251201_151742_record0.SYNC.mcap"
    
    # 获取命令行参数或使用默认值
    mcap_file = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    list_topics(mcap_file)