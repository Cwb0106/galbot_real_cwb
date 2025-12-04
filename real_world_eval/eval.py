import sys
import time
import signal

# 假设 control.py 在同一目录下，或者你已经设置好了 PYTHONPATH
from control import GalbotControl

# ---------------------------------------------------------
# 模拟配置类 (如果你有真实的 config 文件加载逻辑，请替换这里)
# ---------------------------------------------------------
class DummyConfig:
    def __init__(self):
        class SensorConfig:
            camera_list = ["head_rgb", "wrist_rgb"] # 示例配置
        self.sensor = SensorConfig()

# ---------------------------------------------------------
# 核心评测类
# ---------------------------------------------------------
class Evaluation:
    def __init__(self, cfgs):
        print("[System] Initializing Galbot Environment...")
        self.galbot_env = GalbotControl(cfgs)
        
        # TODO: 在这里加载你的模型 (Policy)
        # self.policy = load_model("path/to/checkpoint.ckpt")
        print("[System] Model loaded (Placeholder).")

    def eval_one_episode(self, max_steps=300):
        print(f"\n{'='*20} Start Episode {'='*20}")
        
        # 1. 重置环境
        print("[Galbot] Resetting robot to home position...")
        obs = self.galbot_env.reset()
        
        # 2. 推理循环
        for step in range(max_steps):
            # -------------------------------------------------------
            # TODO: 这里放入你的模型推理代码
            # action = self.policy(obs)
            # -------------------------------------------------------
            
            # 这是一个测试用的伪造 action (全0)
            # 实际真机测试时，请务必小心，确保 action 是安全的！
            dummy_action = [0.0] * 7 
            
            # print(f"[Step {step}] Executing action...") # 调试用，嫌吵可以注释
            
            # 3. 执行动作并获取新的观测
            obs = self.galbot_env.step(dummy_action)
            
            # TODO: 添加终止条件判断 (比如任务完成或出错)
            # if done: break
            
            # 简单的频率控制 (可选，模拟控制频率)
            # time.sleep(0.1)

        print(f"{'='*20} Episode Finished {'='*20}\n")

    def init_system(self):
        # 可以在这里做一些额外的硬件预热或检查
        pass

# ---------------------------------------------------------
# 交互式主逻辑
# ---------------------------------------------------------
def main():
    # 1. 加载配置 (根据你的实际情况修改)
    # cfgs = OmegaConf.load("config.yaml") 
    cfgs = DummyConfig()

    # 2. 初始化评测器
    evaluator = Evaluation(cfgs)
    evaluator.init_system()

    print("\n" + "*"*50)
    print("Galbot Real-World Evaluation System Ready")
    print("*"*50)
    print("Commands:")
    print("  [Enter] : Run single episode (Start Evaluation)")
    print("  q       : Quit system")
    print("*"*50)

    # 3. 进入交互式循环
    try:
        while True:
            # 等待用户输入
            user_input = input("\n>>> Waiting for command (Press Enter to start, 'q' to quit): ").strip().lower()

            if user_input == 'q':
                print("[System] Exiting...")
                break
            
            elif user_input == '':
                # 空字符串代表直接按了回车，开始执行
                try:
                    evaluator.eval_one_episode()
                except Exception as e:
                    print(f"[Error] Episode failed: {e}")
                    # 真机调试时建议打印完整的 traceback
                    import traceback
                    traceback.print_exc()
            else:
                print("[System] Unknown command. Please press Enter or 'q'.")

    except KeyboardInterrupt:
        # 捕获 Ctrl+C，安全退出
        print("\n[System] Interrupted by user. Exiting safely...")

if __name__ == "__main__":
    main()







    