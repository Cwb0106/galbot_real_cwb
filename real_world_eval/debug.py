import sys
import time
import signal

# 假设 control.py 在同一目录下，或者你已经设置好了 PYTHONPATH
from control import GalbotControl


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







    