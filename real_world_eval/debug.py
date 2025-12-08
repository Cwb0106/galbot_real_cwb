import sys
import time
import signal
import numpy as np
import time
from omegaconf import OmegaConf


# 假设 control.py 在同一目录下，或者你已经设置好了 PYTHONPATH
from control import GalbotControl


work_joint_position = np.array(
    [
        # odom
        0,  
        0, 
        0, 
        # -0.3,1.5,2.4,-2.4,-0.4,-0.85,0.3, # left arm
        2.75,
        -1.6,
        -0.9,
        -2.40,
        0.0,
        -0.8,
        0.0,  # left arm
        # 0.3,-1.5,-2.4,2.4,0.4,0.85,-0.3, # right arm
        -2.75,
        1.6,
        0.9,
        2.40,
        0.0,
        0.8,
        0.0,  # right arm
    ]
)

move_joint_position = np.array(
    [
        # odom
        0,
        0,
        0,
        # left_arm
        2.0,
        -1.59,
        -0.6,
        -1.7,
        -0.0,
        -0.8,
        0.0,  
        # right arm
        -2.0,
        1.59,
        0.6,
        1.7,
        0.0,
        0.8,
        0.0,  
    ]
)


def debug(gc):

    obs = gc.reset()
    
    print("finish reset!!")
    
    obs = gc.step(action=demo_action)
    
    print("finish step!!")

def test_execution(gc):
    action = dict()
    action['left_arm'] = work_joint_position[3:10]
    action['right_arm'] = work_joint_position[10:17]
    gc._galbot_execution(action)

    print("运动到初始位置")
    time.sleep(5)

    # 请注意，右臂没有归一化信息，需要考虑将归一化注释掉
    processed_action = gc._post_processing_action(work_joint_position)
    info = gc._galbot_execution(processed_action)

    print("完成第一个动作")
    time.sleep(5)

    processed_action = gc._post_processing_action(move_joint_position)
    info = gc._galbot_execution(processed_action)






if __name__ == "__main__":
    conf = OmegaConf.load('/home/ubuntu/cwb_works/project/galbot_real_world_cwb/real_world_eval/eval_config.yml')
    gc = GalbotControl(conf)
    test_execution(gc)







    