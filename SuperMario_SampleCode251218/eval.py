import numpy as np
import torch
from tqdm import tqdm
import time
import cv2  # 用於錄製影片
import os

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils import preprocess_frame
from model import CustomCNN
from DQN import DQN

# ========== Config ===========
# 選擇最佳模型
# MODEL_PATH = os.path.join("liang_test_extreme", "step_1368_reward_106766.pth")
MODEL_PATH = os.path.join("liang_test","step_1417_reward_37291.pth")
# MODEL_PATH = os.path.join("ckpt_test", "step_1_reward_-441.pth")   


# 影片輸出設定
RECORD_VIDEO = True                     # 是否錄製影片
VIDEO_OUTPUT_PATH = "mario_eval.mp4"    # 輸出影片檔名
VIDEO_FPS = 30                          # 影片幀率

import gym
from gym.wrappers import StepAPICompatibility

# 1) make
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# 2) 拆掉 TimeLimit
if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env

# 3) 固定成舊 step API
env = StepAPICompatibility(env, output_truncation_bool=False)

# 4) 再包 JoypadSpace
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print("Final env:", env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBS_SHAPE = (1, 84, 84)
N_ACTIONS = len(SIMPLE_MOVEMENT) 

VISUALIZE = False                       # 關閉螢幕顯示（避免 pyglet 錯誤）
TOTAL_EPISODES = 1

# ========== Initialize DQN =========== 
dqn = DQN( 
    model=CustomCNN, 
    state_dim=OBS_SHAPE,
    action_dim=N_ACTIONS,
    learning_rate=0.0001,  
    gamma=0.99,          
    epsilon=0.0,
    target_update=1000,
    device=device
)

# ========== 載入模型權重 =========== 
if os.path.exists(MODEL_PATH):
    try:
        model_weights = torch.load(MODEL_PATH, map_location=device)
        dqn.q_net.load_state_dict(model_weights)
        dqn.q_net.eval()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        raise
else:
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# ========== 初始化影片錄製 ===========
video_writer = None
if RECORD_VIDEO:
    # 取得遊戲畫面尺寸
    sample_frame = env.reset()
    height, width = sample_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, VIDEO_FPS, (width, height))
    print(f"📹 Recording video to: {VIDEO_OUTPUT_PATH}")
    print(f"   Resolution: {width}x{height}, FPS: {VIDEO_FPS}")

# ========== Evaluation Loop ===========
for episode in range(1, TOTAL_EPISODES + 1):
    state = env.reset()
    
    # 錄製第一幀
    if RECORD_VIDEO and video_writer is not None:
        frame_rgb = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_rgb)
    
    state = preprocess_frame(state)
    state = np.expand_dims(state, axis=0)
    state = np.expand_dims(state, axis=0)
    
    done = False
    total_reward = 0
    frame_count = 0

    while not done:
        # Take action using the trained policy
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_probs = torch.softmax(dqn.q_net(state_tensor), dim=1)
            action = torch.argmax(action_probs, dim=1).item()
        next_state, reward, done, info = env.step(action)

        # 錄製每一幀
        if RECORD_VIDEO and video_writer is not None:
            frame_rgb = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_rgb)
            frame_count += 1

        # Preprocess next state
        next_state_processed = preprocess_frame(next_state)
        next_state_processed = np.expand_dims(next_state_processed, axis=0)
        next_state_processed = np.expand_dims(next_state_processed, axis=0)

        # Accumulate rewards
        total_reward += reward
        state = next_state_processed

        # 顯示進度
        if frame_count % 100 == 0:
            print(f"  Frame: {frame_count}, x_pos: {info.get('x_pos', 'N/A')}, Reward: {total_reward:.0f}")

    print(f"Episode {episode}/{TOTAL_EPISODES} - Total Reward: {total_reward} - Frames: {frame_count}")

# ========== 關閉資源 ===========
env.close()

if video_writer is not None:
    video_writer.release()
    print(f"✅ Video saved to: {VIDEO_OUTPUT_PATH}")
    print(f"   Total frames: {frame_count}")
