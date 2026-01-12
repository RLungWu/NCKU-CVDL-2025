import os
import numpy as np
import random
import torch
import torch.nn as nn
import cv2
import time
from tqdm import tqdm

import gym_super_mario_bros                                      #導入gym_super_mario_bros，這是一個基於 Gym 的模組，用於模擬《Super Mario Bros》遊戲環境。
from nes_py.wrappers import JoypadSpace                          #從nes_py中導入JoypadSpace，用於限制遊戲中可用的按鈕動作（例如僅允許「移動右」或「跳躍」的動作集合）。
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT         #從 gym_super_mario_bros中導入SIMPLE_MOVEMENT，這是一個預定義的按鈕動作集合（如「右移」、「跳躍」等），用於控制 Mario 的行為。
                                                                 #簡化動作空間 NES 控制器有 8 個按鍵（上下左右、A、B、Select、Start），可能的按鍵組合數非常大

from utils import preprocess_frame                               #用於對遊戲的畫面進行預處理，例如灰階化、調整大小等，將其轉換為適合神經網路輸入的格式
from reward import *  
from reward import EXTREME_MODE                                           #模組中導入所有函式，這些函式用於設計和計算自定義獎勵（例如根據 Mario 的硬幣數量、水平位移等來計算獎勵）。
from model import CustomCNN                                      #自定義的卷積神經網路模型，用於處理遊戲畫面並生成動作決策
from DQN import DQN, ReplayMemory                                #用於執行強化學習的主要邏輯 DQN模組中導入回放記憶體，用於存儲和抽取遊戲的狀態、動作、獎勵等樣本，提升訓練穩定性。



# ========== config ===========
#env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')   #
#env = JoypadSpace(env, SIMPLE_MOVEMENT)
import gym
from gym.wrappers import StepAPICompatibility

# 1) make（這裡可能會自動包 TimeLimit）
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# 2) 🔑 拆掉 TimeLimit（不拆一定炸 expected 5 got 4）
if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env

# 3) 固定成舊 step API（回 4-tuple）
env = StepAPICompatibility(env, output_truncation_bool=False)

# 4) 再包 JoypadSpace
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print("Final env:", env)

#========= basic train config==============================================
LR = 0.005                    
BATCH_SIZE = 64                 # 批次大小
GAMMA = 0.99                    # 提高！更重視長期獎勵（學會跳過敵人）
MEMORY_SIZE = 50000             # 記憶體大小
EPSILON_START = 1.0             # 新增：初始探索率 100%
EPSILON_END = 0.1               # 降低：最終探索率 10%
EPSILON_DECAY = 0.995           # 新增：每回合探索率衰減
TARGET_UPDATE = 100             # 目標網路更新頻率
TOTAL_TIMESTEPS = 10            # 訓練回合數（錄影設 10）
VISUALIZE = True               # 是否渲染遊戲畫面
MAX_STAGNATION_STEPS = 300      # 停滯步數上限
device = torch.device("cuda")

# 加速訓練設定
FRAME_SKIP = 2                  # 讓 Mario 有更多反應時間跳過敵人
TRAIN_FREQUENCY = 4             # 每 N 步訓練一次
RENDER_DELAY = 0.02             # 渲染延遲（秒），設 0 = 最快，0.02 = 正常速度，0.05 = 慢速

# 影片錄製設定
RECORD_VIDEO = True                                         # 是否錄製影片
VIDEO_FPS = 30                                              # 影片幀率
VIDEO_DIR = "videos"                                        # 影片儲存目錄
os.makedirs(VIDEO_DIR, exist_ok=True)                       # 建立目錄
VIDEO_OUTPUT_PATH = os.path.join(VIDEO_DIR, f"mario_train_{'extreme' if EXTREME_MODE else 'normal'}.mp4")





# ========================DQN Initialization==========================================
obs_shape = (1, 84, 84)                         #obs_shape = (1, 84, 84)
n_actions = len(SIMPLE_MOVEMENT)                #定義動作空間大小，使用SIMPLE_MOVEMENT中的動作數量（例如向右移動、跳躍等）
model = CustomCNN                               #指定模型架構為CustomCNN用於處理圖像並預測各動作的 Q 值
dqn = DQN(                                      #初始化 DQN agent
    model=model,
    state_dim=obs_shape,                        #狀態空間大小
    action_dim=n_actions,                       #動作空間大小
    learning_rate=LR,                           #學習率
    gamma=GAMMA,                                #折扣因子，用於計算未來獎勵
    epsilon=EPSILON_START,                      # 使用初始探索率 1.0
    target_update=TARGET_UPDATE,                #目標網路更新頻率
    device=device
)

# ========== 載入預訓練權重 ==========
LOAD_PRETRAINED = True  # 設為 True 載入預訓練模型，False 從頭訓練
PRETRAINED_MODEL_PATH = os.path.join("liang_test_extreme", "step_1368_reward_106766.pth")

if LOAD_PRETRAINED and os.path.exists(PRETRAINED_MODEL_PATH):
    dqn.q_net.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
    dqn.tgt_q_net.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
    print(f"✅ Loaded pretrained model from: {PRETRAINED_MODEL_PATH}")
    
    # 載入預訓練模型時，降低探索率
    EPSILON_START = 0.3  # 從 30% 探索開始（因為已有經驗）
    current_epsilon = EPSILON_START
    dqn.epsilon = current_epsilon
    TOTAL_TIMESTEPS = 10
else:
    if LOAD_PRETRAINED:
        print(f"⚠️ Pretrained model not found: {PRETRAINED_MODEL_PATH}")
    print("🔄 Training from scratch")

memory = ReplayMemory(MEMORY_SIZE)              #創建經驗回放記憶體，用於存儲狀態轉移
step = 0                                        #記錄總步數
best_reward = -float('inf')                     # 儲存最佳累積獎勵
cumulative_reward = 0                           # 當前時間步的總累積獎勵
current_epsilon = EPSILON_START                 # 追蹤當前探索率

# ========== 初始化影片錄製 ===========
video_writer = None
total_video_frames = 0
if RECORD_VIDEO:
    # 取得遊戲畫面尺寸
    sample_frame = env.reset()
    height, width = sample_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, VIDEO_FPS, (width, height))
    print(f"🎬 錄製訓練影片: {VIDEO_OUTPUT_PATH}")
    print(f"   解析度: {width}x{height}, FPS: {VIDEO_FPS}")




#=======================訓練開始============================
for timestep in tqdm(range(1, TOTAL_TIMESTEPS + 1), desc="Training Progress"):  #主訓練迴圈，進行TOTAL_TIMESTEPS次迭代
    state = env.reset()                                                         #重置遊戲環境，獲取初始狀態
    state = preprocess_frame(state)                                             #使用preprocess_frame 將畫面處理為灰階、縮放為84x84
    state = np.expand_dims(state, axis=0)                                       #新增一個維度，適配模型輸入

    done = False                                                                #表示當前遊戲是否結束
    prev_info = {                                                               #用於追蹤遊戲狀態（如水平位置、得分、硬幣數量）
        "x_pos": 0,  # Starting horizontal position (int).
        "y_pos": 0,  # Starting vertical position (int).
        "score": 0,  # Initial score is 0 (int).
        "coins": 0,  # Initial number of collected coins is 0 (int).
        "time": 400,  # Initial time in most levels of Super Mario Bros is 400 (int).
        "flag_get": False,  # Player has not yet reached the end flag (bool).
        "life": 3  # Default initial number of lives is 3 (int).
    }

    cumulative_reward = 0 
    stagnation_time = 0                                                           #stagnation_time記錄遊戲角色在水平方向的停滯時間
    #開始一個回合的遊戲循環
    while not done:
        action = dqn.take_action(state)                                           #輸入目前狀態，交給DQN去做下一步
        
        # ⚡ Frame Skip: 重複執行同一動作 N 次，累積獎勵
        frame_reward = 0
        raw_frame = None  # 儲存原始畫面用於錄影
        for _ in range(FRAME_SKIP):
            next_state, reward, done, info = env.step(action)
            raw_frame = next_state.copy()  # 保存原始 RGB 畫面
            frame_reward += reward
            if done:
                break
        reward = frame_reward  # 使用累積的獎勵
       
        # preprocess image state 將下一狀態進行預處理並調整為適合模型的形狀
        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)

        cumulative_reward += final_reward(info, reward, prev_info)   #更新累積獎勵


        # ===========================Check for x_pos stagnation  如果角色的水平位置未改變超過MAX_STAGNATION_STEPS則強制結束本局遊戲
        if info["x_pos"] == prev_info["x_pos"]:
            stagnation_time += 1
            if stagnation_time >= MAX_STAGNATION_STEPS:
                print(f"Timestep {timestep} - Early stop triggered due to x_pos stagnation.")
                done = True
        else:
            stagnation_time = 0
        
        
        #===========================Store transition in memory 將狀態轉移 (state, action, reward, next_state, done) 存入記憶體
        memory.push(state, action, cumulative_reward //1, next_state, done)      #使用自訂義獎勵
        #memory.push(state, action, final_reward(info, reward, prev_info), next_state, done)                  #使用其預設好的獎勵
        #更新當前狀態
        state = next_state

        #==============================Train DQN 當記憶體中樣本數量達到批次大小時，從記憶體中隨機抽取一批樣本進行網路更新
        # ⚡ 每 TRAIN_FREQUENCY 步才訓練一次，減少訓練開銷
        if len(memory) >= BATCH_SIZE and step % TRAIN_FREQUENCY == 0:
            batch = memory.sample(BATCH_SIZE)

            state_dict = {                                       #將這些數據打包為字典格式，方便傳遞給模型進行訓練
                'states': batch[0],
                'actions': batch[1],
                'rewards': batch[2],
                'next_states': batch[3],
                'dones': batch[4],
            }
            dqn.train_per_step(state_dict)                       #train_per_step是DQN中的方法，用於計算損失並更新神經網路的權重

        #================================更新狀態訊息
        prev_info = info
        step += 1

        if VISUALIZE:                                   #渲染當前遊戲畫面
            env.render()
            time.sleep(RENDER_DELAY)                    # 延遲控制速度
        
        # 錄製影片幀
        if RECORD_VIDEO and video_writer is not None and raw_frame is not None:
            frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_rgb)
            total_video_frames += 1

    # ⚡ Epsilon Decay: 每回合結束後降低探索率
    current_epsilon = max(EPSILON_END, current_epsilon * EPSILON_DECAY)
    dqn.epsilon = current_epsilon

    # Print cumulative reward for the current timestep
    print(f"Timestep {timestep} - Reward: {cumulative_reward:.0f} - Epsilon: {current_epsilon:.3f}")

    #如果當前累積獎勵超過歷史最佳值，保存模型的權重 每次超過最佳值就會保留一次
    if cumulative_reward > best_reward:
        best_reward = cumulative_reward
        if EXTREME_MODE:
            os.makedirs("liang_test_extreme", exist_ok=True)
            #命名邏輯是採第幾步+最佳獎勵+自訂義獎勵的累積總合
            model_path = os.path.join("liang_test_extreme",f"step_{timestep}_reward_{int(best_reward)}.pth")
            torch.save(dqn.q_net.state_dict(), model_path)
            print(f"Model saved: {model_path}")
        else:
            os.makedirs("liang_test", exist_ok=True)
            #命名邏輯是採第幾步+最佳獎勵+自訂義獎勵的累積總合
            model_path = os.path.join("liang_test",f"step_{timestep}_reward_{int(best_reward)}.pth")
            torch.save(dqn.q_net.state_dict(), model_path)
            print(f"Model saved: {model_path}")

env.close()

# ========== 關閉影片錄製 ===========
if video_writer is not None:
    video_writer.release()
    print(f"✅ 訓練影片已儲存: {VIDEO_OUTPUT_PATH}")
    print(f"   總幀數: {total_video_frames}")

