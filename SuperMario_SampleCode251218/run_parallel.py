"""
平行訓練 Super Mario Bros DQN
使用多個環境同時收集經驗，加速訓練

16GB VRAM 建議使用 4-8 個平行環境
"""
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from multiprocessing import Process, Queue, Manager
import time

import gym
import gym_super_mario_bros
from gym.wrappers import StepAPICompatibility
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils import preprocess_frame
from reward import calculate_smart_reward, reset_max_x, reset_enemy_tracking, reset_hole_tracking
from model import CustomCNN
from DQN import DQN, ReplayMemory

# ========== 平行訓練配置 ===========
NUM_ENVS = 8                    # 平行環境數量 (根據 CPU 和 VRAM 調整)
LR = 0.0001                     # 學習率
BATCH_SIZE = 128                 # 批次大小 (多環境可以用更大的批次)
GAMMA = 0.99                    
MEMORY_SIZE = 100000         # 更大的記憶體
EPSILON_START = 1.0             # 從高探索開始
EPSILON_END = 0.1               # 最終探索率
EPSILON_DECAY = 0.9995          # 探索率衰減
TARGET_UPDATE = 100             
TOTAL_TIMESTEPS = 2000          # 總訓練回合
VISUALIZE = False               # 平行訓練時關閉渲染
MAX_STAGNATION_STEPS = 300      
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== 建立環境函數 ===========
def make_env():
    """建立一個 Mario 環境"""
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env
    env = StepAPICompatibility(env, output_truncation_bool=False)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

# ========== 環境工作進程 ===========
class EnvWorker:
    """管理多個平行環境"""
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [make_env() for _ in range(num_envs)]
        self.states = [None] * num_envs
        self.prev_infos = [None] * num_envs
        self.dones = [True] * num_envs
        
    def reset_all(self):
        """重置所有環境"""
        for i in range(self.num_envs):
            state = self.envs[i].reset()
            state = preprocess_frame(state)
            state = np.expand_dims(state, axis=0)
            self.states[i] = state
            self.prev_infos[i] = {
                "x_pos": 0, "y_pos": 0, "score": 0,
                "coins": 0, "time": 400, "flag_get": False, "life": 3
            }
            self.dones[i] = False
        return self.states.copy()
    
    def reset_env(self, idx):
        """重置單個環境"""
        state = self.envs[idx].reset()
        state = preprocess_frame(state)
        state = np.expand_dims(state, axis=0)
        self.states[idx] = state
        self.prev_infos[idx] = {
            "x_pos": 0, "y_pos": 0, "score": 0,
            "coins": 0, "time": 400, "flag_get": False, "life": 3
        }
        self.dones[idx] = False
        return state
    
    def step(self, actions):
        """在所有環境中執行動作"""
        results = []
        for i, action in enumerate(actions):
            if self.dones[i]:
                # 如果環境已結束，重置
                self.reset_env(i)
                results.append((self.states[i], 0, False, self.prev_infos[i], 0))
                continue
            
            next_state, reward, done, info = self.envs[i].step(action)
            
            # 預處理
            next_state_processed = preprocess_frame(next_state)
            next_state_processed = np.expand_dims(next_state_processed, axis=0)
            
            # 計算智慧獎勵
            custom_reward = calculate_smart_reward(
                self.envs[i], info, reward, self.prev_infos[i]
            )
            
            # 更新狀態
            self.states[i] = next_state_processed
            self.prev_infos[i] = info
            self.dones[i] = done
            
            results.append((
                next_state_processed,
                reward,
                done,
                info,
                custom_reward
            ))
        
        return results
    
    def close(self):
        """關閉所有環境"""
        for env in self.envs:
            env.close()

# ========== 主訓練循環 ===========
def train():
    print(f"🎮 Starting parallel training with {NUM_ENVS} environments...")
    
    # 初始化
    obs_shape = (1, 84, 84)
    n_actions = len(SIMPLE_MOVEMENT)
    
    # DQN
    dqn = DQN(
        model=CustomCNN,
        state_dim=obs_shape,
        action_dim=n_actions,
        learning_rate=LR,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        target_update=TARGET_UPDATE,
        device=device
    )
    
    # 經驗回放
    memory = ReplayMemory(MEMORY_SIZE)
    
    # 環境管理器
    env_worker = EnvWorker(NUM_ENVS)
    
    # 統計
    total_steps = 0
    episode_rewards = []
    episode_max_x_list = []         # 追蹤每個 episode 的最遠距離
    best_reward = -float('inf')
    best_max_x = 0                  # 追蹤歷史最遠單次距離
    best_avg_distance = 0           # 🎯 追蹤最佳平均距離 (最重要！)
    best_avg_reward = -float('inf') # 追蹤最佳平均獎勵
    epsilon = EPSILON_START
    
    # 初始化環境
    states = env_worker.reset_all()
    
    # 訓練
    pbar = tqdm(total=TOTAL_TIMESTEPS, desc="Training")
    episode_count = 0
    current_rewards = [0] * NUM_ENVS
    current_max_x = [0] * NUM_ENVS  # 每個環境的當前最遠距離
    
    while episode_count < TOTAL_TIMESTEPS:
        # 為每個環境選擇動作
        actions = []
        for i, state in enumerate(states):
            if np.random.rand() < epsilon:
                actions.append(np.random.randint(n_actions))
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
                    q_values = dqn.q_net(state_tensor)
                    actions.append(q_values.argmax(dim=1).item())
        
        # 在所有環境中執行動作
        results = env_worker.step(actions)
        
        # 處理結果
        for i, (next_state, reward, done, info, custom_reward) in enumerate(results):
            # 存入記憶體
            memory.push(states[i], actions[i], custom_reward, next_state, done)
            
            # 更新狀態
            states[i] = next_state
            current_rewards[i] += reward
            
            # 追蹤這個環境的最遠距離
            current_max_x[i] = max(current_max_x[i], info.get('x_pos', 0))
            total_steps += 1
            
            # 如果回合結束
            if done:
                ep_reward = current_rewards[i]
                ep_max_x = current_max_x[i]
                
                episode_rewards.append(ep_reward)
                episode_max_x_list.append(ep_max_x)
                
                # ========== 保存最佳模型 (專注於平均表現) ==========
                os.makedirs("ckpt_parallel_average", exist_ok=True)
                
                # 計算移動平均 (最近 50 個 episode)
                if len(episode_max_x_list) >= 50:
                    current_avg_dist = np.mean(episode_max_x_list[-50:])
                    current_avg_reward = np.mean(episode_rewards[-50:])
                    
                    # 1. 🎯 主要：基於「平均距離」保存 (最重要！穩定性指標)
                    if current_avg_dist > best_avg_distance:
                        best_avg_distance = current_avg_dist
                        # 刪除舊的平均距離模型
                        for old_model in os.listdir("ckpt_parallel_average"):
                            if old_model.startswith("best_avg_distance_"):
                                try:
                                    os.remove(os.path.join("ckpt_parallel_average", old_model))
                                except:
                                    pass
                        model_path = f"ckpt_parallel_average/best_avg_distance_{int(best_avg_distance)}_ep_{episode_count}.pth"
                        torch.save(dqn.q_net.state_dict(), model_path)
                        print(f"\n📊 New best AVG distance: {best_avg_distance:.0f} (last 50 eps)")
                    
                    # 2. 基於「平均獎勵」保存
                    if current_avg_reward > best_avg_reward:
                        best_avg_reward = current_avg_reward
                        # 刪除舊的平均獎勵模型
                        for old_model in os.listdir("ckpt_parallel_average"):
                            if old_model.startswith("best_avg_reward_"):
                                try:
                                    os.remove(os.path.join("ckpt_parallel_average", old_model))
                                except:
                                    pass
                        model_path = f"ckpt_parallel_average/best_avg_reward_{int(best_avg_reward)}_ep_{episode_count}.pth"
                        torch.save(dqn.q_net.state_dict(), model_path)
                        print(f"\n💰 New best AVG reward: {best_avg_reward:.0f} (last 50 eps)")
                
                # 3. 記錄單次最佳 (僅供參考，不作為主要指標)
                if ep_max_x > best_max_x:
                    best_max_x = ep_max_x
                    # 不再保存單次最佳模型，只記錄
                
                # 4. 每 200 episode 保存一次 checkpoint
                if episode_count > 0 and episode_count % 200 == 0:
                    model_path = f"ckpt_parallel_average/checkpoint_ep_{episode_count}.pth"
                    torch.save(dqn.q_net.state_dict(), model_path)
                    print(f"\n📁 Checkpoint saved: ep_{episode_count}")
                
                # 重置
                current_rewards[i] = 0
                current_max_x[i] = 0
                reset_max_x()
                reset_enemy_tracking()
                reset_hole_tracking()
                episode_count += 1
                pbar.update(1)
                
                # 更新進度條 (專注於平均值)
                avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards) if episode_rewards else 0
                avg_max_x = np.mean(episode_max_x_list[-50:]) if len(episode_max_x_list) >= 50 else np.mean(episode_max_x_list) if episode_max_x_list else 0
                pbar.set_postfix({
                    'avg_dist': f'{avg_max_x:.0f}',
                    'best_avg': f'{best_avg_distance:.0f}',
                    'best_single': f'{best_max_x:.0f}',
                    'ε': f'{epsilon:.3f}'
                })
        
        # 訓練
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            state_dict = {
                'states': batch[0],
                'actions': batch[1],
                'rewards': batch[2],
                'next_states': batch[3],
                'dones': batch[4],
            }
            dqn.train_per_step(state_dict)
        
        # 更新 epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        dqn.epsilon = epsilon
    
    pbar.close()
    env_worker.close()
    
    print(f"\n✅ Training complete!")
    print(f"📊 Best reward: {best_reward}")
    print(f"📈 Average last 100 rewards: {np.mean(episode_rewards[-100:]):.1f}")

if __name__ == "__main__":
    train()
