"""
PPO 模型評估腳本
用於測試 PPO 訓練的模型
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import time

import gym
import gym_super_mario_bros
from gym.wrappers import StepAPICompatibility
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils import preprocess_frame

# ========== 配置 ===========
# MODEL_PATH = "./ckpt_ppo/ppo_best_avg_distance_1677_ep_2556.pth"  # PPO 模型路徑
MODEL_PATH = "/home/liang/Desktop/NCKU-CVDL-2025/SuperMario_SampleCode251218/ckpt_ppo/ppo_best_single_distance_1677_ep_2556.pth"

VISUALIZE = True
FRAME_DELAY = 0.02              # 每幀延遲 (0.02 = 50 FPS)
TOTAL_EPISODES = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Actor-Critic 網路 (必須與訓練時相同) ===========
class ActorCritic(nn.Module):
    """
    Actor-Critic 網路
    共享特徵提取層，分別輸出策略和價值
    """
    def __init__(self, obs_shape, n_actions):
        super(ActorCritic, self).__init__()
        
        # 共享卷積層
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 計算卷積輸出大小
        conv_out_size = self._get_conv_out(obs_shape)
        
        # 共享全連接層
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )
        
        # Actor (策略) 輸出
        self.actor = nn.Linear(512, n_actions)
        
        # Critic (價值) 輸出
        self.critic = nn.Linear(512, 1)
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = x / 255.0  # 正規化
        conv_out = self.conv(x)
        fc_out = self.fc(conv_out)
        
        # 策略 (log probabilities)
        policy = self.actor(fc_out)
        # 價值
        value = self.critic(fc_out)
        
        return policy, value
    
    def get_action(self, state, deterministic=False):
        """選擇動作
        
        Args:
            state: 輸入狀態
            deterministic: 是否使用確定性策略 (選擇最高概率的動作)
        """
        policy, value = self.forward(state)
        probs = torch.softmax(policy, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()
            
        return action

# ========== 環境設置 ===========
def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env
    env = StepAPICompatibility(env, output_truncation_bool=False)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

# ========== 評估 ===========
def evaluate():
    print(f"🎮 Evaluating PPO model: {MODEL_PATH}")
    print(f"Using device: {device}")
    
    # 環境
    env = make_env()
    obs_shape = (1, 84, 84)
    n_actions = len(SIMPLE_MOVEMENT)
    
    # 載入模型
    model = ActorCritic(obs_shape, n_actions).to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            print(f"✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return
    else:
        print(f"❌ Model file not found: {MODEL_PATH}")
        return
    
    # 評估
    episode_rewards = []
    episode_distances = []
    
    for episode in range(1, TOTAL_EPISODES + 1):
        state = env.reset()
        state = preprocess_frame(state)
        state = np.expand_dims(state, axis=0)
        
        done = False
        total_reward = 0
        max_x = 0
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                # 使用確定性策略（選擇最高概率的動作）
                action = model.get_action(state_tensor, deterministic=True)
            
            action_np = action.cpu().numpy()[0]
            next_state, reward, done, info = env.step(action_np)
            
            # 預處理
            next_state = preprocess_frame(next_state)
            next_state = np.expand_dims(next_state, axis=0)
            
            total_reward += reward
            max_x = max(max_x, info['x_pos'])
            state = next_state
            
            if VISUALIZE:
                env.render()
                time.sleep(FRAME_DELAY)
        
        episode_rewards.append(total_reward)
        episode_distances.append(max_x)
        print(f"Episode {episode}/{TOTAL_EPISODES} - Reward: {total_reward:.0f} - Max X: {max_x}")
    
    env.close()
    
    # 統計
    print(f"\n📊 Evaluation Results:")
    print(f"Average Reward: {np.mean(episode_rewards):.1f}")
    print(f"Average Distance: {np.mean(episode_distances):.1f}")
    print(f"Best Distance: {max(episode_distances)}")
    print(f"Worst Distance: {min(episode_distances)}")

if __name__ == "__main__":
    evaluate()
