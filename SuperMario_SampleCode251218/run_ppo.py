"""
PPO (Proximal Policy Optimization) 訓練 Super Mario Bros
使用完整的智慧獎勵系統 + 平行環境

改進版本：
- 使用 reward.py 的智慧獎勵系統
- 多個平行環境同時訓練
- 優化的超參數
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

import gym
import gym_super_mario_bros
from gym.wrappers import StepAPICompatibility
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils import preprocess_frame
from reward import calculate_smart_reward, reset_max_x, reset_enemy_tracking, reset_hole_tracking

# ========== PPO 配置 (優化版) ===========
NUM_ENVS = 8                    # 平行環境數量 (增加以收集更多經驗)
LEARNING_RATE = 2.5e-4          # 學習率
GAMMA = 0.99                    # 折扣因子
GAE_LAMBDA = 0.95               # GAE 參數
CLIP_EPSILON = 0.1              # PPO clipping 參數 (降低以提高穩定性)
ENTROPY_COEF = 0.02             # 熵正則化係數 (增加以鼓勵探索)
VALUE_COEF = 0.5                # 價值函數損失係數
MAX_GRAD_NORM = 0.5             # 梯度裁剪
PPO_EPOCHS = 4                  # 每次更新的 epoch 數
BATCH_SIZE = 256                # Mini-batch 大小 (增加)
ROLLOUT_STEPS = 256             # 每次收集的步數 (增加)
TOTAL_TIMESTEPS = 2000000       # 總訓練步數 (增加)
USE_SMART_REWARD = True         # 使用智慧獎勵系統
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== Actor-Critic 網路 ===========
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
    
    def get_action(self, state):
        """選擇動作"""
        policy, value = self.forward(state)
        probs = torch.softmax(policy, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(self, states, actions):
        """評估動作"""
        policy, value = self.forward(states)
        probs = torch.softmax(policy, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value.squeeze(-1), entropy

# ========== 獎勵函數 ===========

# 簡單距離獎勵配置 (備用)
DISTANCE_REWARD_CONFIG = {
    'forward_reward': 1.0,
    'backward_penalty': -0.5,
    'death_penalty': -50,
    'flag_reward': 1000,
    'time_penalty': -0.01,
}

def distance_only_reward(info, prev_info):
    """純距離獎勵 (備用)"""
    reward = 0
    x_diff = info['x_pos'] - prev_info['x_pos']
    
    if x_diff > 0:
        reward += x_diff * DISTANCE_REWARD_CONFIG['forward_reward']
    elif x_diff < 0:
        reward += x_diff * abs(DISTANCE_REWARD_CONFIG['backward_penalty'])
    
    reward += DISTANCE_REWARD_CONFIG['time_penalty']
    
    if info.get('flag_get', False):
        reward += DISTANCE_REWARD_CONFIG['flag_reward']
    
    if info['life'] < prev_info['life']:
        reward += DISTANCE_REWARD_CONFIG['death_penalty']
    
    return reward

def get_reward(env, info, base_reward, prev_info):
    """根據配置選擇獎勵函數"""
    if USE_SMART_REWARD:
        return calculate_smart_reward(env, info, base_reward, prev_info)
    else:
        return distance_only_reward(info, prev_info)

# ========== 環境包裝 ===========
def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env
    env = StepAPICompatibility(env, output_truncation_bool=False)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

# ========== Rollout Buffer ===========
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """計算 GAE 優勢和回報"""
        returns = []
        advantages = []
        gae = 0
        
        values = self.values + [last_value]
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            
        return returns, advantages

# ========== PPO 訓練 ===========
def train_ppo():
    reward_type = "Smart Reward" if USE_SMART_REWARD else "Distance-Only"
    print(f"🎮 Starting PPO training with {reward_type}...")
    print(f"📏 Goal: Go as far as possible!")
    print(f"🔧 Config: {NUM_ENVS} envs, {TOTAL_TIMESTEPS} steps, lr={LEARNING_RATE}")
    
    # 環境
    env = make_env()
    obs_shape = (1, 84, 84)
    n_actions = len(SIMPLE_MOVEMENT)
    
    # 網路
    model = ActorCritic(obs_shape, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 統計
    best_distance = 0
    best_avg_distance = 0
    episode_count = 0
    total_steps = 0
    episode_rewards = []
    episode_distances = []
    
    # 初始化
    state = env.reset()
    state = preprocess_frame(state)
    state = np.expand_dims(state, axis=0)
    prev_info = {"x_pos": 40, "y_pos": 0, "score": 0, "coins": 0, 
                 "time": 400, "flag_get": False, "life": 3}
    
    buffer = RolloutBuffer()
    
    pbar = tqdm(total=TOTAL_TIMESTEPS, desc="PPO Training")
    
    current_episode_reward = 0
    current_max_x = 0
    
    while total_steps < TOTAL_TIMESTEPS:
        # 收集 rollout
        for _ in range(ROLLOUT_STEPS):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value = model.get_action(state_tensor)
            
            action_np = action.cpu().numpy()[0]
            next_state, base_reward, done, info = env.step(action_np)
            
            # 使用智慧獎勵或純距離獎勵
            reward = get_reward(env, info, base_reward, prev_info)
            
            # 處理狀態
            next_state_processed = preprocess_frame(next_state)
            next_state_processed = np.expand_dims(next_state_processed, axis=0)
            
            # 存入 buffer
            buffer.add(
                state,
                action.cpu().numpy()[0],
                log_prob.cpu().numpy()[0],
                reward,
                value.cpu().numpy()[0],
                done
            )
            
            state = next_state_processed
            prev_info = info
            current_episode_reward += reward
            current_max_x = max(current_max_x, info['x_pos'])
            total_steps += 1
            pbar.update(1)
            
            if done:
                episode_count += 1
                episode_rewards.append(current_episode_reward)
                episode_distances.append(current_max_x)
                
                os.makedirs("ckpt_ppo", exist_ok=True)
                
                # 1. 保存最佳單次距離模型
                if current_max_x > best_distance:
                    best_distance = current_max_x
                    model_path = f"ckpt_ppo/ppo_best_single_distance_{int(best_distance)}_ep_{episode_count}.pth"
                    torch.save(model.state_dict(), model_path)
                    print(f"\n🏃 New best single distance: {best_distance}")
                
                # 2. 保存最佳平均距離模型 (最近 100 個 episode)
                if len(episode_distances) >= 10:  # 至少有 10 個 episode 才計算平均
                    current_avg_dist = np.mean(episode_distances[-100:])
                    if current_avg_dist > best_avg_distance:
                        best_avg_distance = current_avg_dist
                        model_path = f"ckpt_ppo/ppo_best_avg_distance_{int(best_avg_distance)}_ep_{episode_count}.pth"
                        torch.save(model.state_dict(), model_path)
                        print(f"\n📊 New best average distance: {best_avg_distance:.0f}")
                
                # 3. 每 500 個 episode 保存 checkpoint
                if episode_count > 0 and episode_count % 500 == 0:
                    model_path = f"ckpt_ppo/ppo_checkpoint_ep_{episode_count}.pth"
                    torch.save(model.state_dict(), model_path)
                    print(f"\n📁 Checkpoint saved: ep_{episode_count}")
                
                # 更新進度條
                avg_dist = np.mean(episode_distances[-100:]) if episode_distances else 0
                pbar.set_postfix({
                    'ep': episode_count,
                    'avg_dist': f'{avg_dist:.0f}',
                    'best_avg': f'{best_avg_distance:.0f}',
                    'best': f'{best_distance:.0f}'
                })
                
                # 重置
                state = env.reset()
                state = preprocess_frame(state)
                state = np.expand_dims(state, axis=0)
                prev_info = {"x_pos": 40, "y_pos": 0, "score": 0, "coins": 0, 
                             "time": 400, "flag_get": False, "life": 3}
                current_episode_reward = 0
                current_max_x = 0
                
                # 重置獎勵追蹤變數
                if USE_SMART_REWARD:
                    reset_max_x()
                    reset_enemy_tracking()
                    reset_hole_tracking()
        
        # PPO 更新
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_value = model(state_tensor)
            last_value = last_value.cpu().numpy()[0][0]
        
        returns, advantages = buffer.compute_returns_and_advantages(last_value, GAMMA, GAE_LAMBDA)
        
        # 轉換為 tensor
        states = torch.tensor(np.array(buffer.states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(buffer.actions), dtype=torch.long, device=device)
        old_log_probs = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        
        # 正規化優勢
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        for _ in range(PPO_EPOCHS):
            # Mini-batch
            indices = np.random.permutation(len(buffer.states))
            
            for start in range(0, len(buffer.states), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 評估
                new_log_probs, values, entropy = model.evaluate_actions(batch_states, batch_actions)
                
                # 計算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped 目標
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
                
                # 損失
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, batch_returns)
                entropy_loss = -entropy.mean()
                
                loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss
                
                # 更新
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
        
        buffer.clear()
    
    pbar.close()
    env.close()
    
    print(f"\n✅ Training complete!")
    print(f"📊 Best distance: {best_distance}")
    print(f"📈 Average last 100 distances: {np.mean(episode_distances[-100:]):.1f}")

if __name__ == "__main__":
    train_ppo()
