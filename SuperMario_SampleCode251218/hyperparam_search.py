"""
獎勵函數超參數搜索
自動嘗試不同的獎勵配置，找到最佳組合
"""
import os
import json
import numpy as np
import torch
from datetime import datetime
from itertools import product
import random

import gym
import gym_super_mario_bros
from gym.wrappers import StepAPICompatibility
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils import preprocess_frame
from model import CustomCNN
from DQN import DQN, ReplayMemory

# ========== 搜索配置 ===========
SEARCH_METHOD = "random"  # "grid", "random", "manual"
NUM_RANDOM_TRIALS = 10    # 隨機搜索的試驗次數
EPISODES_PER_TRIAL = 100  # 每個配置訓練的 episode 數
EVAL_EPISODES = 10        # 評估時的 episode 數

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 超參數搜索空間 ===========
HYPERPARAMETER_SPACE = {
    # 基本獎勵
    'coin_reward': [5, 10, 20],
    'forward_reward': [0.5, 1.0, 2.0],
    'backward_penalty': [-10, -5, -1],
    
    # 敵人相關
    'kill_base_reward': [10, 20, 50],
    'enemy_approach_penalty': [-10, -5, -2],
    
    # 坑洞相關
    'hole_crossed_reward': [50, 100, 200],
    'fall_death_penalty': [-300, -200, -100],
    
    # 突破獎勵
    'breakthrough_large_reward': [1.0, 2.0, 5.0],
    'stagnation_penalty': [-5, -2, -1],
}

# ========== 預設配置 (基準) ===========
DEFAULT_CONFIG = {
    'coin_reward': 10,
    'forward_reward': 1.0,
    'backward_penalty': 0.01,
    'jump_reward': 0.5,
    'score_reward_multiplier': 0.1,
    'flag_reward': 1000,
    'enemy_approach_penalty': -5,
    'enemy_jump_over_reward': 20,
    'enemy_safe_distance_reward': 1,
    'kill_base_reward': 20,
    'kill_combo_bonus': 10,
    'kill_per_enemy_bonus': 10,
    'stomp_kill_bonus': 5,
    'survival_reward': 0.3,
    'death_penalty': -100,
    'life_lost_penalty': -100,
    'fast_forward_reward': 0.5,
    'efficiency_reward': 2,
    'threat_jump_reward': 10,
    'unnecessary_jump_penalty': -0.1,
    'powerup_reward': 50,
    'oneup_reward': 200,
    'breakthrough_small_reward': 0.5,
    'breakthrough_large_reward': 2.0,
    'stagnation_penalty': -1,
    'jump_at_frontier_reward': 5,
    'forward_jump_reward': 3,
    'hole_approach_jump_reward': 15,
    'hole_approach_no_jump_penalty': -2,
    'hole_over_reward': 20,
    'hole_crossed_reward': 100,
    'falling_penalty': -30,
    'fall_death_penalty': -200,
    'air_forward_reward': 0.5,
}

# ========== 環境建立 ===========
def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env
    env = StepAPICompatibility(env, output_truncation_bool=False)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

# ========== 快速訓練評估 ===========
def quick_train_and_eval(config, num_episodes=100, eval_episodes=10):
    """
    快速訓練並評估一個配置
    返回平均距離和最大距離
    """
    # 更新全局獎勵配置
    import reward
    for key, value in config.items():
        if key in reward.REWARD_CONFIG:
            reward.REWARD_CONFIG[key] = value
    
    # 環境
    env = make_env()
    obs_shape = (1, 84, 84)
    n_actions = len(SIMPLE_MOVEMENT)
    
    # DQN
    dqn = DQN(
        model=CustomCNN,
        state_dim=obs_shape,
        action_dim=n_actions,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=1.0,
        target_update=50,
        device=device
    )
    
    memory = ReplayMemory(10000)
    
    distances = []
    epsilon = 1.0
    
    for episode in range(num_episodes):
        state = env.reset()
        state = preprocess_frame(state)
        state = np.expand_dims(state, axis=0)
        
        prev_info = {"x_pos": 40, "y_pos": 0, "score": 0, "coins": 0, 
                     "time": 400, "flag_get": False, "life": 3}
        
        done = False
        max_x = 0
        
        # 重置追蹤變數
        reward.reset_max_x()
        try:
            reward.reset_enemy_tracking()
            reward.reset_hole_tracking()
        except:
            pass
        
        while not done:
            # 選擇動作
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
                    q_values = dqn.q_net(state_tensor)
                    action = q_values.argmax(dim=1).item()
            
            # 執行動作
            next_state, base_reward, done, info = env.step(action)
            
            # 計算獎勵
            custom_reward = reward.calculate_smart_reward(env, info, base_reward, prev_info)
            
            # 預處理
            next_state_processed = preprocess_frame(next_state)
            next_state_processed = np.expand_dims(next_state_processed, axis=0)
            
            # 存入記憶
            memory.push(state, action, custom_reward, next_state_processed, done)
            
            # 訓練
            if len(memory) >= 32:
                batch = memory.sample(32)
                state_dict = {
                    'states': batch[0],
                    'actions': batch[1],
                    'rewards': batch[2],
                    'next_states': batch[3],
                    'dones': batch[4],
                }
                dqn.train_per_step(state_dict)
            
            state = next_state_processed
            prev_info = info
            max_x = max(max_x, info['x_pos'])
        
        distances.append(max_x)
        epsilon = max(0.1, epsilon * 0.995)
    
    env.close()
    
    # 評估
    avg_distance = np.mean(distances[-eval_episodes:])
    max_distance = max(distances)
    
    return {
        'avg_distance': avg_distance,
        'max_distance': max_distance,
        'all_distances': distances,
    }

# ========== Grid Search ===========
def grid_search():
    """網格搜索所有超參數組合"""
    results = []
    
    # 只搜索部分關鍵參數（全搜索太慢）
    key_params = ['forward_reward', 'kill_base_reward', 'hole_crossed_reward']
    
    combinations = list(product(*[HYPERPARAMETER_SPACE[k] for k in key_params]))
    
    print(f"Grid Search: {len(combinations)} combinations")
    
    for i, combo in enumerate(combinations):
        config = DEFAULT_CONFIG.copy()
        for k, v in zip(key_params, combo):
            config[k] = v
        
        print(f"\n[{i+1}/{len(combinations)}] Testing: {dict(zip(key_params, combo))}")
        
        result = quick_train_and_eval(config, EPISODES_PER_TRIAL, EVAL_EPISODES)
        result['config'] = {k: v for k, v in zip(key_params, combo)}
        results.append(result)
        
        print(f"  Avg Distance: {result['avg_distance']:.1f}, Max: {result['max_distance']}")
    
    return results

# ========== Random Search ===========
def random_search(num_trials=10):
    """隨機搜索超參數"""
    results = []
    
    print(f"Random Search: {num_trials} trials")
    
    for i in range(num_trials):
        config = DEFAULT_CONFIG.copy()
        
        # 隨機選擇部分參數
        sampled = {}
        for key, values in HYPERPARAMETER_SPACE.items():
            if random.random() < 0.5:  # 50% 機率調整這個參數
                config[key] = random.choice(values)
                sampled[key] = config[key]
        
        print(f"\n[{i+1}/{num_trials}] Testing: {sampled}")
        
        result = quick_train_and_eval(config, EPISODES_PER_TRIAL, EVAL_EPISODES)
        result['config'] = sampled
        results.append(result)
        
        print(f"  Avg Distance: {result['avg_distance']:.1f}, Max: {result['max_distance']}")
    
    return results

# ========== 手動調整建議 ===========
def manual_tuning_guide():
    """輸出手動調整的建議"""
    guide = """
    ========================================
    🎯 獎勵超參數手動調整指南
    ========================================
    
    1. 如果 Mario 不願意向前走:
       → 增加 forward_reward (1.0 → 2.0)
       → 減少 backward_penalty (-5 → -10)
    
    2. 如果 Mario 不願意跳躍:
       → 增加 jump_reward (0.5 → 1.0)
       → 增加 threat_jump_reward (10 → 20)
    
    3. 如果 Mario 總是被敵人殺死:
       → 增加 enemy_approach_penalty (-5 → -10)
       → 增加 kill_base_reward (20 → 50)
    
    4. 如果 Mario 掉入坑洞:
       → 增加 hole_crossed_reward (100 → 200)
       → 增加 fall_death_penalty (-200 → -300)
       → 增加 hole_approach_jump_reward (15 → 30)
    
    5. 如果 Mario 卡在某個地方:
       → 增加 stagnation_penalty (-1 → -5)
       → 增加 breakthrough_large_reward (2.0 → 5.0)
    
    6. 如果 Mario 跳太多次:
       → 增加 unnecessary_jump_penalty (-0.1 → -1)
    
    ========================================
    """
    print(guide)
    return guide

# ========== 保存結果 ===========
def save_results(results, filename=None):
    """保存搜索結果"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hyperparam_search_{timestamp}.json"
    
    # 轉換為可序列化格式
    serializable = []
    for r in results:
        s = {
            'config': r['config'],
            'avg_distance': float(r['avg_distance']),
            'max_distance': int(r['max_distance']),
        }
        serializable.append(s)
    
    # 排序
    serializable.sort(key=lambda x: x['avg_distance'], reverse=True)
    
    # 保存
    os.makedirs("hyperparam_results", exist_ok=True)
    filepath = os.path.join("hyperparam_results", filename)
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\n📁 Results saved to {filepath}")
    
    # 輸出最佳配置
    print("\n🏆 Best Configurations:")
    for i, s in enumerate(serializable[:3]):
        print(f"  {i+1}. Avg: {s['avg_distance']:.1f}, Max: {s['max_distance']}")
        print(f"     Config: {s['config']}")
    
    return filepath

# ========== 主函數 ===========
def main():
    print("=" * 50)
    print("🔍 Reward Hyperparameter Search")
    print("=" * 50)
    
    if SEARCH_METHOD == "grid":
        results = grid_search()
    elif SEARCH_METHOD == "random":
        results = random_search(NUM_RANDOM_TRIALS)
    elif SEARCH_METHOD == "manual":
        manual_tuning_guide()
        return
    else:
        print(f"Unknown method: {SEARCH_METHOD}")
        return
    
    save_results(results)
    
    print("\n✅ Search complete!")

if __name__ == "__main__":
    main()
