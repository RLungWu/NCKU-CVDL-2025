import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# ==================== HYPERPARAMETERS TABLE ====================
# 調整此表格來修改所有獎勵/懲罰的數值
# Adjust this table to modify all reward/penalty values
# ===============================================================

# 切換模式：True = 極端模式, False = 正常模式
EXTREME_MODE = True

# ===== 正常模式 (Normal Mode) - 快速通關優化版 =====
# 設計原則：
#   1. 前進是最重要的目標 → forward_reward 最高
#   2. 強調速度 → 增加速度獎勵、時間效率
#   3. 避免矛盾信號 → fall_penalty 設為 0
#   4. 死亡是最嚴重的 → death_penalty 最大
#   5. 完成關卡巨大獎勵 → flag_reward 極高
NORMAL_CONFIG = {
    # === Core Reward (核心獎勵) ===
    "forward_reward": 30,           # 前進獎勵（主要驅動力）
    "flag_reward": 1000,            # 通關獎勵（巨大激勵）
    "speed_bonus": 15,              # 新增：快速前進額外獎勵
    
    # === Support Reward (輔助獎勵) ===
    "jump_reward": 3,               # 跳躍獎勵（適度鼓勵）
    "coin_reward": 8,               # 硬幣獎勵（輕度鼓勵）
    "score_reward": 10,             # 得分獎勵
    "life_bonus": 100,              # 獲得額外生命
    "momentum_bonus": 20,           # 新增：連續前進獎勵
    "obstacle_clear_reward": 25,    # 新增：成功越過障礙
    
    # === Core Penalty (核心懲罰) ===
    "death_penalty": -300,          # 死亡懲罰（嚴重）
    "backward_penalty": -25,        # 後退懲罰
    "stagnation_penalty": -20,      # 停滯懲罰
    
    # === Light Penalty (輕度懲罰) ===
    "fall_penalty": 0,              # 下落不懲罰（跳躍需要下落）
    "no_score_penalty": 0,          # 無得分不懲罰
    "time_waste_penalty": -8,       # 浪費時間懲罰
    "time_waste_threshold": 3,      # 時間浪費閾值
    "slow_progress_penalty": -10,   # 新增：進度太慢懲罰
}


# ===== Extreme Mode - Strategy Amplification Version =====
# Design Principles:
#   1. Not simply x10, but 'strategy amplification'
#   2. Amplify core signals more, reduce or remove noise signals
#   3. Let AI learn 'always move right + never die' faster
EXTREME_CONFIG = {
    # === Core Reward === (Timex 4)
    "forward_reward": 100,          
    "flag_reward": 2000,            
    "speed_bonus": 50,
    
    # === Support Reward === (Timex 2 ~ 3)
    "jump_reward": 10,              
    "coin_reward": 20,              
    "score_reward": 30,             
    "life_bonus": 300,
    "momentum_bonus": 60,
    "obstacle_clear_reward": 80,
    
    # === Core Penalty === (Timex 2 ~ 3)
    "death_penalty": -500,          
    "backward_penalty": -80,        
    "stagnation_penalty": -50,      
    
    # === Light Penalty === (Timex 2 ~ 3)
    "fall_penalty": 0,             
    "no_score_penalty": 0,          
    "time_waste_penalty": -15,      
    "time_waste_threshold": 3,
    "slow_progress_penalty": -30,
}


REWARD_CONFIG = EXTREME_CONFIG if EXTREME_MODE else NORMAL_CONFIG

print(f"Reward Mode: {'EXTREME' if EXTREME_MODE else 'NORMAL'}")

# Env state 
# info = {
#     "x_pos",  # (int) The player's horizontal position in the level.
#     "y_pos",  # (int) The player's vertical position in the level.
#     "score",  # (int) The current score accumulated by the player.
#     "coins",  # (int) The number of coins the player has collected.
#     "time",   # (int) The remaining time for the level.
#     "flag_get",  # (bool) True if the player has reached the end flag (level completion).
#     "life"   # (int) The number of lives the player has left.
# }


# # simple actions_dim = 7 
# SIMPLE_MOVEMENT = [
#     ["NOOP"],       # Do nothing.
#     ["right"],      # Move right.
#     ["right", "A"], # Move right and jump.
#     ["right", "B"], # Move right and run.
#     ["right", "A", "B"], # Move right, run, and jump.
#     ["A"],          # Jump straight up.
#     ["left"],       # Move left.
# ]

# ==================== 全域變數追蹤 ====================
# 用於追蹤連續前進的步數（momentum計算）
_momentum_counter = 0
_last_x_pos = 0
_max_x_reached = 0

def reset_tracking():
    """重置追蹤變數，每個 episode 開始時呼叫"""
    global _momentum_counter, _last_x_pos, _max_x_reached
    _momentum_counter = 0
    _last_x_pos = 0
    _max_x_reached = 0

#-----------------------------------------------------------------------------
# 獎勵函數
'''
get_coin_reward         : 根據硬幣數量變化提供額外獎勵

'''
'''
環境資訊 (info)
1."x_pos": 水平位置，用於判斷角色的前進情況
2."y_pos": 垂直位置，用於分析跳躍或下落行為
3."score": 玩家目前的遊戲分數
4."coins": 收集到的硬幣數量
5."time": 剩餘時間
5."flag_get": 是否到達終點旗幟（遊戲完成）
6."life": 玩家剩餘的生命數
'''

#=============== 原有獎勵函數（優化版）===============

# 1. 用來獎勵玩家蒐集硬幣的行為
def get_coin_reward(info, reward, prev_info):
    total_reward = reward
    total_reward += (info['coins'] - prev_info['coins']) * REWARD_CONFIG["coin_reward"]
    return total_reward

# 2. 用來鼓勵玩家進行跳躍或高度變化
def distance_y_offset_reward(info, reward, prev_info):
    y_diff = info['y_pos'] - prev_info['y_pos']
    if y_diff > 0:
        return REWARD_CONFIG["jump_reward"]
    else:
        return REWARD_CONFIG["fall_penalty"]  # 現在是 0

# 3. 用來鼓勵玩家前進，懲罰原地停留或後退
def distance_x_offset_reward(info, reward, prev_info):
    x_diff = info['x_pos'] - prev_info['x_pos']
    if x_diff > 0:
        return REWARD_CONFIG["forward_reward"]
    elif x_diff < 0:
        return REWARD_CONFIG["backward_penalty"]
    return 0

# 4. 用來鼓勵玩家提高分數（例如擊敗敵人)
def monster_score_reward(info, reward, prev_info):
    if info['score'] - prev_info['score'] > 0:
        return REWARD_CONFIG["score_reward"]
    else:
        return REWARD_CONFIG["no_score_penalty"]

# 5. 用來鼓勵玩家完成關卡（到達終點旗幟）
def final_flag_reward(info, reward):
    if info['flag_get']:
        return REWARD_CONFIG["flag_reward"]
    else:
        return 0

# 6. 時間效率獎勵：鼓勵玩家在有限時間內快速通關
def time_efficiency_reward(info, reward, prev_info):
    """
    根據剩餘時間給予獎勵，時間減少過快會受到懲罰
    """
    time_diff = prev_info['time'] - info['time']
    if time_diff > REWARD_CONFIG["time_waste_threshold"]:
        return REWARD_CONFIG["time_waste_penalty"]
    return 0

# 7. 死亡懲罰：玩家失去生命時給予大量懲罰
def death_penalty(info, reward, prev_info):
    """
    當玩家失去生命時給予嚴重懲罰
    """
    if info['life'] < prev_info['life']:
        return REWARD_CONFIG["death_penalty"]
    return 0

# 8. 停滯懲罰：懲罰長時間停留在同一位置
def stagnation_penalty(info, reward, prev_info):
    """
    如果玩家的 x 位置沒有變化，給予小懲罰
    """
    if info['x_pos'] == prev_info['x_pos']:
        return REWARD_CONFIG["stagnation_penalty"]
    return 0

# 9. 生命獎勵：獎勵玩家獲得額外生命
def life_bonus_reward(info, reward, prev_info):
    """
    當玩家獲得額外生命時給予獎勵
    """
    if info['life'] > prev_info['life']:
        return REWARD_CONFIG["life_bonus"]
    return 0


#=============== 新增獎勵函數（快速通關優化）===============

# 10. 速度獎勵：獎勵快速前進
def speed_bonus_reward(info, reward, prev_info):
    """
    當玩家快速前進時給予額外獎勵
    x_diff > 5 表示跑動狀態（按住B鍵）
    """
    x_diff = info['x_pos'] - prev_info['x_pos']
    if x_diff > 5:  # 快速移動閾值
        return REWARD_CONFIG["speed_bonus"]
    return 0

# 11. 連續前進獎勵（動量獎勵）
def momentum_bonus_reward(info, reward, prev_info):
    """
    連續多步前進時給予累積獎勵
    鼓勵持續保持前進狀態
    """
    global _momentum_counter
    x_diff = info['x_pos'] - prev_info['x_pos']
    
    if x_diff > 0:
        _momentum_counter += 1
        if _momentum_counter >= 5:  # 連續前進 5 步以上
            return REWARD_CONFIG["momentum_bonus"]
    else:
        _momentum_counter = 0
    return 0

# 12. 新紀錄獎勵：到達新的最遠位置
def new_distance_record_reward(info, reward, prev_info):
    """
    當玩家到達目前 episode 的最遠位置時給予獎勵
    鼓勵探索和突破
    """
    global _max_x_reached
    current_x = info['x_pos']
    
    if current_x > _max_x_reached:
        bonus = (current_x - _max_x_reached) * 0.5  # 每多一格 0.5 獎勵
        _max_x_reached = current_x
        return bonus
    return 0

# 13. 成功越過障礙獎勵
def obstacle_clear_reward(info, reward, prev_info):
    """
    當玩家在跳躍後成功前進較大距離時給予獎勵
    表示成功越過了障礙或敵人
    """
    x_diff = info['x_pos'] - prev_info['x_pos']
    y_diff = info['y_pos'] - prev_info['y_pos']
    
    # 從高處落下且有明顯前進 = 成功跳過障礙
    if y_diff < -10 and x_diff > 3:
        return REWARD_CONFIG["obstacle_clear_reward"]
    return 0

# 14. 進度效率獎勵：根據時間計算進度
def progress_efficiency_reward(info, reward, prev_info):
    """
    根據當前位置和剩餘時間計算效率
    時間越多、位置越遠 = 效率越高
    """
    # 遊戲總時間通常是 400
    time_remaining_ratio = info['time'] / 400.0
    # 第一關終點大約在 x = 3200
    progress_ratio = info['x_pos'] / 3200.0
    
    # 如果進度快於時間消耗，給予獎勵
    if progress_ratio > (1 - time_remaining_ratio) + 0.1:
        return 5  # 小額持續獎勵
    elif progress_ratio < (1 - time_remaining_ratio) - 0.2:
        return REWARD_CONFIG["slow_progress_penalty"]
    return 0

# 15. 跳躍精準度獎勵
def precise_jump_reward(info, reward, prev_info):
    """
    當跳躍達到較高高度且沒有死亡時給予獎勵
    鼓勵精準跳躍而非隨意跳躍
    """
    y_diff = info['y_pos'] - prev_info['y_pos']
    x_diff = info['x_pos'] - prev_info['x_pos']
    
    # 跳得高且有前進
    if y_diff > 20 and x_diff > 0:
        return 5
    return 0

# 16. 完美通關獎勵：剩餘時間越多獎勵越高
def perfect_clear_bonus(info, reward, prev_info):
    """
    通關時根據剩餘時間給予額外獎勵
    鼓勵快速通關
    """
    if info['flag_get']:
        time_bonus = info['time'] * 2  # 每秒剩餘時間給 2 點獎勵
        return time_bonus
    return 0


# ==============總獎勵計算========================================
def final_reward(info, reward, prev_info):
    """
    計算單步的總獎勵
    整合所有獎勵函數，返回加權後的總獎勵
    """
    total_reward = 0
    
    # === 原有的 9 個獎勵函數 ===
    total_reward += get_coin_reward(info, reward, prev_info)
    total_reward += distance_y_offset_reward(info, reward, prev_info)
    total_reward += distance_x_offset_reward(info, reward, prev_info)
    total_reward += monster_score_reward(info, reward, prev_info)
    total_reward += final_flag_reward(info, reward)
    total_reward += time_efficiency_reward(info, reward, prev_info)
    total_reward += death_penalty(info, reward, prev_info)
    total_reward += stagnation_penalty(info, reward, prev_info)
    total_reward += life_bonus_reward(info, reward, prev_info)
    
    # === 新增的 7 個快速通關獎勵函數 ===
    total_reward += speed_bonus_reward(info, reward, prev_info)          # 10. 速度獎勵
    total_reward += momentum_bonus_reward(info, reward, prev_info)       # 11. 連續前進獎勵
    total_reward += new_distance_record_reward(info, reward, prev_info)  # 12. 新紀錄獎勵
    total_reward += obstacle_clear_reward(info, reward, prev_info)       # 13. 越過障礙獎勵
    total_reward += progress_efficiency_reward(info, reward, prev_info)  # 14. 進度效率獎勵
    total_reward += precise_jump_reward(info, reward, prev_info)         # 15. 跳躍精準度獎勵
    total_reward += perfect_clear_bonus(info, reward, prev_info)         # 16. 完美通關獎勵
    
    return total_reward
