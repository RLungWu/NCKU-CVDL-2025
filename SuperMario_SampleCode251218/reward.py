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

# 🔥 切換模式：True = 極端模式, False = 正常模式
EXTREME_MODE = True

# ===== 正常模式 (Normal Mode) =====
NORMAL_CONFIG = {
    "coin_reward": 5,               # 收集硬幣獎勵
    "jump_reward": 10,              # 跳躍獎勵
    "forward_reward": 10,           # 前進獎勵
    "score_reward": 8,              # 擊敗敵人獎勵
    "flag_reward": 100,             # 終點獎勵
    "life_bonus": 50,               # 1UP獎勵
    "fall_penalty": -10,            # 下落懲罰
    "backward_penalty": -10,        # 後退懲罰
    "no_score_penalty": -10,        # 無得分懲罰
    "death_penalty": -100,          # 死亡懲罰
    "stagnation_penalty": -5,       # 停滯懲罰
    "time_waste_penalty": -5,       # 時間浪費懲罰
    "time_waste_threshold": 3,
}

# ===== 極端模式 (EXTREME Mode) - 更強烈的獎懲訊號 =====
EXTREME_CONFIG = {
    "coin_reward": 50,              # 硬幣獎勵 x10
    "jump_reward": 30,              # 跳躍獎勵 x3
    "forward_reward": 50,           # 前進獎勵 x5 (最重要!)
    "score_reward": 40,             # 擊敗敵人 x5
    "flag_reward": 1000,            # 終點獎勵 x10
    "life_bonus": 200,              # 1UP獎勵 x4
    "fall_penalty": -20,            # 下落懲罰 x2
    "backward_penalty": -50,        # 後退懲罰 x5 (強烈阻止後退!)
    "no_score_penalty": 0,          # 不懲罰無得分 (減少噪音)
    "death_penalty": -500,          # 死亡懲罰 x5
    "stagnation_penalty": -30,      # 停滯懲罰 x6 (強烈阻止原地不動!)
    "time_waste_penalty": -20,      # 時間浪費懲罰 x4
    "time_waste_threshold": 2,      # 更嚴格的時間閾值
}

# 根據模式選擇配置
REWARD_CONFIG = EXTREME_CONFIG if EXTREME_MODE else NORMAL_CONFIG

print(f"🎮 Reward Mode: {'🔥 EXTREME' if EXTREME_MODE else '📊 NORMAL'}")

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
#-----------------------------------------------------------------------------
#獎勵函數
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

#===============to do===============================請自定義獎勵函數 至少7個(包含提供的)

#例子:用來獎勵玩家蒐集硬幣的行為
def get_coin_reward(info, reward, prev_info):
    #寫下蒐集到硬幣會對應多少獎勵
    total_reward = reward                                         #獲得目前已有的獎勵數量
    total_reward += (info['coins'] - prev_info['coins']) * REWARD_CONFIG["coin_reward"]
    return total_reward

#用來鼓勵玩家進行跳躍或高度變化(因為有時前方有障礙物 會被卡住)
def distance_y_offset_reward(info, reward, prev_info):
    if info['y_pos'] - prev_info['y_pos'] > 0:
        return REWARD_CONFIG["jump_reward"]
    else:
        return REWARD_CONFIG["fall_penalty"]

#用來鼓勵玩家前進，懲罰原地停留或後退
def distance_x_offset_reward(info, reward, prev_info):
    if info['x_pos'] - prev_info['x_pos'] > 0:
        return REWARD_CONFIG["forward_reward"]
    else:
        return REWARD_CONFIG["backward_penalty"]

#用來鼓勵玩家提高分數（例如擊敗敵人)
def monster_score_reward(info, reward, prev_info):
    if info['score'] - prev_info['score'] > 0:
        return REWARD_CONFIG["score_reward"]
    else:
        return REWARD_CONFIG["no_score_penalty"]

#用來鼓勵玩家完成關卡（到達終點旗幟）
def final_flag_reward(info,reward):
    if info['flag_get']:
        return REWARD_CONFIG["flag_reward"]
    else:
        return 0



#===============to do==========================================

# 6. 時間效率獎勵：鼓勵玩家在有限時間內快速通關
def time_efficiency_reward(info, reward, prev_info):
    """
    根據剩餘時間給予獎勵，時間越多代表玩家效率越高
    時間減少過快會受到懲罰
    """
    time_diff = prev_info['time'] - info['time']
    if time_diff > REWARD_CONFIG["time_waste_threshold"]:
        return REWARD_CONFIG["time_waste_penalty"]
    return 0

# 7. 死亡懲罰：玩家失去生命時給予大量懲罰
def death_penalty(info, reward, prev_info):
    """
    當玩家失去生命時給予嚴重懲罰
    這會讓 AI 學習避免危險情況
    """
    if info['life'] < prev_info['life']:
        return REWARD_CONFIG["death_penalty"]
    return 0

# 8. 停滯懲罰：懲罰長時間停留在同一位置
def stagnation_penalty(info, reward, prev_info):
    """
    如果玩家的 x 位置沒有變化，給予小懲罰
    鼓勵持續前進
    """
    if info['x_pos'] == prev_info['x_pos']:
        return REWARD_CONFIG["stagnation_penalty"]
    return 0

# 9. 生命獎勵：獎勵玩家獲得額外生命（如吃到1UP蘑菇）
def life_bonus_reward(info, reward, prev_info):
    """
    當玩家獲得額外生命時給予獎勵
    鼓勵玩家尋找隱藏的1UP獎勵
    """
    if info['life'] > prev_info['life']:
        return REWARD_CONFIG["life_bonus"]
    return 0







# ==============to do========================================
def final_reward(info,reward, prev_info):
    final_reward = 0
    # 原有的 5 個獎勵函數
    final_reward += get_coin_reward(info, reward, prev_info)
    final_reward += distance_y_offset_reward(info, reward, prev_info)
    final_reward += distance_x_offset_reward(info, reward, prev_info)
    final_reward += monster_score_reward(info, reward, prev_info)
    final_reward += final_flag_reward(info, reward)
    
    # 新增的 4 個獎勵/懲罰函數
    final_reward += time_efficiency_reward(info, reward, prev_info)  # 時間效率獎勵
    final_reward += death_penalty(info, reward, prev_info)           # 死亡懲罰
    final_reward += stagnation_penalty(info, reward, prev_info)      # 停滯懲罰
    final_reward += life_bonus_reward(info, reward, prev_info)       # 生命獎勵
    
    return final_reward


