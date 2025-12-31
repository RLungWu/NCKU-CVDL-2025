import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# ==================== 獎勵超參數配置 ====================
# 在這裡集中調整所有獎勵的權重，方便實驗不同配置
# =========================================================

REWARD_CONFIG = {
    # === 基本獎勵 ===
    'coin_reward': 10,              # 每個硬幣的獎勵
    'forward_reward': 1.0,          # 向前移動的獎勵 (per pixel)
    'backward_penalty': -0.01,         # 向後移動的懲罰
    'jump_reward': 0.5,             # 跳躍時的 Y 移動獎勵 (per pixel)
    'score_reward_multiplier': 0.1, # 分數變化的獎勵倍率
    'flag_reward': 1000,            # 到達終點旗幟的獎勵
    
    # === 敵人相關獎勵 ===
    'enemy_approach_penalty': -5,   # 接近敵人的懲罰 (per danger level)
    'enemy_jump_over_reward': 10,   # 跳過敵人的獎勵 (per danger level)
    'enemy_safe_distance_reward': 1,# 保持安全距離的獎勵
    'kill_base_reward': 10,         # 擊殺敵人的基礎獎勵
    'kill_combo_bonus': 0,         # 連殺加成 (per combo)
    'kill_per_enemy_bonus': 0,     # 每個敵人的額外獎勵
    'stomp_kill_bonus': 0,         # 踩殺的額外獎勵
    
    # === 生存獎勵 ===
    'survival_reward': 0.1,         # 每一步存活的獎勵
    'death_penalty': -100,          # 死亡懲罰
    'life_lost_penalty': -100,      # 失去生命的懲罰
    
    # === 速度獎勵 ===
    'fast_forward_reward': 0.5,     # 快速前進的獎勵 (per pixel when x_diff > 5)
    'efficiency_reward': 2,         # 高效率移動的獎勵
    
    # === 跳躍時機獎勵 ===
    'threat_jump_reward': 10,       # 有威脅時跳躍的獎勵
    'unnecessary_jump_penalty': -0.1, # 不必要的跳躍懲罰
    
    # === 道具獎勵 ===
    'powerup_reward': 50,           # 吃到蘑菇/花的獎勵
    'oneup_reward': 200,            # 1UP 的獎勵
    
    # === 障礙物突破獎勵 ===
    'breakthrough_small_reward': 0.5,  # 小幅突破的獎勵 (per pixel)
    'breakthrough_large_reward': 2.0,  # 大幅突破的獎勵 (per pixel, when > 20)
    'stagnation_penalty': -1,          # 停滯的懲罰
    'jump_at_frontier_reward': 5,      # 在邊界跳躍的獎勵
    'forward_jump_reward': 3,          # 前進中跳躍的獎勵
    
    # === 坑洞跨越獎勵 ===
    'hole_approach_jump_reward': 15,   # 接近坑洞時跳躍的獎勵
    'hole_approach_no_jump_penalty': -2, # 接近坑洞但不跳的懲罰
    'hole_over_reward': 20,            # 在坑洞上空的獎勵
    'hole_crossed_reward': 100,        # 成功跨越坑洞的獎勵
    'falling_penalty': -30,            # 正在掉落的懲罰
    'fall_death_penalty': -200,        # 掉入坑洞死亡的懲罰
    'air_forward_reward': 0.5,         # 空中水平移動的獎勵 (per pixel)
}

# 快捷訪問函數
def get_reward(key):
    """獲取獎勵配置值"""
    return REWARD_CONFIG.get(key, 0)


# ==================== NES RAM 地址對照表 ====================
# 這些是 Super Mario Bros 的記憶體地址，可以直接讀取遊戲狀態
RAM_ADDRESSES = {
    # Mario 狀態
    'mario_x_pos': 0x006D,          # Mario 螢幕上 X 位置
    'mario_y_pos': 0x00CE,          # Mario 螢幕上 Y 位置 (可能需要調整)
    'mario_state': 0x000E,          # Mario 狀態 (0=小, 1=大, 2+=火焰)
    'mario_floating': 0x001D,       # Mario 是否在空中
    
    # 敵人狀態 (最多 5 個敵人)
    'enemy_drawn': [0x000F, 0x0010, 0x0011, 0x0012, 0x0013],  # 敵人是否被繪製
    'enemy_type': [0x0016, 0x0017, 0x0018, 0x0019, 0x001A],   # 敵人類型
    'enemy_x_pos': [0x0087, 0x0088, 0x0089, 0x008A, 0x008B],  # 敵人 X 位置
    'enemy_y_pos': [0x00CF, 0x00D0, 0x00D1, 0x00D2, 0x00D3],  # 敵人 Y 位置
    
    # 世界狀態
    'player_status': 0x0770,         # 0=死亡, 1=存活
    'current_level': 0x075F,         # 當前關卡
}

# 敵人類型對照
ENEMY_TYPES = {
    0x00: 'Goomba (綠)',
    0x06: 'Goomba (棕)',
    0x01: 'Koopa (綠)',
    0x02: 'Koopa (紅)',
    0x05: 'Piranha Plant',
    0x07: 'Hammer Bro',
    0x09: 'Bullet Bill',
}

# ==================== 輔助函數 ====================

def read_ram(env, address):
    """從 NES RAM 讀取數值"""
    try:
        # 取得底層環境
        base_env = env.unwrapped
        return base_env.ram[address]
    except:
        return 0

def get_enemies_info(env):
    """獲取所有敵人的資訊"""
    enemies = []
    try:
        base_env = env.unwrapped
        for i in range(5):
            # 檢查敵人是否存在
            is_drawn = base_env.ram[RAM_ADDRESSES['enemy_drawn'][i]]
            if is_drawn:
                enemy = {
                    'type': base_env.ram[RAM_ADDRESSES['enemy_type'][i]],
                    'x': base_env.ram[RAM_ADDRESSES['enemy_x_pos'][i]],
                    'y': base_env.ram[RAM_ADDRESSES['enemy_y_pos'][i]],
                }
                enemies.append(enemy)
    except:
        pass
    return enemies

def get_mario_screen_pos(env):
    """獲取 Mario 在螢幕上的位置"""
    try:
        base_env = env.unwrapped
        x = base_env.ram[RAM_ADDRESSES['mario_x_pos']]
        y = base_env.ram[RAM_ADDRESSES['mario_y_pos']]
        return x, y
    except:
        return 128, 128  # 預設中間位置

def is_mario_in_air(env):
    """檢查 Mario 是否在空中"""
    try:
        base_env = env.unwrapped
        return base_env.ram[RAM_ADDRESSES['mario_floating']] > 0
    except:
        return False

# ==================== 基本獎勵函數 ====================

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

#=============== 基本獎勵函數 ===============================

#例子:用來獎勵玩家蒐集硬幣的行為
def get_coin_reward(info, reward, prev_info):
    total_reward = reward
    total_reward += (info['coins'] - prev_info['coins']) * 10
    return total_reward

#用來鼓勵玩家進行跳躍或高度變化
def distance_y_offset_reward(info, reward, prev_info):
    total_reward = reward
    y_diff = abs(info['y_pos'] - prev_info['y_pos'])
    if y_diff > 0:
        total_reward += y_diff * 0.5
    return total_reward

#用來鼓勵玩家前進，懲罰原地停留或後退
def distance_x_offset_reward(info, reward, prev_info):
    total_reward = reward
    x_diff = info['x_pos'] - prev_info['x_pos']
    if x_diff > 0:
        total_reward += x_diff * 1.0
    elif x_diff < 0:
        total_reward -= 5
    return total_reward

#用來鼓勵玩家提高分數（例如擊敗敵人)
def monster_score_reward(info, reward, prev_info):
    total_reward = reward
    score_diff = info['score'] - prev_info['score']
    if score_diff > 0:
        total_reward += score_diff * 0.1
    return total_reward

#用來鼓勵玩家完成關卡（到達終點旗幟）
def final_flag_reward(info, reward):
    total_reward = reward
    if info['flag_get']:
        total_reward += 1000
    return total_reward

# ==================== 🧠 進階智慧獎勵函數 ====================

def enemy_avoidance_reward(env, info, reward, prev_info):
    """
    智慧敵人迴避獎勵
    - 當附近有敵人時，獎勵保持距離
    - 跳過敵人時給予額外獎勵
    """
    total_reward = reward
    
    try:
        mario_x, mario_y = get_mario_screen_pos(env)
        enemies = get_enemies_info(env)
        
        for enemy in enemies:
            # 計算與敵人的距離 (轉為 int 避免 uint8 overflow)
            dx = int(enemy['x']) - int(mario_x)
            dy = int(enemy['y']) - int(mario_y)
            distance = np.sqrt(dx**2 + dy**2)
            
            # 如果敵人在前方且很近 (危險區域)
            if 0 < dx < 50 and abs(dy) < 30:
                danger_level = max(0, 50 - dx) / 50  # 越近越危險
                
                # 如果 Mario 在空中 (跳過敵人)
                if is_mario_in_air(env) and mario_y < enemy['y']:
                    total_reward += 20 * danger_level  # 獎勵跳過敵人
                else:
                    total_reward -= 5 * danger_level   # 懲罰接近敵人
                    
            # 如果成功保持安全距離
            elif distance > 60:
                total_reward += 1  # 小獎勵保持距離
                
    except Exception as e:
        pass  # 如果讀取失敗，不影響遊戲
    
    return total_reward

def survival_time_reward(info, reward, prev_info):
    """
    存活時間獎勵
    - 每存活一段時間給予小獎勵
    - 鼓勵持續生存
    """
    total_reward = reward
    
    # 如果生命沒有減少，給予存活獎勵
    if info['life'] >= prev_info['life']:
        total_reward += 0.1  # 每一步小獎勵
    else:
        total_reward -= 100  # 死亡大懲罰
        
    return total_reward

def speed_reward(info, reward, prev_info):
    """
    速度獎勵
    - 獎勵快速前進
    - 懲罰浪費時間
    """
    total_reward = reward
    
    # 計算速度 (x 位移)
    x_diff = info['x_pos'] - prev_info['x_pos']
    
    # 快速前進獎勵
    if x_diff > 5:
        total_reward += x_diff * 0.5  # 速度越快獎勵越多
    
    # 時間效率獎勵
    time_diff = prev_info['time'] - info['time']
    progress = x_diff / max(time_diff, 1)  # 每單位時間前進距離
    if progress > 2:
        total_reward += 2  # 高效率獎勵
        
    return total_reward

def jump_timing_reward(env, info, reward, prev_info):
    """
    跳躍時機獎勵
    - 在適當時機跳躍 (如躲避敵人或跨越障礙)
    """
    total_reward = reward
    
    try:
        enemies = get_enemies_info(env)
        mario_x, mario_y = get_mario_screen_pos(env)
        in_air = is_mario_in_air(env)
        
        # 檢查是否有需要跳過的威脅
        threat_nearby = False
        for enemy in enemies:
            dx = int(enemy['x']) - int(mario_x)  # 轉為 int 避免 overflow
            if 20 < dx < 60:  # 敵人在前方適當距離
                threat_nearby = True
                break
        
        # 有威脅時跳躍 = 好的時機
        if threat_nearby and in_air:
            total_reward += 10
        # 沒有威脅時不必要的跳躍 = 浪費
        elif not threat_nearby and in_air:
            total_reward -= 1
            
    except:
        pass
        
    return total_reward

def power_up_reward(info, reward, prev_info):
    """
    收集道具獎勵
    - 吃到蘑菇變大
    - 吃到花獲得火焰能力
    """
    total_reward = reward
    
    # 通過分數變化來判定是否獲得道具
    # 蘑菇/花 = 1000分, 1UP = 額外生命
    score_diff = info['score'] - prev_info['score']
    
    if score_diff == 1000:
        total_reward += 50  # 吃到道具獎勵
    
    # 獲得額外生命
    if info['life'] > prev_info['life']:
        total_reward += 200  # 1UP 大獎勵
        
    return total_reward

# ==================== 組合獎勵函數 ====================

# 追蹤歷史最大 X 位置 (用於檢測是否突破障礙)
_max_x_reached = {'value': 0}

def reset_max_x():
    """重置最大 X 位置 (在每個 episode 開始時調用)"""
    _max_x_reached['value'] = 0

def obstacle_breakthrough_reward(info, reward, prev_info):
    """
    障礙物突破獎勵
    - 當 Mario 突破歷史最遠距離時，給予額外獎勵
    - 這表示他可能成功跳過了水管或其他障礙
    """
    total_reward = reward
    current_x = info['x_pos']
    
    # 如果突破了歷史最遠距離
    if current_x > _max_x_reached['value']:
        breakthrough_distance = current_x - _max_x_reached['value']
        
        # 大幅突破 = 可能跳過了障礙物
        if breakthrough_distance > 20:
            total_reward += breakthrough_distance * 2  # 大獎勵
        else:
            total_reward += breakthrough_distance * 0.5  # 小獎勵
            
        _max_x_reached['value'] = current_x
    
    return total_reward

def stagnation_penalty(info, reward, prev_info):
    """
    停滯懲罰
    - 如果 Mario 在同一位置停留太久（可能卡在水管前）
    - 給予懲罰促使他嘗試跳躍
    """
    total_reward = reward
    
    x_diff = abs(info['x_pos'] - prev_info['x_pos'])
    
    # 如果幾乎沒有移動
    if x_diff < 2:
        total_reward -= 2  # 懲罰停滯
    
    return total_reward

def jump_attempt_reward(env, info, reward, prev_info):
    """
    跳躍嘗試獎勵
    - 當 Mario 接近歷史最遠位置時，鼓勵跳躍
    - 這有助於學習在障礙物前跳躍
    """
    total_reward = reward
    
    try:
        current_x = info['x_pos']
        in_air = is_mario_in_air(env)
        
        # 如果接近歷史最遠位置且在空中
        if current_x >= _max_x_reached['value'] - 30 and in_air:
            total_reward += 5  # 獎勵在挑戰區域跳躍
            
        # 如果正在向前移動且跳躍
        x_diff = info['x_pos'] - prev_info['x_pos']
        if x_diff > 0 and in_air:
            total_reward += 3  # 獎勵前進中的跳躍
            
    except:
        pass
        
    return total_reward
# 追蹤歷史敵人數量 (用於檢測擊殺)
_prev_enemy_count = {'value': 0}

def reset_enemy_tracking():
    """重置敵人追蹤 (在每個 episode 開始時調用)"""
    _prev_enemy_count['value'] = 0

def count_active_enemies(env):
    """計算當前活躍的敵人數量"""
    count = 0
    try:
        base_env = env.unwrapped
        for i in range(5):
            is_drawn = base_env.ram[RAM_ADDRESSES['enemy_drawn'][i]]
            if is_drawn:
                count += 1
    except:
        pass
    return count

def kill_enemy_reward(env, info, reward, prev_info):
    """
    擊殺敵人獎勵
    - 通過追蹤敵人數量變化來判定擊殺
    - 通過分數變化來確認擊殺（踩死敵人得 100 分）
    """
    total_reward = reward
    
    try:
        # 方法 1: 通過分數變化判斷擊殺
        # 踩死 Goomba = 100 分
        # 踩死 Koopa = 100 分
        # 連續踩殺 = 200, 400, 800, 1000...
        score_diff = info['score'] - prev_info['score']
        
        # 擊殺得分模式
        kill_scores = [100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000]
        
        if score_diff in kill_scores:
            # 根據連殺數給予不同獎勵
            kill_index = kill_scores.index(score_diff) if score_diff in kill_scores else 0
            kill_bonus = 30 + (kill_index * 20)  # 基礎 30，連殺加成
            total_reward += kill_bonus
            
        # 方法 2: 通過敵人數量變化判斷
        current_enemy_count = count_active_enemies(env)
        prev_count = _prev_enemy_count['value']
        
        # 如果敵人數量減少了（可能是擊殺或離開畫面）
        if current_enemy_count < prev_count and score_diff >= 100:
            # 確認是擊殺（有分數增加）
            enemies_killed = prev_count - current_enemy_count
            total_reward += enemies_killed * 25  # 每個敵人額外 25 分獎勵
            
        _prev_enemy_count['value'] = current_enemy_count
        
    except:
        pass
    
    return total_reward

def stomp_kill_reward(env, info, reward, prev_info):
    """
    踩殺獎勵 (專門獎勵從上方踩死敵人)
    - 檢測 Mario 是否在空中下落並接近敵人
    """
    total_reward = reward
    
    try:
        mario_x, mario_y = get_mario_screen_pos(env)
        enemies = get_enemies_info(env)
        in_air = is_mario_in_air(env)
        
        score_diff = info['score'] - prev_info['score']
        
        # 如果在空中並且得分增加（可能踩死敵人）
        if in_air and score_diff >= 100:
            # 檢查是否有敵人在 Mario 附近下方
            for enemy in enemies:
                dx = abs(int(enemy['x']) - int(mario_x))
                dy = int(mario_y) - int(enemy['y'])  # Mario 的 y 比敵人小 = 在上方
                
                # 如果 Mario 在敵人上方且距離很近
                if dx < 20 and dy < 0:
                    total_reward += 50  # 踩殺額外獎勵
                    break
                    
    except:
        pass
    
    return total_reward

def calculate_smart_reward(env, info, reward, prev_info):
    """
    完整的智慧獎勵計算
    結合所有獎勵函數
    """
    total = reward
    
    # 基本獎勵
    total = get_coin_reward(info, total, prev_info)
    total = distance_x_offset_reward(info, total, prev_info)
    total = distance_y_offset_reward(info, total, prev_info)
    total = monster_score_reward(info, total, prev_info)
    total = final_flag_reward(info, total)
    
    # 進階智慧獎勵
    total = enemy_avoidance_reward(env, info, total, prev_info)
    total = survival_time_reward(info, total, prev_info)
    total = speed_reward(info, total, prev_info)
    total = jump_timing_reward(env, info, total, prev_info)
    total = power_up_reward(info, total, prev_info)
    
    # 🆕 障礙物相關獎勵
    total = obstacle_breakthrough_reward(info, total, prev_info)
    total = stagnation_penalty(info, total, prev_info)
    total = jump_attempt_reward(env, info, total, prev_info)
    
    # 🆕 擊殺敵人獎勵
    total = kill_enemy_reward(env, info, total, prev_info)
    total = stomp_kill_reward(env, info, total, prev_info)
    
    # 🆕 坑洞跨越獎勵
    total = hole_crossing_reward(env, info, total, prev_info)
    total = fall_penalty(info, total, prev_info)
    
    return total

# ==================== 坑洞跨越獎勵系統 ====================

# Super Mario Bros 1-1 關卡的坑洞位置 (x_pos 範圍)
# 這些是已知的坑洞區域
LEVEL_1_1_HOLES = [
    (1550, 1584),   # 第一個坑洞
    (1712, 1744),   # 第二個坑洞 (兩個水管之間)
    (2480, 2550),   # 第三個坑洞 (較大的)
    (2832, 2896),   # 第四個坑洞
]

# 追蹤已跨越的坑洞
_crossed_holes = {'value': set()}

def reset_hole_tracking():
    """重置坑洞追蹤 (在每個 episode 開始時調用)"""
    _crossed_holes['value'] = set()

def is_near_hole(x_pos, distance=50):
    """檢查是否接近坑洞"""
    for i, (hole_start, hole_end) in enumerate(LEVEL_1_1_HOLES):
        # 在坑洞前方一定距離
        if hole_start - distance <= x_pos < hole_start:
            return True, i, 'approaching'
        # 在坑洞上方
        elif hole_start <= x_pos <= hole_end:
            return True, i, 'over'
        # 剛剛跨過坑洞
        elif hole_end < x_pos <= hole_end + 30:
            return True, i, 'crossed'
    return False, -1, None

def hole_crossing_reward(env, info, reward, prev_info):
    """
    坑洞跨越獎勵
    - 接近坑洞時跳躍給予獎勵
    - 成功跨越坑洞給予大獎勵
    - 在坑洞上空給予鼓勵
    """
    total_reward = reward
    
    try:
        x_pos = info['x_pos']
        prev_x = prev_info['x_pos']
        in_air = is_mario_in_air(env)
        
        near_hole, hole_idx, status = is_near_hole(x_pos)
        prev_near, prev_idx, prev_status = is_near_hole(prev_x)
        
        if near_hole:
            if status == 'approaching':
                # 接近坑洞時跳躍
                if in_air:
                    total_reward += 15  # 獎勵在坑洞前跳躍
                # 接近但沒跳 = 小懲罰（鼓勵跳躍）
                elif x_pos > prev_x:  # 正在前進
                    total_reward -= 2
                    
            elif status == 'over':
                # 在坑洞上空
                if in_air:
                    total_reward += 20  # 獎勵在坑洞上空飛行
                # 注意: 如果不在空中但在坑洞位置，可能快掉下去了
                    
            elif status == 'crossed':
                # 成功跨越坑洞！
                if hole_idx not in _crossed_holes['value']:
                    _crossed_holes['value'].add(hole_idx)
                    total_reward += 100  # 大獎勵！成功跨越坑洞
                    print(f"🕳️ Successfully crossed hole {hole_idx + 1}!")
                    
    except Exception as e:
        pass
    
    return total_reward

def fall_penalty(info, reward, prev_info):
    """
    掉落懲罰
    - 檢測 Mario 是否掉入坑洞（y_pos 突然增加很多）
    - 生命減少時給予大懲罰
    """
    total_reward = reward
    
    try:
        y_pos = info['y_pos']
        prev_y = prev_info['y_pos']
        x_pos = info['x_pos']
        
        # 檢查是否在坑洞區域
        near_hole, hole_idx, status = is_near_hole(x_pos)
        
        # Mario 的 y_pos 在 NES 中是從上到下增加的
        # 如果 y_pos 突然大幅增加，可能正在掉落
        y_drop = y_pos - prev_y
        
        if near_hole and status in ['approaching', 'over']:
            # 如果在坑洞區域且 y 值增加（下落）
            if y_drop > 10:
                total_reward -= 30  # 懲罰掉落
                
        # 如果生命減少（死亡）
        if info['life'] < prev_info['life']:
            # 檢查是否是掉入坑洞死亡（而不是被敵人殺死）
            if near_hole:
                total_reward -= 200  # 掉入坑洞的大懲罰
            else:
                total_reward -= 100  # 普通死亡懲罰
                
    except:
        pass
    
    return total_reward

def jump_over_gap_reward(env, info, reward, prev_info):
    """
    跳躍跨越獎勵
    - 專門獎勵長距離跳躍（跨越坑洞需要的）
    """
    total_reward = reward
    
    try:
        x_pos = info['x_pos']
        prev_x = prev_info['x_pos']
        in_air = is_mario_in_air(env)
        
        # 計算水平移動距離
        x_diff = x_pos - prev_x
        
        # 如果在空中且快速前進
        if in_air and x_diff > 3:
            total_reward += x_diff * 0.5  # 獎勵空中的水平移動
            
    except:
        pass
    
    return total_reward

#===============to do==========================================
# 你可以繼續添加更多獎勵函數：
# - 躲避子彈獎勵
# - 水管探索獎勵  
# - 連續擊敗敵人獎勵 ✅ (已新增 kill_enemy_reward)
# - 坑洞跨越獎勵 ✅ (已新增 hole_crossing_reward)
# - 無傷通關獎勵
# ============================================================



