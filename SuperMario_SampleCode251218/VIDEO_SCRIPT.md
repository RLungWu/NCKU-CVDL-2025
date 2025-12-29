# Super Mario Bros 強化學習專案 - 影片腳本

## 📹 影片結構建議 (約 8-10 分鐘)

---

## 第一部分：專案介紹 (約 1 分鐘)

### 講稿：

大家好，這是我的 Super Mario Bros 強化學習專案報告。

本專案使用兩種強化學習方法訓練 AI 代理玩超級瑪利歐兄弟第一關：
1. **DQN (Deep Q-Network)** - 基於值函數的方法
2. **PPO (Proximal Policy Optimization)** - 基於策略的方法

**專案架構包含以下程式碼檔案：**

| 檔案 | 說明 |
|------|------|
| `run.py` / `run_parallel.py` | DQN 訓練腳本 |
| `run_ppo.py` | PPO 訓練腳本 |
| `eval.py` | DQN 評估腳本 |
| `eval_ppo.py` | PPO 評估腳本 |
| `reward.py` | 自定義獎勵函數 ⭐ 核心創新 |
| `model.py` | CustomCNN 神經網路 |
| `DQN.py` | DQN 演算法實現 |
| `utils.py` | 工具函數 |

---

## 第二部分：Custom Reward 函數設計 (約 2-3 分鐘)

### 講稿：

接下來介紹我設計的自定義獎勵函數，這是本專案的核心創新。

**設計動機：**
原始的遊戲獎勵只包含分數變化，這對於學習複雜行為是不夠的。
我的目標是讓 Mario 學會：跳過敵人、跨越坑洞、快速前進。

**獎勵函數架構 (reward.py)：**

### 1. 集中式超參數配置
```python
REWARD_CONFIG = {
    # === 基本獎勵 ===
    'coin_reward': 10,              # 每個硬幣的獎勵
    'forward_reward': 1.0,          # 向前移動的獎勵
    'flag_reward': 1000,            # 到達終點旗幟的獎勵
    
    # === 敵人相關獎勵 ===
    'kill_base_reward': 20,         # 擊殺敵人的基礎獎勵
    'stomp_kill_bonus': 5,          # 踩殺的額外獎勵
    
    # === 坑洞跨越獎勵 ===
    'hole_crossed_reward': 100,     # 成功跨越坑洞的獎勵
    'fall_death_penalty': -200,     # 掉入坑洞死亡的懲罰
    ...
}
```

### 2. NES RAM 直接讀取
```python
RAM_ADDRESSES = {
    'mario_x_pos': 0x006D,          # Mario 螢幕上 X 位置
    'mario_floating': 0x001D,       # Mario 是否在空中
    'enemy_drawn': [0x000F, ...],   # 敵人是否存在
    'enemy_x_pos': [0x0087, ...],   # 敵人 X 位置
}
```

**技術亮點：** 我直接讀取 NES 模擬器的 RAM 來獲取敵人的精確座標，
這比圖像識別更準確、更快速。

### 3. 五大類獎勵函數

| 類別 | 函數 | 說明 |
|------|------|------|
| 基本獎勵 | `get_coin_reward()`, `distance_x_offset_reward()` | 硬幣、前進 |
| 敵人相關 | `enemy_avoidance_reward()`, `kill_enemy_reward()` | 迴避、擊殺 |
| 坑洞跨越 | `hole_crossing_reward()`, `fall_penalty()` | 跳過坑洞 |
| 障礙物突破 | `obstacle_breakthrough_reward()` | 突破新區域 |
| 智慧跳躍 | `jump_timing_reward()` | 正確時機跳躍 |

### 4. 坑洞位置定義
```python
LEVEL_1_1_HOLES = [
    (1550, 1584),   # 第一個坑洞
    (1712, 1744),   # 第二個坑洞
    (2480, 2550),   # 第三個坑洞
    (2832, 2896),   # 第四個坑洞
]
```

---

## 第三部分：DQN 訓練與結果 (約 2 分鐘)

### 講稿：

首先展示 DQN 的訓練過程和結果。

**DQN 訓練配置：**
| 參數 | 值 |
|------|-----|
| 平行環境 | 8 個 |
| Batch Size | 128 |
| Memory Size | 100,000 |
| Epsilon Decay | 1.0 → 0.1 |
| 總 Episodes | 2,000 |
| 訓練時間 | ~45 分鐘 |

**[播放 DQN 訓練錄影片段]**

**訓練成果：**
- Episode 5: 最佳距離達到 1124
- Episode 66: 首次跨越坑洞 1 和 2 🕳️
- Episode 147: 跨越坑洞 3 和 4 🕳️
- Episode 152: **首次通關！** 最佳距離達到 3161 🏆

**DQN 測試結果：**
- 最佳獎勵: 2962 分
- 最遠距離: 3161 (通關)
- 平均獎勵: 約 600 分

---

## 第四部分：PPO 實現與比較 (約 2 分鐘)

### 講稿：

接下來介紹 PPO (Proximal Policy Optimization) 的實現。

**為什麼嘗試 PPO？**
- PPO 是目前最流行的策略梯度方法之一
- 相比 DQN 更穩定、更容易調參
- 想比較 Value-based 和 Policy-based 方法的表現

**PPO 網路架構 (Actor-Critic)：**
```python
class ActorCritic(nn.Module):
    # 共享卷積層
    self.conv = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=8, stride=4),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
    )
    # Actor (策略) 和 Critic (價值) 分開輸出
    self.actor = nn.Linear(512, n_actions)
    self.critic = nn.Linear(512, 1)
```

**PPO 配置：**
| 參數 | 值 |
|------|-----|
| 學習率 | 2.5e-4 |
| Clip Epsilon | 0.1 |
| 熵係數 | 0.02 |
| GAE Lambda | 0.95 |
| 總步數 | 2,000,000 |

**DQN vs PPO 比較：**
| 方面 | DQN | PPO |
|------|-----|-----|
| 類型 | Value-based | Policy-based |
| 穩定性 | 需要經驗回放 | 本身較穩定 |
| 樣本效率 | 較高 (off-policy) | 較低 (on-policy) |
| 訓練速度 | 較慢 | 較快 (per step) |
| 最終表現 | 通關 ✅ | 待訓練更久 |

---

## 第五部分：測試結果展示 (約 1-2 分鐘)

### 講稿：

現在展示訓練完成的模型表現。

**[播放 DQN 測試錄影]**

使用最佳距離模型 `best_distance_3161_ep_152.pth` 進行測試。

**觀察到的行為：**
1. ✅ Mario 學會了快速向前移動
2. ✅ 能夠跳過敵人 (Goomba)
3. ✅ 能夠跨越水管
4. ✅ 成功跨越所有四個坑洞
5. ✅ 到達終點旗幟

**關鍵時刻：**
- x=1550-1584: 跨越第一個坑洞
- x=1712-1744: 跨越第二個坑洞
- x=2480-2550: 跨越第三個坑洞 (最大的)
- x=2832-2896: 跨越第四個坑洞
- x=3161: 🏆 到達終點旗幟！

---

## 第六部分：觀察與結論 (約 1 分鐘)

### 講稿：

**主要觀察：**

1. **獎勵塑形的重要性**
   - 單純使用遊戲原生獎勵很難學會複雜行為
   - 自定義獎勵可以引導 AI 學習特定技能
   - 坑洞獎勵讓 Mario 學會跳過危險區域

2. **RAM 讀取的優勢**
   - 直接讀取記憶體比圖像識別更精確
   - 可以獲得敵人的準確位置資訊
   - 計算效率更高

3. **平行訓練的效率**
   - 8 個環境平行訓練，大幅加速學習
   - 45 分鐘內完成通關

4. **DQN vs PPO**
   - DQN 在這個任務中表現更好
   - PPO 需要更多訓練時間
   - 兩種方法都能從智慧獎勵系統受益

**總結：**
透過精心設計的獎勵函數，我們成功訓練出能夠通關 Super Mario Bros 1-1 的 AI。
這個專案展示了獎勵塑形在強化學習中的重要性。

謝謝觀看！

---

## 📦 繳交檔案清單

### 程式碼 (6+ 個檔案)
1. `run_parallel.py` - DQN 平行訓練腳本
2. `run_ppo.py` - PPO 訓練腳本
3. `eval.py` - DQN 評估腳本
4. `eval_ppo.py` - PPO 評估腳本
5. `reward.py` - 自定義獎勵函數 ⭐
6. `model.py` - 神經網路架構
7. `DQN.py` - DQN 演算法
8. `utils.py` - 工具函數

### 權重檔 (1 個)
- `best_distance_3161_ep_152.pth` - DQN 通關模型 🏆

### 影片 (1 個)
- 包含以上所有講解內容
- 訓練過程錄影
- 測試過程錄影

---

## 🎬 錄影建議

1. **螢幕錄製軟體**: OBS Studio 或 Kazam (Linux)
   ```bash
   sudo apt install kazam
   # 或
   sudo apt install obs-studio
   ```

2. **訓練錄影**: 可以快轉播放訓練過程
   ```bash
   uv run python run_parallel.py
   ```

3. **測試錄影**: 正常速度展示 Mario 的表現
   ```bash
   uv run python eval.py
   ```

4. **程式碼展示**: 用 VS Code 展示重要的程式碼段落

---

## 📝 重要程式碼段落展示

### 1. reward.py 的 REWARD_CONFIG
展示集中式超參數配置

### 2. calculate_smart_reward() 函數
展示如何組合所有獎勵函數

### 3. RAM 讀取敵人位置
```python
def get_enemies_info(env):
    enemies = []
    for i in range(5):
        is_drawn = base_env.ram[RAM_ADDRESSES['enemy_drawn'][i]]
        if is_drawn:
            enemy = {
                'type': base_env.ram[RAM_ADDRESSES['enemy_type'][i]],
                'x': base_env.ram[RAM_ADDRESSES['enemy_x_pos'][i]],
                'y': base_env.ram[RAM_ADDRESSES['enemy_y_pos'][i]],
            }
            enemies.append(enemy)
```

### 4. 坑洞位置定義與獎勵
```python
LEVEL_1_1_HOLES = [
    (1550, 1584),   # 第一個坑洞
    (1712, 1744),   # 第二個坑洞
    (2480, 2550),   # 第三個坑洞
    (2832, 2896),   # 第四個坑洞
]

def hole_crossing_reward(env, info, reward, prev_info):
    # 成功跨越坑洞給予 100 分獎勵！
    if hole_idx not in _crossed_holes['value']:
        _crossed_holes['value'].add(hole_idx)
        total_reward += 100
```

### 5. PPO Actor-Critic 網路
```python
class ActorCritic(nn.Module):
    def get_action(self, state, deterministic=False):
        policy, value = self.forward(state)
        probs = torch.softmax(policy, dim=-1)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()
        return action
```

---

## 🎯 時間分配建議

| 段落 | 時間 | 內容 |
|------|------|------|
| 專案介紹 | 1 分鐘 | 架構、檔案說明 |
| Reward 設計 | 2-3 分鐘 | 核心創新 ⭐ |
| DQN 訓練 | 2 分鐘 | 過程、結果 |
| PPO 比較 | 2 分鐘 | 實現、差異 |
| 測試展示 | 1-2 分鐘 | 實際表現 |
| 結論 | 1 分鐘 | 觀察、總結 |
| **總計** | **8-10 分鐘** | |
