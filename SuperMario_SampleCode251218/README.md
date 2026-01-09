# 🎮 Super Mario Bros DQN 強化學習

使用 Deep Q-Network (DQN) 訓練 AI 玩 Super Mario Bros 的強化學習專案。

## 📋 專案概述

本專案實現了一個基於 DQN 的強化學習 Agent，通過自定義獎勵函數來訓練 AI 學習玩 Super Mario Bros。

### 主要特色
- ✅ **16 種自定義獎勵/懲罰規則**（快速通關優化版）
- ✅ 支援 Normal / Extreme 兩種訓練模式
- ✅ Epsilon Decay 探索策略
- ✅ Frame Skip 加速訓練
- ✅ 自動錄製評估影片

---

## 🚀 快速開始

### 環境安裝

```bash
# 建立虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install torch gym gym-super-mario-bros nes-py opencv-python tqdm
```

### 訓練模型

```bash
python run.py
```

### 評估模型（錄製影片）

```bash
python eval.py
# 影片輸出：mario_eval.mp4
```

---

## 📁 檔案結構

```
SuperMario_SampleCode251218/
├── run.py          # 訓練腳本
├── eval.py         # 評估腳本（可錄製影片）
├── reward.py       # 獎勵函數定義（16種規則）
├── model.py        # CNN 模型架構
├── DQN.py          # DQN 演算法實現
├── utils.py        # 工具函數
├── liang_test/     # Normal 模式權重
└── liang_test_extreme/  # Extreme 模式權重
```

---

## 🎯 獎勵函數設計

### 設計原則
1. **前進優先**：前進獎勵最高，鼓勵 AI 向右移動
2. **速度獎勵**：快速移動和連續前進給予額外獎勵
3. **避免矛盾**：下落懲罰設為 0，避免 AI 不敢跳躍
4. **死亡懲罰**：死亡給予最大懲罰，讓 AI 學會避險
5. **通關激勵**：完成關卡給予巨大獎勵

### 獎勵/懲罰規則（共 16 種）

#### 核心獎勵函數（原有 9 種）

| 函數 | Normal | Extreme | 說明 |
|------|--------|---------|------|
| `distance_x_offset_reward` | +30 | +100 | 前進獎勵（核心驅動力） |
| `final_flag_reward` | +1000 | +2000 | 通關獎勵 |
| `get_coin_reward` | +8 | +20 | 收集硬幣 |
| `distance_y_offset_reward` | +3/0 | +10/0 | 跳躍/下落 |
| `monster_score_reward` | +10 | +30 | 擊敗敵人 |
| `life_bonus_reward` | +100 | +300 | 獲得 1UP |
| `death_penalty` | -300 | -500 | 死亡懲罰（核心） |
| `stagnation_penalty` | -20 | -50 | 停滯懲罰 |
| `time_efficiency_reward` | -8 | -15 | 時間浪費懲罰 |

#### 快速通關專用獎勵函數（新增 7 種）

| 函數 | Normal | Extreme | 說明 |
|------|--------|---------|------|
| `speed_bonus_reward` | +15 | +50 | 快速前進（跑動）獎勵 |
| `momentum_bonus_reward` | +20 | +60 | 連續前進 5 步以上獎勵 |
| `new_distance_record_reward` | 動態 | 動態 | 達到最遠距離獎勵（每格 +0.5）|
| `obstacle_clear_reward` | +25 | +80 | 成功越過障礙/敵人 |
| `progress_efficiency_reward` | +5/-10 | +5/-30 | 進度效率獎勵/懲罰 |
| `precise_jump_reward` | +5 | +5 | 精準跳躍（高跳+前進）|
| `perfect_clear_bonus` | 動態 | 動態 | 通關時剩餘時間獎勵（每秒 ×2）|

### 模式切換

在 `reward.py` 中修改：
```python
EXTREME_MODE = True   # 極端模式（推薦快速學習）
EXTREME_MODE = False  # 正常模式
```

---

## ⚙️ 訓練超參數

| 參數 | 數值 | 說明 |
|------|------|------|
| `LR` | 0.005 | 學習率（加速收斂） |
| `BATCH_SIZE` | 64 | 批次大小 |
| `GAMMA` | 0.99 | 折扣因子（重視長期獎勵） |
| `EPSILON_START` | 1.0 | 初始探索率 |
| `EPSILON_END` | 0.1 | 最終探索率 |
| `EPSILON_DECAY` | 0.995 | 探索率衰減 |
| `TOTAL_TIMESTEPS` | 2000 | 訓練回合數 |
| `FRAME_SKIP` | 2 | 跳幀數 |

### Epsilon Decay 說明

```
Episode 1:    ε=1.000 → 100% 隨機探索
Episode 100:  ε=0.606 → 60% 探索
Episode 500:  ε=0.082 → 8% 探索（開始利用策略）
```

---

## 📊 實驗結果

### 訓練表現

| 模式 | 最佳獎勵 | x_pos 最遠 | 說明 |
|------|----------|------------|------|
| Normal | 37,291 | ~719 | 較穩定的學習曲線 |
| Extreme | 106,766 | ~1072 | 較快速的學習，但波動較大 |

### 評估結果

| 測試模型 | 總獎勵 | 最終 x_pos | Frames |
|----------|--------|------------|--------|
| Extreme (step_1368) | 1,051 | 1,072 | 432 |
| Normal (step_1417) | 494 | 719 | 3,478 |

### 觀察結論

1. **Extreme 模式學習更快**：x_pos 可達 1072，超越第一個敵人
2. **Normal 模式更穩定**：雖然進度較慢但不易崩潰
3. **速度獎勵有效**：新增的 `speed_bonus` 和 `momentum_bonus` 鼓勵持續快速前進

---

## 🔬 分析與討論

### 為什麼 16 個獎勵函數比 9 個效果好？

1. **速度激勵更明確**：`speed_bonus` 直接獎勵跑動（按住 B 鍵）
2. **連續性獎勵**：`momentum_bonus` 鼓勵持續動量，避免停停走走
3. **進度追蹤**：`new_distance_record_reward` 激勵探索新區域
4. **完美通關激勵**：`perfect_clear_bonus` 根據剩餘時間給予額外獎勵

### 可能的改進方向

- 增加訓練回合數（10,000+）
- 使用更先進的演算法（PPO、A3C）
- 課程學習（Curriculum Learning）
- 優先經驗回放（Prioritized Experience Replay）

---

## 📚 參考資料

- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

---

## 👨‍💻 作者

NCKU CVDL 2025 課程作業
