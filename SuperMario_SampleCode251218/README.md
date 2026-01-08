# 🎮 Super Mario Bros DQN 強化學習

使用 Deep Q-Network (DQN) 訓練 AI 玩 Super Mario Bros 的強化學習專案。

## 📋 專案概述

本專案實現了一個基於 DQN 的強化學習 Agent，通過自定義獎勵函數來訓練 AI 學習玩 Super Mario Bros。

### 主要特色
- ✅ 9 種自定義獎勵/懲罰規則
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
├── reward.py       # 獎勵函數定義
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
2. **避免矛盾**：下落懲罰保持低值，避免 AI 不敢跳躍
3. **死亡懲罰**：死亡給予最大懲罰，讓 AI 學會避險

### 獎勵/懲罰規則（共 9 種）

| 函數 | Normal | Extreme | 說明 |
|------|--------|---------|------|
| `forward_reward` | +25 | +100 | 前進獎勵（核心） |
| `flag_reward` | +500 | +2000 | 通關獎勵 |
| `coin_reward` | +10 | +20 | 收集硬幣 |
| `jump_reward` | +5 | +10 | 跳躍獎勵 |
| `score_reward` | +15 | +30 | 擊敗敵人 |
| `life_bonus` | +100 | +300 | 獲得 1UP |
| `death_penalty` | -200 | -500 | 死亡懲罰（核心） |
| `backward_penalty` | -20 | -80 | 後退懲罰 |
| `stagnation_penalty` | -15 | -50 | 停滯懲罰 |

### 模式切換

在 `reward.py` 中修改：
```python
EXTREME_MODE = True   # 極端模式
EXTREME_MODE = False  # 正常模式
```

---

## ⚙️ 訓練超參數

| 參數 | 數值 | 說明 |
|------|------|------|
| `LR` | 0.00025 | 學習率 |
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

| 模式 | 最佳獎勵 | x_pos 最遠 |
|------|----------|------------|
| Normal | 14,087 | ~191 |
| Extreme | 48,806 | ~191 |

### 觀察結論

1. **AI 學會向右移動**：x_pos 從 40 增加到 191
2. **第一個敵人是瓶頸**：需要精確的跳躍時機
3. **需要更多訓練**：2000 回合不足以學會通過第一個敵人

---

## 🔬 分析與討論

### 為什麼無法通過第一個敵人？

1. **信用分配問題**：跳躍動作和「活著」之間的因果關係難以學習
2. **訓練量不足**：專業研究通常需要數百萬步訓練
3. **DQN 限制**：DQN 在需要精確時機的任務上效率較低

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
