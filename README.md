# posttrain-lab

小模型（Qwen3-1.7B）后训练全流程复现：数据 → SFT → DPO → GRPO/RLVR → 评测。

> **这个仓库的目的不只是"跑通"，而是"每一步都能讲清楚为什么"。**
> 代码里凡是标了 `# 知识点` 的注释，都是面试会问的地方。

---

## 实验结果（2026-10-31，第一轮）

Qwen3-1.7B / GSM8K / LoRA(r=16, α=32) / 2000 条训练样本 / 100 步 / greedy 评测 100 条

| 组 | lr | pass@1 | 输出词数 | `####` 格式率 | 截断率 |
|---|---|---|---|---|---|
| base（未微调） | — | **0.830** | 139.2 | 0.720 | 0.030 |
| SFT（原始 solution） | 1e-4 | 0.580 | 47.5 | 1.000 | 0.000 |
| SFT（清洗 `<<>>` 注释） | 1e-4 | 0.580 | 46.4 | 0.980 | 0.020 |
| SFT（清洗 + 降 lr） | 2e-5 | 0.510 | 77.9 | 0.900 | 0.100 |

**结论：这是一个负结果。** 格式率从 72% 提到 98–100%，train loss 从 2.4 降到 0.42，
但 pass@1 掉了 25 个点。逐项排除后确认根因是 **训练数据的输出风格（51 词紧凑解答）
覆盖了模型原生的长推理（139 词）**，而非数据噪声或学习率过高：

- 清洗 `<<48/2=24>>` 计算注释 → 分数不变（证伪"数据污染"假设）
- lr 1e-4 → 2e-5 → 分数更低且出现复读循环（证伪"学习率过高"假设）

原因：GSM8K 对 Qwen3-1.7B 几近饱和（base 83%），且官方 solution 的推理链长度
只有模型原生输出的 1/3。**在已饱和任务上做"风格覆盖式" SFT 是负收益。**

下一步：换更难的主指标（MATH-500）+ 用模型自己的正确输出做拒绝采样（RFT）。

---

## 0. 平台速查（智川云 RiverMind，开工前 5 分钟）

官网 https://gpuhome.cc ｜ 文档 https://docs.gpuhome.cc

| 项 | 值 |
|---|---|
| 数据盘 | **`/root/rivermind-data/`**（默认 50G，扩容 0.02 元/GB/日） |
| 文件存储 | **`/root/rivermind-fs/`**（跨实例共享，20G 内免费，**释放实例也不丢**） |
| 系统盘 | `/` 只有 30G，**别把模型缓存放这里** |
| 无卡启动 | 先在控制台关机 → 「更多选项」→「无卡启动」，**0.1 元/时** |
| 关机 | 数据保留，但**连续关机满 7 天实例会被释放**，数据盘清空 |
| Docker | 容器内**不支持**，要 Docker 得租裸金属 |

**第一件事：把这些写进 `~/.bashrc`**（否则缓存会悄悄写满 30G 系统盘）

```bash
mkdir -p /root/rivermind-data/{models,cache,datasets,runs} /root/rivermind-fs/posttrain-lab
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/root/rivermind-data/cache/huggingface
export MODELSCOPE_CACHE=/root/rivermind-data/cache/modelscope
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
EOF
source ~/.bashrc
df -h / /root/rivermind-data
```

> 智川云**没有** `/etc/network_turbo` 这个脚本，别照 AutoDL 的教程敲。
> 优先从 ModelScope 下模型/数据集（国内直连），HF 才走 `hf-mirror`。

---

## 1. 目录

```
posttrain-lab/
├── common.py              # seed / 日志 / jsonl / 配置落盘 / 共用 prompt 模板
├── data/
│   └── prepare_gsm8k.py   # 下载 + 转成统一 jsonl
├── sft/
│   └── train_sft.py       # SFT 训练（LoRA + assistant-only loss）
├── eval/
│   ├── extract.py         # 答案抽取（评测的命门，带单测）
│   └── run_eval.py        # 评测主程序（pass@1 / 长度 / 截断率 / 格式率）
└── notes/                 # env.md、budget.md、实验笔记
```

## 2. 快速开始

```bash
# 0. 环境（无卡启动模式就能装）
pip install -r requirements.txt

# 1. 数据（无卡启动，不占 GPU）
python data/prepare_gsm8k.py --out data/gsm8k

# 2. 自测抽取器（10 秒，别跳过）
python eval/extract.py --selftest        # 要看到 PASS

# 3. 训练（切 GPU 模式）
python sft/train_sft.py \
  --model /root/rivermind-data/models/Qwen3-1.7B \
  --data data/gsm8k/train.jsonl \
  --out /root/rivermind-data/runs/sft-smoke \
  --max_len 1024 --bs 2 --accum 8 --max_steps 100 --limit 2000

# 4. 评测
python eval/run_eval.py \
  --model /root/rivermind-data/models/Qwen3-1.7B \
  --adapter /root/rivermind-data/runs/sft-smoke/final \
  --data data/gsm8k/test.jsonl --limit 200 \
  --out results/sft_200.jsonl

# 5. 配好环境后去控制台「保存镜像」，下次开机不用重装
```

## 3. 三个约定

1. **所有路径走参数，不写死。** 换机器不改代码。
2. **每次实验把配置 dump 成 json**（`--out` 目录下的 `run_config.json`），报告里的数字只从那里取。
3. **结果落三份**：`/root/rivermind-fs/`、本地电脑、GitHub。只放数据盘等于没备份。
