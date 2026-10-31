#!/usr/bin/env bash
# 环境体检脚本 —— 在智川云实例里跑，把输出整段贴给我，我就能替你排错。
# 用法： bash scripts/check_env.sh
# 说明：全部是只读命令，不改任何东西，无卡启动 / GPU 模式下都能跑。

echo "==================== 1. 机器与系统 ===================="
uname -a
grep -E 'PRETTY_NAME|VERSION_ID' /etc/os-release 2>/dev/null
echo "CPU 核数: $(nproc)"
free -g 2>/dev/null | head -2

echo
echo "==================== 2. GPU ===================="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 | head -5
  nvidia-smi 2>&1 | grep -E 'CUDA Version' | head -1
else
  echo "没有 nvidia-smi（无卡启动模式下属正常）"
fi

echo
echo "==================== 3. Python 与关键包 ===================="
which python; python -V 2>&1
echo "--- pip 里已装的相关包 ---"
python - <<'PY' 2>&1
import importlib.metadata as md
want = ["torch","transformers","tokenizers","accelerate","peft","trl","datasets","vllm","modelscope","datasketch"]
for name in want:
    try:
        print(f"{name:14s} {md.version(name)}")
    except Exception:
        print(f"{name:14s} -")
PY
echo "--- torch 能否看到 GPU ---"
python - <<'PY' 2>&1
try:
    import torch
    print("torch", torch.__version__, "cuda_available =", torch.cuda.is_available(),
          "| cuda", torch.version.cuda)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch 导入失败:", type(e).__name__, e)
PY

echo
echo "==================== 4. 缓存与路径（最容易写错的地方）===================="
echo "HF_HOME            = ${HF_HOME:-<未设置>}"
echo "MODELSCOPE_CACHE   = ${MODELSCOPE_CACHE:-<未设置>}"
echo "HF_ENDPOINT        = ${HF_ENDPOINT:-<未设置>}"
echo "PIP_INDEX_URL      = ${PIP_INDEX_URL:-<未设置>}"
echo "当前目录           = $(pwd)"
echo "--- 磁盘 ---"
df -h / /root/rivermind-data /root/rivermind-fs 2>/dev/null
echo "--- 系统盘上有没有偷偷攒缓存 ---"
du -sh /root/.cache/huggingface /root/.cache/modelscope 2>/dev/null || echo "（没有旧缓存）"

echo
echo "==================== 5. 权重与数据 ===================="
ls -lh /root/rivermind-data/models 2>/dev/null || echo "models 目录不存在"
ls /root/rivermind-data/models/Qwen3-1.7B 2>/dev/null | head -10
ls -lh /root/rivermind-data/datasets 2>/dev/null | head -5

echo
echo "==================== 6. 网络 ===================="
for u in https://modelscope.cn https://hf-mirror.com https://pypi.tuna.tsinghua.edu.cn https://github.com; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 "$u" 2>/dev/null)
  echo "$u -> ${code:-超时}"
done

echo
echo "==================== 7. 工具 ===================="
echo "tmux: $(tmux -V 2>/dev/null || echo 未安装)"
echo "git : $(git --version 2>/dev/null || echo 未安装)"
echo "docker（容器内应不可用）: $(docker --version 2>/dev/null || echo 不可用，正常)"

echo
echo "==================== 8. 代码仓库 ===================="
ls -d ~/posttrain-lab 2>/dev/null && (cd ~/posttrain-lab && git log --oneline -3 2>/dev/null)
echo "自测抽取器:"
python ~/posttrain-lab/eval/extract.py --selftest 2>&1 | tail -3 || echo "（还没把仓库传上来）"

echo
echo "==================== 体检结束，把上面整段复制给我 ===================="
