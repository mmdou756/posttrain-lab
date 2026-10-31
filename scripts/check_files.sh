#!/usr/bin/env bash
# 文件版本校验：确认实例上的代码和最新版一致。
# 用法： cd /root/posttrain-lab && bash scripts/check_files.sh
# 全部 OK 才能跑训练；出现 OLD 就说明那个文件没更新，重新上传即可。

cd "$(dirname "$0")/.." || exit 1
fail=0

check() {  # $1=文件 $2=应该出现的字符串 $3=说明
  if [ ! -f "$1" ]; then
    printf "MISSING  %-24s %s\n" "$1" "$3"; fail=1; return
  fi
  if grep -q "$2" "$1"; then
    printf "OK       %-24s %s\n" "$1" "$3"
  else
    printf "OLD      %-24s %s\n" "$1" "$3"; fail=1
  fi
}

echo "==================== 文件版本校验 ===================="
check "common.py"              "load_causal_lm"        "含 transformers 5.x 兼容加载器"
check "common.py"              "build_prompt"          "含共用 prompt 模板"
check "sft/train_sft.py"       "load_causal_lm"        "走版本自适应加载"
check "data/prepare_gsm8k.py"  "AI-ModelScope/gsm8k"   "走 ModelScope（HF 不通）"
check "eval/run_eval.py"       "load_causal_lm"        "走版本自适应加载"
check "eval/extract.py"        "selftest"              "含答案抽取单测"

echo
echo "==================== 数据是否就位 ===================="
for f in data/gsm8k/train.jsonl data/gsm8k/test.jsonl; do
  if [ -f "$f" ]; then
    printf "OK       %-28s %s 行\n" "$f" "$(wc -l < "$f")"
  else
    printf "MISSING  %-28s 还没跑 prepare_gsm8k.py\n" "$f"; fail=1
  fi
done

echo
echo "==================== 依赖 ===================="
python - <<'PY'
import importlib.metadata as md
for n in ["torch", "transformers", "peft", "trl", "datasets", "accelerate", "modelscope"]:
    try:
        print(f"  {n:14s} {md.version(n)}")
    except Exception:
        print(f"  {n:14s} -")
PY

echo
if [ "$fail" -eq 0 ]; then
  echo "==== 全部通过，可以开跑 ===="
else
  echo "==== 上面标 MISSING / OLD 的文件需要重新上传 ===="
  echo "最快的做法：把本地整个 posttrain-lab 压成 zip 拖上去，覆盖解压。"
fi
