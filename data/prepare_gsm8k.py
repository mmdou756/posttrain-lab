"""把 GSM8K 转成统一的 jsonl 格式（支持 ModelScope 下载，HF 不通也能用）。

输出格式（后面 SFT / DPO / GRPO 都吃这一种）：
  {"question": "...", "answer": "72", "solution": "... #### 72", "prompt": null}

知识点：为什么先统一成 jsonl？
训练脚本、评测脚本、数据统计脚本要吃同一种格式，否则每换一个环节就要改一次解析。
"数据格式统一"是工程上最省时间的一件事，也是面试里"你数据管线怎么设计的"的答案开端。

智川云实测：huggingface.co 和 hf-mirror.com 都不通，github.com 也不通，
但 modelscope.cn 通（302）。所以默认走 ModelScope。

用法：
  # 默认：从 ModelScope 下载（推荐，国内直连）
  python data/prepare_gsm8k.py --out data/gsm8k

  # 只转换已经下好的本地目录（不联网）
  python data/prepare_gsm8k.py --out data/gsm8k --local_dir /root/rivermind-data/datasets/gsm8k
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import write_jsonl

# ModelScope 上的 GSM8K 镜像（HF 的完整镜像，含 train/test 两个 split）
MS_DATASET = "AI-ModelScope/gsm8k"
DEFAULT_LOCAL = "/root/rivermind-data/datasets/gsm8k"


def parse_answer(solution: str):
    """从 GSM8K 的参考答案里抽最终数字。

    GSM8K 的格式固定为 "... #### 72"，所以先按 #### 切。
    抽不到就返回 None（这种样本后面会被过滤掉）。

    知识点：GSM8K 的答案里有逗号（1,000）和美元符号（$50），必须清洗，
    否则评测时模型答 "1000"、标准答案 "1,000"，你会误判成错。
    """
    if not isinstance(solution, str) or "####" not in solution:
        return None
    tail = solution.split("####")[-1].strip()
    tail = tail.replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", tail)
    return m.group(0) if m else None


def download_from_modelscope(local_dir):
    """把数据集拉到数据盘。返回本地目录。"""
    from modelscope import dataset_snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    print(f"[modelscope] downloading {MS_DATASET} -> {local_dir}")
    dataset_snapshot_download(MS_DATASET, local_dir=local_dir)
    return local_dir


# GSM8K 原始 solution 里带着 "<<"..."">>" 的计算注释，是给人工标注看的中间量：
#   "Natalia sold 48/2 = <<48/2=24>>24 clips in May."
# 直接拿它当训练目标，模型会学会输出这种标记（占 token、打断推理），
# 而且会被拉成这种"51 词紧凑风格"——实测 pass@1 从 83% 掉到 58%。
CALC_RE = re.compile(r"<<[^>]*>>")


def clean_solution(text, strip_calc=True):
    """把 solution 清成自然语言。实测这一步是 SFT 能不能涨的前提。"""
    if not isinstance(text, str):
        return ""
    if strip_calc:
        text = CALC_RE.sub("", text)
    # 清理多空格 / 多余空行
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guess_split(path):
    """从文件路径猜 split。train / test / val 三种写法都覆盖。"""
    p = path.lower()
    if "test" in p or "/validation" in p or "_val" in p or "valid" in p:
        return "test"
    if "train" in p:
        return "train"
    return None


def load_local(local_dir):
    """从本地目录读取，自动识别 parquet / json / jsonl。

    为什么写这么"笨"的自动识别：不同镜像仓库的目录结构不一样
    （有的 main/train-xxx.parquet，有的 train.json），写死路径迟早会崩。
    """
    from datasets import load_dataset

    files = [p for p in glob.glob(os.path.join(local_dir, "**", "*"), recursive=True)
             if os.path.isfile(p)]

    # 知识点：GSM8K 有 main 和 socratic 两个"配置"，镜像里常常两个都下下来。
    # 只留 main，否则条数会翻倍（14946 而不是 7473），而且数据重复。
    def prefer_main(paths):
        main = [p for p in paths if "/main/" in p.replace("\\", "/")]
        return main if main else paths

    parquet = prefer_main([p for p in files if p.endswith(".parquet")])
    if parquet:
        data_files = {}
        for p in parquet:
            s = guess_split(p)
            if s:
                data_files.setdefault(s, []).append(p)
        if not data_files:  # 文件名里没有 split 信息，当 train 用
            data_files["train"] = parquet
        print("[load] parquet:", {k: len(v) for k, v in data_files.items()})
        return load_dataset("parquet", data_files=data_files)

    jsonf = prefer_main([p for p in files if p.endswith((".json", ".jsonl"))])
    if jsonf:
        data_files = {}
        for p in jsonf:
            s = guess_split(p)
            if s:
                data_files.setdefault(s, []).append(p)
        if not data_files:
            raise SystemExit(f"找不到带 train/test 的 json 文件：{jsonf}")
        print("[load] json:", {k: len(v) for k, v in data_files.items()})
        return load_dataset("json", data_files=data_files)

    raise SystemExit(f"{local_dir} 里没有 parquet/json 文件，看看是不是没下完")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gsm8k", help="输出目录")
    ap.add_argument("--local_dir", default=DEFAULT_LOCAL, help="数据集落盘目录")
    ap.add_argument("--skip_download", action="store_true", help="本地已经有数据时不下载")
    ap.add_argument("--keep_calc", action="store_true",
                    help="保留 <<a+b=c>> 计算注释（默认清洗掉，实测保留会让 SFT 掉 25 分）")
    args = ap.parse_args()

    if not args.skip_download and not os.path.isdir(args.local_dir):
        download_from_modelscope(args.local_dir)

    ds = load_local(args.local_dir)
    print("[splits]", list(ds.keys()))

    split_map = []
    for src in ds.keys():
        split_map.append((src, "test" if guess_split(src) == "test" else "train"))

    # 至少要有 test，否则评测无从谈起
    if not any(t == "test" for _, t in split_map):
        raise SystemExit(
            "没找到 test split。请检查下载是否完整；"
            "如果镜像里只有一个文件，手动指定 --local_dir 到含 test 文件的目录。"
        )

    for src, out_split in split_map:
        rows, bad, dup = [], 0, 0
        seen = set()
        for ex in ds[src]:
            q = str(ex["question"]).strip()
            if q in seen:          # 兜底去重：镜像里偶尔 main/socratic 会混在一起
                dup += 1
                continue
            seen.add(q)
            ans = parse_answer(ex.get("answer", ""))
            if ans is None:
                bad += 1
                continue
            rows.append({
                "question": q,
                "answer": ans,
                # 只清洗 train 的 target；test 的 solution 不参与训练，无所谓
                "solution": clean_solution(ex["answer"], strip_calc=not args.keep_calc),
                "prompt": None,
            })
        path = os.path.join(args.out, f"{out_split}.jsonl")
        write_jsonl(path, rows)
        print(f"[save] {path}  rows={len(rows)}  dropped={bad}  dup={dup}")

    stats = {}
    for split in ["train", "test"]:
        p = os.path.join(args.out, f"{split}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        qlen = [len(r["question"]) for r in rows]
        stats[split] = {
            "num": len(rows),
            "q_char_mean": round(sum(qlen) / len(qlen), 1),
            "q_char_max": max(qlen),
        }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("[stats]", json.dumps(stats, ensure_ascii=False))

    # 自检：test 的数量应该接近 1319，train 接近 7473
    train_n = stats.get("train", {}).get("num", 0)
    test_n = stats.get("test", {}).get("num", 0)
    print(f"[check] train={train_n} test={test_n}  "
          f"（标准值 7473 / 1319，差太多说明下载不完整）")


if __name__ == "__main__":
    main()
