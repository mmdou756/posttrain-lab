"""清洗 RFT 训练目标：去掉模型复读 prompt 模板留下的残影。

问题（11/02 实测）：
  模型采样时会把 prompt 里的格式指令复读出来，产生这些尾巴：
    "#### #### 26"
    "#### <number> 180"
    "#### Final Answer in the required format:\n#### #### 120"
  这些被当成"正确答案"训进去后，模型学会了在奇怪的地方输出 ####，
  导致评测时 format_rate 掉到 0.57、抽取容易抓错。

清洗规则（三条）：
  1. 找到第一个「#### 数字」，把后面所有内容全部删掉（截断）
  2. 删掉字面的占位符 "<number>"
  3. 如果整段都没有「#### 数字」，用抽取器抓最后一个数字补上

用法：
  python data/clean_rft.py --in data/rft_sft_500.jsonl --out data/rft_sft_clean.jsonl
"""
import argparse
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import read_jsonl, write_jsonl
from eval.extract import extract_number

# 「####」后面必须跟数字才算答案标记（避免匹配到 "#### <number>" 里的 ####）
ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
HASHTAG_GARBAGE = re.compile(r"####\s*<number>")


def clean_solution(text: str):
    """返回 (清洗后文本, 是否改动过)。"""
    if not text:
        return text, False
    orig = text

    # 1) 去掉字面占位符
    text = HASHTAG_GARBAGE.sub("", text)

    # 2) 截断到第一个「#### 数字」为止
    m = ANSWER_RE.search(text)
    if m:
        num = m.group(1).replace(",", "")
        head = text[: m.start()].rstrip()
        text = f"{head}\n#### {num}\n"
    else:
        # 3) 兜底：没有合规的 #### 数字，就用抽取器抓最后一个数字补上
        num = extract_number(text)
        if num is not None:
            head = text.rstrip()
            text = f"{head}\n#### {num}\n"

    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text, (text != orig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.inp)
    changed, before, after = 0, [], []
    for r in rows:
        sol = r.get("solution", "")
        new, did = clean_solution(sol)
        if did:
            changed += 1
        before.append(len(sol.split()))
        after.append(len(new.split()))
        r["solution"] = new

    write_jsonl(args.outp, rows)

    print("=" * 56)
    print(f"样本数        : {len(rows)}")
    print(f"被改动的      : {changed}  ({changed/len(rows):.1%})")
    print(f"清洗前平均词数: {statistics.mean(before):.1f}")
    print(f"清洗后平均词数: {statistics.mean(after):.1f}")
    # 检查是否还有模板残影
    bad = sum(1 for r in rows if "<number>" in r["solution"]
              or "required format" in r["solution"]
              or r["solution"].count("####") != 1)
    print(f"仍有残影的    : {bad}   ← 应该是 0")
    print("=" * 56)

    print("\n--- 清洗前后对比（前 2 条）---")
    orig = read_jsonl(args.inp)
    for i in range(min(2, len(rows))):
        print(f"[原] {orig[i]['solution'][-90:]!r}")
        print(f"[新] {rows[i]['solution'][-90:]!r}")
        print()


if __name__ == "__main__":
    main()
