"""拒绝采样（Rejection Sampling）造偏好数据 —— RFT / DPO 的第一步。

做什么：
  同一道题，用当前模型采样 n 条 → 逐条判对错 → 对的当 chosen、错的当 rejected
  只保留「有对有错」的题（全对/全错都没有梯度信号）

为什么必须用自己的模型采样（这是 10/31 那次负结果的直接教训）：
  如果你拿「别人写的解答」当 chosen、模型自己的输出当 rejected，
  模型学到的不是「哪个推理对」，而是「谁写的 / 哪种风格」——
  昨天的 SFT 已经证明：训练目标的统计特征会被完整模仿，连长度都会对齐。
  同源采样的 8 条长度、格式、措辞分布完全一致，唯一差别就是「推没推对」。

输出格式（TRL 的 DPOTrainer 直接能吃）：
  {"prompt": "...", "chosen": "...", "rejected": "...", "n_correct": 3, "n_total": 8}

用法：
  # 先小规模验证（50 题，约 5 分钟）
  python dpo/make_preferences.py \
    --model /root/rivermind-data/models/Qwen3-1.7B \
    --data data/gsm8k/train.jsonl \
    --out data/pref_50.jsonl --limit 50 --n_samples 8

  # 看长度分布是否失衡（昨天踩过的坑）
  python dpo/make_preferences.py ... --report_only data/pref_50.jsonl
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import build_prompt, load_causal_lm, log, read_jsonl, set_seed, write_jsonl
from eval.extract import extract_number, is_equal


def load_model_and_tok(model_path, adapter, trust_remote_code=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 采样必须左边 padding：右上 padding 会让模型在 pad 之后接着生成，位置错乱
    tok.padding_side = "left"
    model = load_causal_lm(model_path, device_map="auto", trust_remote_code=trust_remote_code)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


@torch.inference_mode()
def sample_completions(model, tok, prompts, n_samples, temperature, top_p, max_new_tokens):
    """对一批 prompt 各采样 n_samples 条，返回 [batch][n] 的文本。

    知识点：num_return_sequences=n 是一次前向展开 n 份，比循环 n 次快得多。
    代价是显存 = n × KV cache，n=8 时要留够。
    """
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n_samples,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
    # 每 n_samples 条属于同一个 prompt
    return [texts[i * n_samples:(i + 1) * n_samples] for i in range(len(prompts))]


def report(rows):
    """统计：可用率 + chosen/rejected 的长度分布（防 length bias）。"""
    cl = [len(r["chosen"].split()) for r in rows]
    rl = [len(r["rejected"].split()) for r in rows]
    nc = [r["n_correct"] for r in rows]
    print()
    print("=" * 58)
    print(f"可用偏好对        : {len(rows)}")
    print(f"每题平均正确条数  : {statistics.mean(nc):.2f} / {rows[0]['n_total']}")
    print(f"chosen  平均词数  : {statistics.mean(cl):.1f}  (中位 {statistics.median(cl):.0f})")
    print(f"rejected 平均词数 : {statistics.mean(rl):.1f}  (中位 {statistics.median(rl):.0f})")
    gap = statistics.mean(cl) - statistics.mean(rl)
    print(f"长度差（chosen−rejected）: {gap:+.1f} 词")
    if abs(gap) > 0.15 * statistics.mean(cl):
        print("⚠️  长度差超过 15%：DPO 可能学到「长的更好」而不是「对的更好」")
        print("    缓解：长度归一化、或按长度分桶采样")
    print("=" * 58)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="用哪个 checkpoint 采样（默认纯 base）")
    ap.add_argument("--data", required=True, help="题目池 jsonl（需含 question/answer）")
    ap.add_argument("--out", default="data/pref.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0, help="从第几条题开始（分批跑用）")
    ap.add_argument("--n_samples", type=int, default=8, help="每题采样条数（GRPO 里叫 G）")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--bs", type=int, default=2, help="每个 batch 的题数（× n_samples 是实际并发）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--sft_out", default=None,
                    help="额外输出一份 SFT 格式 jsonl（用 chosen 当 solution），可直接喂 train_sft.py")
    ap.add_argument("--report_only", default=None, help="只统计已有结果，不跑模型")
    args = ap.parse_args()

    if args.report_only:
        report(read_jsonl(args.report_only))
        return

    set_seed(args.seed)
    rows_in = read_jsonl(args.data)[args.offset:args.offset + args.limit]
    log(f"题目数: {len(rows_in)}  每题采样 {args.n_samples} 条")

    model, tok = load_model_and_tok(args.model, args.adapter, args.trust_remote_code)

    prompts = [build_prompt(tok, r["question"]) for r in rows_in]
    kept, all_correct, all_wrong, t0 = [], 0, 0, time.time()

    for i in range(0, len(prompts), args.bs):
        batch = prompts[i:i + args.bs]
        golds = [r["answer"] for r in rows_in[i:i + args.bs]]
        samples = sample_completions(model, tok, batch, args.n_samples,
                                     args.temperature, args.top_p, args.max_new_tokens)

        for q_idx, (texts, gold) in enumerate(zip(samples, golds)):
            correct = [t for t in texts if is_equal(extract_number(t), gold)]
            wrong = [t for t in texts if not is_equal(extract_number(t), gold)]

            # 知识点：全对（无梯度）和全错（学不到正确解）都要丢
            if len(correct) == 0:
                all_wrong += 1
                continue
            if len(wrong) == 0:
                all_correct += 1
                continue

            kept.append({
                "prompt": rows_in[i + q_idx]["question"],
                "chosen": correct[0],
                "rejected": wrong[0],
                "n_correct": len(correct),
                "n_total": args.n_samples,
            })

        done = min(i + args.bs, len(prompts))
        el = time.time() - t0
        log(f"  {done}/{len(prompts)} 题  可用 {len(kept)}  "
            f"（全对 {all_correct} / 全错 {all_wrong}）  {el:.0f}s")

    if not kept:
        raise SystemExit("一个可用的偏好对都没生成：题目太简单（全对）或太难（全错）？")

    write_jsonl(args.out, kept)
    log(f"保存: {args.out}")
    report(kept)

    # 知识点：同一批采样可以喂两条下游路线——
    # ① chosen/rejected → DPO 的偏好对
    # ② 只用 chosen + 标准答案 → SFT 的训练数据（就是 RFT，拒绝采样微调）
    # 因为 chosen 是模型自己写的，风格和它本来的输出一致，
    # 不会再出现 10/31 那种「被别人的紧凑解答覆盖掉推理能力」的问题。
    if args.sft_out:
        gold_map = {r["question"]: r["answer"] for r in rows_in}
        sft_rows = [{
            "question": r["prompt"],
            "answer": gold_map.get(r["prompt"], ""),
            "solution": r["chosen"],
            "prompt": None,
        } for r in kept]
        write_jsonl(args.sft_out, sft_rows)
        log(f"SFT 数据已保存: {args.sft_out}  ({len(sft_rows)} 条)")
        cl = [len(r["solution"].split()) for r in sft_rows]
        print(f"RFT 训练目标平均词数: {statistics.mean(cl):.1f}  "
              f"（对比 GSM8K 官方解答约 51 词、base 模型自称 139 词）")

    # 知识点：可用率 = 1 − 全对率 − 全错率，它直接决定采样成本。
    # 如果可用率很低（比如 <30%），说明题目难度和模型能力不匹配——
    # 这正是 10/31 发现的「GSM8K 对 Qwen3-1.7B 已饱和（83%）」的后果：
    # 8 条里全对的概率约 0.83^8 ≈ 22%，这些样本全被丢掉。


if __name__ == "__main__":
    main()
