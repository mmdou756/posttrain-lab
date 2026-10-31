"""评测主程序：跑一个模型（可带 LoRA adapter）在 GSM8K 上的分数。

用法：
  # 先自测抽取器（10 秒钟，能救你一天）
  python eval/extract.py --selftest

  # 测 base 模型（找 baseline）
  python eval/run_eval.py --model /root/rivermind-data/models/Qwen3-1.7B \
      --data data/gsm8k/test.jsonl --limit 100 --out results/base_100.jsonl

  # 测 LoRA 微调后的模型
  python eval/run_eval.py --model /root/rivermind-data/models/Qwen3-1.7B \
      --adapter /root/rivermind-data/runs/sft-smoke/final \
      --data data/gsm8k/test.jsonl --limit 200 --out results/sft_200.jsonl

  # 多次采样（报告 ± 用）
  python eval/run_eval.py ... --n_samples 4 --temperature 0.7

输出：
  - 结果 jsonl（每题的输出、抽取结果、对错）
  - 控制台打印 pass@1 / 平均长度 / 截断率 / 格式率 / 均值±标准差
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import build_prompt, load_causal_lm, read_jsonl, set_seed, write_jsonl
from eval.extract import extract_number, has_format, is_equal


def load_model(model_path, adapter, trust_remote_code=False):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 知识点：批量生成时必须左边 padding。
    # 右边 padding 会让模型在 pad 之后继续生成，位置编码错位，分数莫名其妙偏低。
    tok.padding_side = "left"

    model = load_causal_lm(
        model_path, device_map="auto", trust_remote_code=trust_remote_code,
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        # 知识点：评测不需要梯度，把 LoRA 合并进基座可以少一次矩阵加法，推理快 5~15%。
        # 合并后模型结构变了，就取不到 adapter 了——所以只在评测里 merge。
        model = model.merge_and_unload()
    model.eval()
    return model, tok


@torch.inference_mode()
def generate_batch(model, tok, prompts, max_new_tokens, temperature, top_p):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    do_sample = temperature > 0
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        # 知识点：评测时不要用 repetition_penalty 之类的技巧性参数。
        # 评测口径要"干净"，否则你无法解释分数变化来自模型还是来自解码技巧。
    )
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(new_tokens, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA 目录，不给就测 base")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--n_samples", type=int, default=1, help=">1 时开启采样，用于报方差")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    rows = read_jsonl(args.data)
    if args.limit and args.limit < len(rows):
        # 知识点：固定取前 N 条（不要随机采），这样两次实验是可比的
        rows = rows[: args.limit]

    model, tok = load_model(args.model, args.adapter, args.trust_remote_code)
    prompts = [build_prompt(tok, r["question"]) for r in rows]

    t0 = time.time()
    all_preds, all_raw, accs, lens, truncs, fmts = [], [], [], [], [], []
    for s in range(args.n_samples):
        preds, raws, acc = [], [], []
        for i in range(0, len(prompts), args.bs):
            batch = prompts[i:i + args.bs]
            texts = generate_batch(model, tok, batch, args.max_new_tokens,
                                   args.temperature, args.top_p)
            gold = [r["answer"] for r in rows[i:i + args.bs]]
            for text, g in zip(texts, gold):
                p = extract_number(text)
                preds.append(p)
                raws.append(text)
                acc.append(is_equal(p, g))
                lens.append(len(text.split()))
                fmts.append(has_format(text))
                # 截断率：输出长度顶到上限 = 没生成完，答案多半是错的
                truncs.append(int(len(tok(text, add_special_tokens=False)["input_ids"])
                                  >= args.max_new_tokens - 2))
        accs.append(sum(acc) / len(acc))
        all_preds.append(preds)
        all_raw.append(raws)
        print(f"[sample {s+1}/{args.n_samples}] acc={accs[-1]:.4f}")

    mean_acc = statistics.mean(accs)
    std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0

    # 落盘：只存第一份采样的原文，避免文件过大
    if args.out:
        out_rows = []
        for i, r in enumerate(rows):
            out_rows.append({
                "question": r["question"],
                "gold": r["answer"],
                "pred": all_preds[0][i],
                "correct": is_equal(all_preds[0][i], r["answer"]),
                "output": all_raw[0][i],
            })
        write_jsonl(args.out, out_rows)
        print(f"[save] {args.out}")

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "data": args.data,
        "n": len(rows),
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "pass@1_mean": round(mean_acc, 4),
        "pass@1_std": round(std_acc, 4),
        "avg_len_words": round(sum(lens) / len(lens), 1),
        "trunc_rate": round(sum(truncs) / len(truncs), 4),
        "format_rate": round(sum(fmts) / len(fmts), 4),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
