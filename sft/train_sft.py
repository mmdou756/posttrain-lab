"""SFT 训练脚本（Qwen3 + LoRA），从你 SciCaption 的 fintune_llm.py 改造而来。

改造的"三处"之外的四个关键修正（都在本文件里有对应注释）：
1. 结尾 token：原来追加的是 pad_token_id，应该是 eos_token（否则模型学不会停）。
2. 超长样本：原来从尾部截断，会把 response 截光、labels 全变 -100（这条样本白训）。
3. chat 模板：原来手拼 <|im_start|>，现在用 tokenizer.apply_chat_template（换模型不用改代码）。
4. 随机性：原来没设 seed，两次跑结果不一样，做不了消融。

用法（先跑 100 步冒烟，别一上来跑全量）：
  python sft/train_sft.py \
    --model /root/rivermind-data/models/Qwen3-1.7B \
    --data data/gsm8k/train.jsonl \
    --out /root/rivermind-data/runs/sft-smoke \
    --max_steps 100 --bs 2 --accum 8 --max_len 1024

做"全 token loss vs 只算 assistant"的对照（W1 的 Day 8）：
  python sft/train_sft.py ... --no_mask --out /root/rivermind-data/runs/sft-fulltok
"""
import argparse
import json
import math
import os
import re
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import build_prompt, dump_config, load_causal_lm, log, read_jsonl, set_seed


def build_tokenizer(path, trust_remote_code=False):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    # 知识点：Qwen 系 tokenizer 自带 pad_token；有些模型没有，必须补一个，
    # 否则 batch > 1 时 DataCollator 会报错。
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 知识点：decoder-only 训练时 padding 在右侧，和生成时一致，避免 position 错位。
    tok.padding_side = "right"
    return tok


def build_model(path, args, logfile):
    # 知识点：bf16 是训练默认精度。为什么不 fp16？fp16 的指数位只有 5 位，
    # 梯度小到 1e-8 就变 0（underflow），要配 GradScaler；bf16 指数位和 fp32 一样，
    # 范围够大，不需要 scaler，训练更稳。代价是尾数精度低（有效数字约 3 位），对训练无碍。
    model = load_causal_lm(
        path,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False  # 知识点：训练时必须关 KV cache，否则和 gradient checkpointing 冲突

    if not args.no_lora:
        # 知识点（面试必问）：LoRA 冻结原权重，只训练低秩分解 ΔW = B @ A（r 远小于 d）。
        # - r：秩。越大表达力越强、参数越多。7B 上常用 8~64；1.7B 小模型用 16 足够。
        # - alpha：缩放。实际缩放系数是 alpha / r，所以 alpha 一般取 2r。
        # - dropout：防过拟合，数据少时调到 0.05~0.1。
        # - target_modules：注意力四个投影 + MLP 三个投影，覆盖越全效果越好、参数越多。
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    if not args.no_grad_ckpt:
        # 知识点：gradient checkpointing 用"重算"换显存——不存中间激活，反向时重新前向。
        # 显存大约降 60~70%，速度慢 20~30%。
        # PEFT + checkpointing 必须开 input_require_grads，否则梯度传不到 LoRA 层（常见坑）。
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        log("gradient_checkpointing: ON", logfile)

    return model


def encode_one(tok, ex, mode, max_len):
    """把一条样本变成 input_ids / labels。

    这是整个 SFT 最核心的 40 行，面试问到"你怎么训的"就从这里讲。
    """
    prompt = build_prompt(tok, ex["question"])
    if mode == "cot":
        # 训练完整推理链：模型学会"一步步算"，这是后面 RLVR 能work的前提
        response = ex["solution"].strip()
    else:
        # 只训练最终答案：模型学会格式，但学不会推理（对照实验用）
        response = f"#### {ex['answer']}"

    # 知识点：追加 eos。原来是 pad_token_id ——那是错的。
    # 不教模型"什么时候停"，推理时它会一直往外吐，评测时你的答案抽取就会抓错数字。
    response = response + tok.eos_token

    # 知识点：prompt 和 response 分开 tokenize，才能精确知道 assistant 从第几个 token 开始。
    # 加了 add_special_tokens=False 防止 tokenizer 偷偷在 response 头部加 BOS。
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(response, add_special_tokens=False)["input_ids"]

    input_ids = p_ids + r_ids
    if len(input_ids) > max_len:
        return None  # 超长直接丢，见下面的说明

    # 知识点：labels 里 -100 的位置不参与 loss（PyTorch 的 CrossEntropyLoss 约定）。
    # 只对 assistant 段算 loss，模型才是在学"怎么答"，而不是在学"怎么问"。
    labels = [-100] * len(p_ids) + r_ids
    if MODE_NO_MASK[0]:
        # 对照实验：全 token 都算 loss。你会看到模型开始模仿提问风格、输出变短、格式崩
        labels = list(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


MODE_NO_MASK = [False]  # 用可变容器传参，避免全局变量在 map 里出问题


def build_training_args(kwargs, logfile=None):
    """构造 TrainingArguments，自动跳过当前 transformers 不认识的参数。

    知识点：transformers 大版本会改名/删参数（4.x 的 warmup_ratio 在 5.x 没了）。
    与其一个个试错，不如捕获 TypeError 里的参数名、摘掉它再重试，
    并且把"跳过了什么"打进日志——否则你会在不知情的情况下失去某个训练技巧。
    """
    dropped = []
    while True:
        try:
            targs = TrainingArguments(**kwargs)
            break
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '([^']+)'", str(e))
            if not m or m.group(1) not in kwargs:
                raise
            k = m.group(1)
            dropped.append((k, kwargs.pop(k)))
    if dropped:
        log(f"⚠️ 当前 transformers 不支持的参数已跳过: {dropped}", logfile)
    return targs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="train.jsonl 路径")
    ap.add_argument("--out", required=True, help="输出目录（adapter 存这里）")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=2, help="per_device_train_batch_size")
    ap.add_argument("--accum", type=int, default=8, help="梯度累积步数")
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA 常用 1e-4 ~ 2e-4")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1, help=">0 时忽略 epochs（冒烟用）")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--mode", choices=["cot", "answer"], default="cot")
    ap.add_argument("--no_mask", action="store_true", help="对照实验：全 token 算 loss")
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="关掉梯度检查点（显存够时可以提速）")
    ap.add_argument("--resume", default=None, help="从某个 checkpoint 续训（断点续训必用）")
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只取前 N 条数据（调试用）")
    args = ap.parse_args()

    MODE_NO_MASK[0] = args.no_mask
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    logfile = os.path.join(args.out, "train.log")
    log(f"args = {vars(args)}", logfile)

    tok = build_tokenizer(args.model, args.trust_remote_code)
    model = build_model(args.model, args, logfile)

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    log(f"raw samples: {len(rows)}", logfile)

    # 知识点：读数据第一件事是"校验字段"，不是直接开训。
    # 字段不对时让它第 1 秒就报错并打印实际字段，别等 map 到第 300 条才炸。
    if rows:
        need = {"question", "answer"} | ({"solution"} if args.mode == "cot" else set())
        missing = need - set(rows[0].keys())
        if missing:
            raise SystemExit(
                f"数据缺字段 {sorted(missing)}，实际字段是 {list(rows[0].keys())}。\n"
                f"用仓库里的脚本重新生成：python data/prepare_gsm8k.py --out data/gsm8k"
            )

    t0 = time.time()
    encoded, dropped = [], 0
    for ex in rows:
        e = encode_one(tok, ex, args.mode, args.max_len)
        if e is None:
            dropped += 1
            continue
        encoded.append(e)
    log(f"encoded: {len(encoded)}  dropped(too_long): {dropped}  "
        f"({time.time() - t0:.1f}s)", logfile)

    # 知识点：丢掉超长样本而不是从尾部截断。
    # 从尾部截断会把 response 截没，labels 全 -100 -> 这条样本 loss 恒为 0，
    # 更糟的是整个 batch 全 -100 时 CrossEntropy 会返回 nan。
    assert encoded, "所有样本都被过滤了：max_len 太小或数据格式不对"
    if dropped / max(len(rows), 1) > 0.05:
        log(f"⚠️ 超过 5% 的样本被丢弃，考虑调大 max_len", logfile)

    lens = [len(e["input_ids"]) for e in encoded]
    log(f"token len: mean={sum(lens)/len(lens):.0f} max={max(lens)} "
        f"p95={sorted(lens)[int(len(lens)*0.95)]}", logfile)

    ds = Dataset.from_list(encoded)

    # 知识点：warmup 用 steps 而不是 ratio。
    # transformers 5.x 删掉了 `warmup_ratio`（4.x 有），只留 `warmup_steps`。
    # 自己算出总步数再乘 3%，两个版本都能跑——这比判断版本号稳。
    steps_per_epoch = max(1, math.ceil(len(ds) / (args.bs * args.accum)))
    total_steps = args.max_steps if (args.max_steps and args.max_steps > 0) \
        else max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = max(1, int(0.03 * total_steps))
    log(f"total_steps≈{total_steps} warmup_steps={warmup_steps}", logfile)

    ta_kwargs = dict(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,          # 只留最近两份，别把数据盘撑爆
        bf16=True,
        gradient_checkpointing=not args.no_grad_ckpt,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,           # 梯度裁剪：GRPO 阶段更重要
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,  # 保持 False：我们自己控制的 collator 只输出三列
        report_to="none",
        seed=args.seed,
        logging_first_step=True,
    )
    targs = build_training_args(ta_kwargs, logfile)

    # 知识点：DataCollatorForSeq2Seq 做两件事——把 batch 内序列 pad 到等长，
    # 并且把 labels 里的 pad 位置改成 -100（label_pad_token_id）。
    # 如果这一步漏了，模型会去学"预测 padding"，loss 会虚高但学不到东西。
    collator = DataCollatorForSeq2Seq(
        tokenizer=tok, padding=True, label_pad_token_id=-100, return_tensors="pt"
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)

    eff_bs = args.bs * args.accum
    log(f"effective batch size = {eff_bs}  samples  "
        f"(~{eff_bs * sum(lens)/len(lens) / 1e3:.1f}k tokens/step)", logfile)

    trainer.train(resume_from_checkpoint=args.resume)

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    log(f"saved: {final}", logfile)

    # 知识点：把配置落盘。报告里每个数字都要能追到这里。
    dump_config(os.path.join(args.out, "run_config.json"), **vars(args))
    log("done", logfile)


if __name__ == "__main__":
    main()
