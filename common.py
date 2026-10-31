"""公共工具：随机种子、日志、jsonl 读写、配置落盘。

为什么单独抽一个文件：
面试官问"你的实验可复现吗"，标准答案是"我固定了所有随机源，并且把完整配置存盘"。
seed 只设 Python 的没用——torch / numpy / 以及 DataLoader 的 worker 都要设。
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch


def set_seed(seed: int = 42):
    """固定所有随机源。

    知识点：
    - random / numpy / torch 是三个独立的随机源，必须分别设。
    - torch.cuda.manual_seed_all 管多卡，单卡也写上，换机器不踩坑。
    - cudnn.deterministic=True 会让卷积变慢，但 Transformer 主要用 matmul，
      影响可忽略，换来的是"同一配置两次跑结果一致"（这是你后面做消融的前提）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Tee:
    """同时写文件和控制台。

    为什么要它：训练挂了以后，你唯一能看的就是日志文件。
    只 print 到终端，SSH 一断就全没了。
    """

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def log(msg, path=None):
    """带时间戳的日志。"""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump_config(path, **kwargs):
    """把这次实验的完整配置存盘。

    知识点：报告里的每个数字都必须能追到一份配置。
    只凭记忆写报告，三天后你就不知道那个 0.42 是哪个 lr 跑出来的。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kwargs, f, ensure_ascii=False, indent=2)


def count_tokens(tokenizer, texts):
    """统计 token 数（用来估训练量、估成本）。"""
    n = 0
    for t in texts:
        n += len(tokenizer(t, add_special_tokens=False)["input_ids"])
    return n


# ============================================================
# 模型加载：兼容 transformers 4.x / 5.x
# ============================================================

def load_causal_lm(path, dtype=None, **kwargs):
    """加载 causal LM，自动适配 transformers 的版本差异。

    知识点（你现在正好撞上了）：transformers 5.x 把 `torch_dtype` 改名成了 `dtype`。
    写死任何一个，都会在另一个版本上报 TypeError。
    所以先试新名字，不行再退老名字——这比判断版本号更稳（镜像版本号五花八门）。
    """
    import torch
    from transformers import AutoModelForCausalLM

    dtype = dtype or torch.bfloat16
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kwargs)


# ============================================================
# prompt 模板：训练和评测必须共用同一个，这是"训推一致"的第一层
# ============================================================

# 知识点：这条 instruction 决定了模型的输出格式。
# 它必须和 eval/extract.py 的抽取规则严格对应——
# 训练时要求 "#### 答案"，评测时也优先按 "####" 抓，否则分数会莫名偏低。
# 很多"我训了但分数没涨"的案例，根因就是这里不一致。
GSM8K_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "Put the final numeric answer on the last line in the format: #### <number>\n\n"
)


def build_prompt(tokenizer, question, instruction=GSM8K_INSTRUCTION):
    """用模型自带的 chat 模板构造 prompt。

    知识点：为什么不手拼 <|im_start|>？
    1. 换模型（Qwen -> Llama -> GLM）模板全变，手拼的代码全部要改；
    2. Qwen3 的模板还带 thinking 开关，手拼会漏掉；
    3. apply_chat_template 保证训练和推理用的模板完全一致。
    """
    messages = [{"role": "user", "content": instruction + question}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    # Qwen3 的模板支持 enable_thinking 开关；别的模型模板里没有这个变量，
    # 传进去会报错。所以先检查模板字符串里有没有它——这比 try/except 更可靠。
    tmpl = getattr(tokenizer, "chat_template", "") or ""
    if "enable_thinking" in tmpl:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)
