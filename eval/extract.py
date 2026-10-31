"""答案抽取：从模型输出里找出最终数字。

为什么单独一个文件：
评测的分数 = 抽取函数 × 模型。**抽取写错，你后面所有实验结论都错。**
所以这个文件必须有单元测试（见文件末尾的 `python eval/extract.py --selftest`）。

知识点（面试会问）：GSM8K 的评测为什么"看起来简单但很容易做错"？
1. 数字有逗号（1,000）、美元（$50）、百分号（50%），不清洗就比对不上；
2. 小数结尾有 0（3.50 vs 3.5），按字符串比会判错；
3. 模型可能输出多个数字（中间步骤），必须按优先级取"最后那个"；
4. 模型可能没写 #### 就停了，要有兜底规则。
"""
import re
import sys

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def clean_text(text: str) -> str:
    """去掉干扰符号。注意：先删 $ 和 %，再抽数字，否则 "50%" 会变成 "50,%"。"""
    return text.replace("$", "").replace("%", "").replace("，", ",")


def normalize(num: str) -> str:
    """把 '1,000' -> '1000'，'3.50' -> '3.5'，'-0' -> '0'。"""
    num = num.replace(",", "").strip()
    try:
        f = float(num)
        if f == int(f):
            return str(int(f))
        return repr(round(f, 6)).rstrip("0").rstrip(".")
    except ValueError:
        return num


def _last_number(s: str):
    ms = NUM_RE.findall(s)
    return normalize(ms[-1]) if ms else None


def extract_number(text: str):
    """按优先级抽取答案，返回规范化后的字符串或 None。

    优先级设计的原因：
    1. `####` 是我们在 prompt 里明确要求的格式，最可信，所以先找它，并且取**最后一次**出现
       （模型有时会在推理中间提到 #### 的格式说明）；
    2. `\\boxed{}` 是数学模型的常见习惯；
    3. "answer is / 答案是 / therefore" 是自然语言兜底；
    4. 最后才退回"全文最后一个数字"——这条最容易抽错，但没它分数会低得离谱。
    """
    if not text:
        return None
    t = clean_text(text)

    if "####" in t:
        tail = t.split("####")[-1]
        n = _last_number(tail)
        if n is not None:
            return n

    m = re.search(r"\\boxed\{([^}]*)\}", t)
    if m:
        n = _last_number(m.group(1))
        if n is not None:
            return n

    m = re.search(r"(?:answer is|answer:|答案是|答案：|therefore,?)\s*([^\n]{0,40})", t, re.I)
    if m:
        n = _last_number(m.group(1))
        if n is not None:
            return n

    return _last_number(t)


def is_equal(pred, ref) -> bool:
    """比较两个答案。两边都能转数字就按数值比，否则按字符串比。"""
    if pred is None or ref is None:
        return False
    try:
        return abs(float(pred) - float(ref)) < 1e-6
    except ValueError:
        return pred.strip().lower() == ref.strip().lower()


def has_format(text: str) -> bool:
    """统计格式遵循率：有没有按要求写 ####。这是"格式奖励"的原始信号。"""
    return "####" in (text or "")


def _selftest():
    cases = [
        ("#### 72", "72"),
        ("blah blah\n#### 1,000", "1000"),
        ("The answer is $50.", "50"),
        ("so \\boxed{3.50}", "3.5"),
        ("答案是：-12", "-12"),
        ("we get 6 apples and 4 pears, total 10", "10"),
        ("no digits here", None),
        ("", None),
    ]
    ok = True
    for text, want in cases:
        got = extract_number(text)
        flag = "OK " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"[{flag}] {text[:40]!r:45} -> {got!r} (want {want!r})")
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(extract_number(sys.stdin.read()))
