"""从 trainer_state.json 里把 loss 曲线画出来（没有 matplotlib 就打印 ASCII 图）。

用法：
  python scripts/plot_log.py /root/rivermind-data/runs/sft100_bs8/checkpoint-100/trainer_state.json
  python scripts/plot_log.py <run目录>          # 自动找最新的 checkpoint

为什么单独写一个：
报告和简历里需要的不是"loss 从 2.2 降到 0.42"这句话，而是**一张图**。
图能一眼看出：有没有抖动、什么时候收敛、有没有异常 spike。
"""
import json
import os
import sys


def find_state(path):
    """传入 run 目录或 trainer_state.json 都能用。"""
    if os.path.isfile(path):
        return path
    cands = []
    for root, _dirs, files in os.walk(path):
        if "trainer_state.json" in files:
            cands.append(os.path.join(root, "trainer_state.json"))
    if not cands:
        raise SystemExit(f"{path} 下没找到 trainer_state.json")
    # 取步数最大的那个 checkpoint
    def steps(p):
        d = json.load(open(p, encoding="utf-8"))
        return d.get("global_step", 0)
    return max(cands, key=steps)


def ascii_plot(xs, ys, width=60, height=12):
    """终端里的 ASCII 折线图，不依赖任何库。"""
    if not ys:
        return
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1.0
    grid = [[" "] * width for _ in range(height)]
    for i, y in enumerate(ys):
        xi = int(i / max(1, len(ys) - 1) * (width - 1))
        yi = int((y - lo) / span * (height - 1))
        grid[height - 1 - yi][xi] = "*"
    print(f"  {hi:6.3f} ┐")
    for row in grid:
        print("         │" + "".join(row))
    print(f"  {lo:6.3f} ┴" + "─" * width)
    print(f"         step {xs[0]} {' ' * (width - 20)} step {xs[-1]}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    p = find_state(sys.argv[1])
    d = json.load(open(p, encoding="utf-8"))
    hist = [r for r in d.get("log_history", []) if "loss" in r]
    if not hist:
        raise SystemExit("log_history 里没有 loss 记录")

    steps = [r.get("step", i) for i, r in enumerate(hist)]
    loss = [r["loss"] for r in hist]
    gnorm = [r.get("grad_norm") for r in hist]
    lr = [r.get("learning_rate") for r in hist]

    print(f"file      : {p}")
    print(f"global_step: {d.get('global_step')}")
    print(f"records   : {len(hist)}")
    print(f"loss      : {loss[0]:.4f} -> {loss[-1]:.4f}   (min {min(loss):.4f})")
    if gnorm and gnorm[0] is not None:
        print(f"grad_norm : {gnorm[0]:.3f} -> {gnorm[-1]:.3f}")
    if lr:
        print(f"lr        : {max(lr):.3e} (peak) -> {lr[-1]:.3e}")

    print("\nloss 曲线：")
    ascii_plot(steps, loss)

    # 有 matplotlib 就顺便存一张 PNG（报告用）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        axes[0].plot(steps, loss, marker="o", ms=3)
        axes[0].set_xlabel("step"); axes[0].set_ylabel("loss"); axes[0].set_title("SFT loss")
        axes[0].grid(alpha=.3)
        if lr:
            axes[1].plot(steps, lr, marker="o", ms=3, color="tab:orange")
            axes[1].set_xlabel("step"); axes[1].set_ylabel("learning rate"); axes[1].set_title("LR schedule")
            axes[1].grid(alpha=.3)
        fig.tight_layout()
        # 存在 run 根目录，而不是 checkpoint 里——否则你清理 checkpoint 时图会一起被删
        run_root = os.path.dirname(p)
        while os.path.basename(run_root).startswith("checkpoint"):
            run_root = os.path.dirname(run_root)
        out = os.path.join(run_root, "loss_curve.png")
        fig.savefig(out, dpi=150)
        print(f"\n[已保存图] {out}")
    except ImportError:
        print("\n（没装 matplotlib，只打印文字版；装：pip install matplotlib）")


if __name__ == "__main__":
    main()
