#!/usr/bin/env python3
"""
对比 baseline 和 balance_split 两份性能数据，输出速度提升统计。
用法: python compare_perf.py baseline.txt balance_split.txt [-o report.txt]
"""

import sys
import argparse


def parse_perf_file(path):
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if (
                not line
                or line.startswith("[")
                or line.startswith("=")
                or line.startswith("Format")
            ):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 10:
                continue
            try:
                category = parts[0]
                sample = parts[1]
                perf = float(parts[9])
                key = (category, sample)
                results[key] = {
                    "category": category,
                    "sample": sample,
                    "M": int(parts[2]),
                    "K": int(parts[3]),
                    "N": int(parts[4]),
                    "nnz": int(parts[5]),
                    "window_num": int(parts[6]),
                    "block_num": int(parts[7]),
                    "fill_rate": float(parts[8]),
                    "perf": perf,
                }
            except (ValueError, IndexError):
                continue
    return results


def main():
    parser = argparse.ArgumentParser(description="对比两份性能数据")
    parser.add_argument("baseline", help="baseline 性能文件")
    parser.add_argument("optimized", help="优化后性能文件")
    parser.add_argument("-o", "--output", default=None, help="输出文件 (默认终端)")
    parser.add_argument(
        "--sort",
        default="speedup",
        choices=["speedup", "abs", "sample", "blocks"],
        help="排序方式: speedup/abs/sample/blocks",
    )
    args = parser.parse_args()

    base = parse_perf_file(args.baseline)
    opt = parse_perf_file(args.optimized)

    common_keys = sorted(set(base.keys()) & set(opt.keys()))
    only_base = set(base.keys()) - set(opt.keys())
    only_opt = set(opt.keys()) - set(base.keys())

    rows = []
    for key in common_keys:
        b = base[key]
        o = opt[key]
        speedup = b["perf"] / o["perf"] if o["perf"] > 0 else float("inf")
        diff = b["perf"] - o["perf"]
        rows.append(
            {
                "category": b["category"],
                "sample": b["sample"],
                "M": b["M"],
                "blocks": b["block_num"],
                "base_us": b["perf"],
                "opt_us": o["perf"],
                "diff_us": diff,
                "speedup": speedup,
            }
        )

    if args.sort == "speedup":
        rows.sort(key=lambda r: r["speedup"], reverse=True)
    elif args.sort == "abs":
        rows.sort(key=lambda r: r["diff_us"], reverse=True)
    elif args.sort == "sample":
        rows.sort(key=lambda r: (r["category"], r["sample"]))
    elif args.sort == "blocks":
        rows.sort(key=lambda r: r["blocks"], reverse=True)

    out = open(args.output, "w") if args.output else sys.stdout

    out.write(
        f"{'Category':<20} {'Sample':<25} {'M':>6} {'Blocks':>7} {'Base(us)':>10} {'Opt(us)':>10} {'Diff(us)':>10} {'Speedup':>8}\n"
    )
    out.write("-" * 100 + "\n")

    total_base = 0
    total_opt = 0
    improved = 0
    regressed = 0
    unchanged = 0
    speedups = []

    for r in rows:
        total_base += r["base_us"]
        total_opt += r["opt_us"]
        speedups.append(r["speedup"])
        if r["speedup"] > 1.005:
            improved += 1
        elif r["speedup"] < 0.995:
            regressed += 1
        else:
            unchanged += 1

        out.write(
            f"{r['category']:<20} {r['sample']:<25} {r['M']:>6} {r['blocks']:>7} "
            f"{r['base_us']:>10.2f} {r['opt_us']:>10.2f} {r['diff_us']:>10.2f} {r['speedup']:>7.3f}x\n"
        )

    out.write("-" * 100 + "\n")
    out.write(f"\n总样本数: {len(rows)}\n")
    out.write(f"提速: {improved}, 持平: {unchanged}, 变慢: {regressed}\n")
    out.write(
        f"总耗时: baseline={total_base:.2f}us, optimized={total_opt:.2f}us, "
        f"总加速比={total_base / total_opt:.3f}x\n"
    )

    if speedups:
        import statistics

        out.write(
            f"单样本加速比: min={min(speedups):.3f}x, max={max(speedups):.3f}x, "
            f"mean={statistics.mean(speedups):.3f}x, median={statistics.median(speedups):.3f}x\n"
        )

    if only_base:
        out.write(f"\n仅在 baseline 中出现: {len(only_base)} 个样本\n")
    if only_opt:
        out.write(f"仅在 optimized 中出现: {len(only_opt)} 个样本\n")

    if args.output:
        out.close()
        print(f"报告已写入 {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
