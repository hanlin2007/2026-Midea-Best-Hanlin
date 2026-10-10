"""SZY 系统命令行入口（`python -m szy.cli`）。

职责单一：解析命令行 → 加载配置 → 批量读样本 → 逐样本交给
:class:`szy.pipeline.SzyRunner` → 断点续跑 checkpoint → 汇总并写出
提交物（submission / detail / manifest 三件套 + 人类可读报告）。

"职责单一"的体现：本文件只做"编排"（把各组件按顺序串起来），所有
决策逻辑都在 pipeline.py、所有汇总逻辑在 reporting.py。这样 CLI 本身
薄到几乎不需要测试，逻辑改动都发生在可独立单测的模块里。

四件产物的分工：
  - submission.json  竞赛格式提交物（交卷用）；
  - detail.json      每样本全字段明细（排查/离线重扫阈值用）；
  - manifest.json    资产指纹（记录这次结果由哪份配置+哪份权重跑出，
                     复现实验时对着哈希核对即可）；
  - detail_report.txt 人类可读报告（逐样本"病历"，自动生成）。

用法（仓库根目录，推荐直接走 scripts/run.sh 已配好环境变量）：
    PYTHONPATH=src:model/runtime/vendor \
      python -m szy.cli \
      --data-root dataset/dataset_test/datasetA/datasetA \
      --jsonl dataset/dataset_test/datasetA/datasetA/pos.jsonl \
              dataset/dataset_test/datasetA/datasetA/neg.jsonl \
      --config configs/pipeline_firered.json \
      --output outputs/submission.json

PYTHONPATH 的作用：把 src/ 和 FireRed 的运行时依赖目录加进模块搜索
路径，python 才能找到 szy 包和 kaldi_native_fbank 等 vendor 库
（第三方代码不在 pip 里的通用解法，同 separator.py 的 sys.path 注入）。
"""
from __future__ import annotations

import argparse                                  # 标准库命令行解析
import json                                      # checkpoint / manifest 序列化
from pathlib import Path                         # 路径处理
import time                                      # 计时
from typing import Any                           # 结果行的宽松类型

from tqdm.auto import tqdm                       # 进度条（.auto 自动适配终端/notebook）

from midea_qiji.infer import read_many_jsonl, submission_id
from szy.helpers import sha256_file              # 资产指纹
from szy.pipeline import SzyConfig, SzyRunner, load_config
from szy.reporting import generate_report, summarize, write_outputs


def build_parser() -> argparse.ArgumentParser:
    """组装命令行解析器。所有输出路径都有默认值，日常只需给数据参数。

    单独封装成函数（而非在 main 里直接写）是为了可测试：单测可以
    parse_args([...]) 检查解析行为，不必真正跑 main。
    """
    parser = argparse.ArgumentParser(description="SZY fusion-gate MossFormer2 target-speaker ASR")
    parser.add_argument("--data-root", required=True)       # 音频根目录（jsonl 内相对路径的基准）
    parser.add_argument("--jsonl", nargs="+", required=True)  # 一个或多个样本清单（pos/neg 可混）；nargs="+" = 至少一个、可多个
    parser.add_argument("--config", default="configs/pipeline_firered.json")
    parser.add_argument("--output", default="outputs/submission.json")   # 竞赛格式提交物
    parser.add_argument("--final-output", default="outputs/detail.json") # 全字段明细（含门控特征）
    parser.add_argument("--checkpoint-output", default="outputs/checkpoint.json")  # 断点续跑存档
    parser.add_argument("--manifest-output", default="outputs/manifest.json")     # 资产指纹清单
    parser.add_argument("--model-root")                      # 覆盖配置里的模型根目录（可选，不传则用配置文件值）
    parser.add_argument("--limit", type=int)                 # 分层抽样 N 条冒烟（正负比例与全集一致）；type=int 表示解析成整数，不合法直接报错
    parser.add_argument("--resume", action="store_true")     # 从 checkpoint 续跑；store_true = 开关型参数，传了为 True 不传为 False
    return parser


def _auto_report(detail_path: Path) -> None:
    """管线跑完后自动生成人类可读报告（失败不拖垮主流程）。

    报告是"锦上添花"的产物：detail.json 才是数据本体。生成失败（如
    旧格式 detail 缺字段）只打印警告，不让一次评测白跑——这是"非关键
    路径容错"与主流程"关键路径严格"的配合。
    """
    if not detail_path.is_file():
        return
    try:
        report_path = detail_path.with_name(detail_path.stem + "_report.txt")
        report_path.write_text(generate_report(detail_path), encoding="utf-8")
        print(f"[report] {report_path}", flush=True)
    except Exception as exc:
        print(f"[report] generation failed: {exc}", flush=True)


def main() -> None:
    # parse_args 读 sys.argv 并解析；未知参数会自动报错并打印用法
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.model_root:
        # frozen dataclass 的"复制+覆盖"惯用法：config 不可变，不能直接
        # 赋值 config.model_root = ...；这里取 __dict__（全部字段值的
        # 字典）→ 展开 → 覆盖 model_root 键 → 重新构造一个新实例。
        # 旧实例不受影响，新实例只差这一个字段
        config = SzyConfig(**{**config.__dict__, "model_root": args.model_root})
    # 读样本：jsonl 每行一条 {"唤醒音频": ..., "识别音频": ..., "识别文本": ...}，
    # 相对路径以 data_root 为基准拼成绝对路径（Sample 对象列表）
    samples = read_many_jsonl(args.jsonl, args.data_root)
    if args.limit is not None:
        # 分层抽样冒烟：按全集正负比例抽 N 条（seed 固定，可复现）。
        # 只取前 N 条会 skew——pos.jsonl 排在前面时前 N 条几乎全是正
        # 样本，冒烟既验不出拒识路径也算不出有意义的 RR。
        import random

        random.seed(42)
        pos = [s for s in samples if s.label is not None]
        neg = [s for s in samples if s.label is None]
        n_pos = round(args.limit * len(pos) / len(samples))
        n_neg = args.limit - n_pos
        n_pos = min(n_pos, len(pos))
        n_neg = min(n_neg, len(neg))
        samples = random.sample(pos, n_pos) + random.sample(neg, n_neg)
        random.shuffle(samples)
        print(f"[limit] {args.limit} samples: {n_pos} pos + {n_neg} neg (ratio {len(pos)}/{len(pos)+len(neg)})")

    # ---- 断点续跑：读回 checkpoint，跳过已完成的样本 ----
    # 为什么要 checkpoint：2000 条样本的评测可能跑数小时，任何中断
    # （断电、显存不足、bug）都不该让已算结果作废；checkpoint 让
    # "重跑"变成"续跑"，只补算没完成的
    checkpoint_path = Path(args.checkpoint_output)
    results: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if args.resume:
        # 续跑的前提是存在上次的存档，没有就直接报错（明确失败好过
        # 静默从头跑，掩盖"我以为在续跑"的误解）
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        # get(键, 默认) 兼容旧版本存档缺字段的情况
        results = list(checkpoint.get("results", []))
        elapsed_before = float(checkpoint.get("duration", 0.0))
        # 已完成样本的 id 集合（set 判重 O(1)）；str(item["id"]) 兜底 id
        # 类型差异——checkpoint 里存的是 JSON 还原后的值
        completed = {str(item["id"]) for item in results}
        # 过滤出未完成样本：id 不在 completed 里才重新跑
        samples = [sample for sample in samples if submission_id(sample, config.id_mode) not in completed]

    # runner 的输出目录取提交物所在目录（detail/checkpoint 的默认值也
    # 都在 outputs/ 下）；Path(...).parent 取父目录
    runner = SzyRunner(config, Path(args.output).parent)
    started = time.perf_counter()
    pending_checkpoint_rows = 0
    # tqdm 包住可迭代对象即显示进度条：desc 是前缀文字，unit="utt" 让
    # 速率显示为 "x utt/s"（utt = utterance，语音界的"条"）
    for sample in tqdm(samples, desc="szy", unit="utt"):
        result = runner.process(sample)
        results.append(result)
        pending_checkpoint_rows += 1
        elapsed = elapsed_before + time.perf_counter() - started
        # 批量落盘 checkpoint（粒度 = config.batch_writes）：全量结果
        # JSON 可能几 MB，每条样本都重写会把 IO 放大成瓶颈；批量写的
        # 代价是"中断最多损失一个批次"，秒级推理下可接受
        if pending_checkpoint_rows >= max(config.batch_writes, 1):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps({"version": "szy", "duration": elapsed, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
            pending_checkpoint_rows = 0
    duration = elapsed_before + time.perf_counter() - started
    # 正常结束时保存尾部不足一个批次的结果；恰好落盘则不重复重写
    if pending_checkpoint_rows or not checkpoint_path.is_file():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps({"version": "szy", "duration": duration, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    # 收尾把 ASR 缓存里最后不足一批的条目落盘（进程内缓存始终有，
    # 是否持久化由配置控制）
    runner.flush_caches()

    # ---- 汇总 + 提交物 ----
    summary = summarize(results, duration)
    summary["route"] = "szy"                     # 汇总也带上路线标记（审计口径）
    write_outputs(results, summary, Path(args.output), Path(args.final_output))
    _auto_report(Path(args.final_output))

    # ---- manifest：资产指纹（哪份配置 + 哪份权重跑出来的，可追溯） ----
    # 三份关键权重的 SHA-256：实验复现的第一道保险。文件名可能被替换
    # 而内容不动，哈希才能证明"这次跑的和上次跑的是同一份权重"。
    # resolve() 把相对路径转绝对路径，任何工作目录下哈希目标一致
    manifest = {
        "version": "szy",
        "config": config.__dict__,               # 完整配置快照（frozen 对象的 __dict__ 即字段字典）
        "data_root": str(Path(args.data_root).resolve()),
        "jsonl": [str(Path(path).resolve()) for path in args.jsonl],
        "sample_count": len(results),
        "moss_model_sha256": sha256_file(Path(config.moss_model).resolve()),
        "fusion_backend_sha256": sha256_file(Path(config.fusion_backend).resolve()),
        "finetuned_sv_ckpt_sha256": sha256_file(Path(config.finetuned_sv_ckpt).resolve()),
        "output": str(Path(args.output).resolve()),
        "detail": str(Path(args.final_output).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # ensure_ascii=False：中文/路径原样输出（否则全部 \uXXXX 转义，不可读）；
    # flush=True：立即刷到终端——管道重定向日志时，不 flush 可能迟迟看不到
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    # 作为脚本直接执行（python -m szy.cli / python cli.py）时运行 main；
    # 被 import 时（如单测 import build_parser）不运行——Python 惯用法
    main()
