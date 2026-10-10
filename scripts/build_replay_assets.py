#!/usr/bin/env python3
"""构建全链路重放所需的一次性资产（FireRed ASR）。

part=dataseta：为 sweep-ep6 全量运行中**被门控拒绝**的样本补 transcribe
  （raw cmd 文本+置信度、cmd 两分离通道文本）——接受样本的文本已在
  detail.json 里。产物 outputs/sweep/ep6_replay_texts.json。
part=valid：为 dataset_dasynth_v4sep（新分离器）的 valid mos0/mos1 通道
  转写文本，产物 dataset_dasynth_v4sep/metadata/valid_moss_text.json。

为什么叫"重放资产"：这是重放（replay）工程手法的前半段。语音管线跑一次
全量实验的昂贵部分有两块：ASR 转写（FireRed 模型前向）和语音分离。如果把
"文本 + 分离结果"在第一次运行时落盘，之后再做参数实验（换 SV 快照、换
阈值、做消融）时就完全不碰这两块，只重算便宜的决策部分——全量管线跑一次
数小时，重放一次几分钟。代价只是提前用这个脚本把缺的文本补齐。落盘的
JSON 就是这个脚本的产物，它被 replay_full.py / replay_gate_ablation.py
直接读取。

注意脚本顶部的手动 sys.path 注入（见下）：脚本位于 scripts/ 而不是包内，
直接执行时 Python 默认只把 scripts/ 放进搜索路径，import midea_qiji / szy
会失败，所以必须在 import 之前把 src 和 vendor 插进去。

用法（yuyin，仓库根目录）：
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/build_replay_assets.py --part dataseta
说明：需要 GPU（FireRed 模型前向）；CUDA_VISIBLE_DEVICES=1 指定用 1 号卡。

依赖：midea_qiji.recognizers.FireRedRecognizer、szy.helpers（两个都靠下面
的 sys.path 注入才能导入）；标准库 argparse/json/sys/pathlib。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---- 手动扩展模块搜索路径（必须在 import 仓库包之前执行）----
# __file__ = 本脚本路径；resolve() 转绝对路径并解开软链；
# parents[1] 往上数一级 = 仓库根（scripts/ 的上一级）；_WS 再上一级 =
# 工作区根（model/、dataset/ 等资产目录所在处）。
_REPO = Path(__file__).resolve().parents[1]
_WS = _REPO.parent
# 脚本不在包内，直接执行时 Python 的模块搜索路径（sys.path）里只有
# scripts/ 目录和标准库位置，找不到 src/ 下的 midea_qiji、szy 包。
# 两种解决方式：1) 运行时设 PYTHONPATH=src（用户要做额外动作，易忘）；
# 2) 脚本自己 insert（开箱即用）。insert(0, ...) 插到最前面 = 最高优先级，
# 保证 import 时一定先用仓库里的版本，而不是环境里可能装过的同名旧包。
sys.path.insert(0, str(_REPO / "src"))
# vendor 目录是 FireRed 运行时依赖的补丁/适配层，同样需要先可见。
sys.path.insert(0, str(_WS / "model" / "runtime" / "vendor"))

# 上面 insert 之后 import 才能成功。noqa: E402 是告诉 flake8/linter：
# "我知道 import 不在文件顶部（违反 E402 规则），这是有意的"——
# 因为本脚本的 import 依赖前面几行路径注入。
from core.firered import FireRedRecognizer  # noqa: E402
from pipeline.helpers import transcribe_firered_rows  # noqa: E402

# 各数据/产物路径常量。集中在顶部的好处：改路径只动一处；
# 且一眼能看出本脚本的输入输出全貌。
# datasetA：竞赛数据集（正负样本的音频 + 唤醒音频）。
DATASETA = _WS / "dataset" / "dataset_test" / "datasetA" / "datasetA"
# ep6_detail.json：全量运行产出的逐样本 detail（含 gate_features、accepted、
# 以及已接受样本的转写文本）——"已有文本"的来源。
DETAIL = _REPO / "outputs" / "sweep_v4" / "ep6_detail.json"
# 分离缓存目录：szy 管线分离器落盘的唤醒/指令两个说话人通道
# （命名规则 {item}_wake_ms{0,1}.wav / {item}_command_ms{0,1}.wav，
# item 里 "/" 换成 "_"）。
CACHE = _REPO / "outputs" / "szy_cache_sweep_best" / "separation"
# part=dataseta 的产物：uid -> {raw_text, raw_confidence, cmd0_text, cmd1_text}
OUT_A = _REPO / "outputs" / "sweep_v4" / "ep6_replay_texts.json"
# 新分离器数据集（valid 侧用于重训 logistic 定阈）。
V4SEP = _WS / "dataset_dasynth_v4sep"
# part=valid 的产物：id -> {ms0, ms1}（两分离通道的转写文本）。
OUT_B = V4SEP / "metadata" / "valid_moss_text.json"

# FireRed ASR 的三段路径：模型权重、代码仓库、运行时补丁。
FIRERED_MODEL = _WS / "model" / "external" / "FireRedASR2-AED"
FIRERED_REPO = _WS / "model" / "external" / "FireRedASR2S"
FIRERED_VENDOR = _WS / "model" / "runtime" / "vendor"


def part_dataseta(asr: FireRedRecognizer, batch: int) -> None:
    # 补全 datasetA 的重放文本：detail.json 里已接受的样本有 raw_text，
    # 被门控拒绝的样本没有——但重放（换阈值/换 SV 快照后）这些样本可能
    # 被新决策接受，届时需要它们的转写文本，所以现在补上。
    detail = json.load(DETAIL.open(encoding="utf-8"))["results"]
    assets: dict[str, dict] = {}
    todo: list[tuple[str, list[Path]]] = []
    for r in detail:
        uid = r["id"]
        # 分离缓存文件命名规则：id 里的 "/" 换成 "_"
        # （与 szy.helpers.separation_paths_for 保持一致）。
        item = uid.replace("/", "_")
        if r.get("raw_text") is not None:
            # 已接受样本：文本现成，直接搬进 assets（float() 把
            # detail 里字符串形式的置信度转成数值，重放计算要用）。
            assets[uid] = {
                "raw_text": r["raw_text"],
                "raw_confidence": float(r["raw_confidence"]),
                "cmd0_text": r.get("command_source0_text", ""),
                "cmd1_text": r.get("command_source1_text", ""),
            }
        else:
            # 被拒绝样本：需要转写三条音频——原始指令混音 + 两个分离通道。
            # datasetA 的目录结构：第一级是子集目录，第二级是 {stem}.wav。
            sample_dir = DATASETA / uid.split("/")[0]
            raw_cmd = sample_dir / f"{uid.split('/')[1]}.wav"
            todo.append((uid, [
                raw_cmd,
                CACHE / f"{item}_command_ms0.wav",
                CACHE / f"{item}_command_ms1.wav",
            ]))
    # flush=True：print 默认行缓冲，输出被重定向到文件/管道时变成块缓冲，
    # 日志会"憋着"很久不落盘；flush 强制立刻写出，长任务看进度就靠它。
    print(f"已有文本 {len(assets)}，待补 {len(todo)}", flush=True)
    for start in range(0, len(todo), batch):
        # 分批处理：每批最多 batch 个样本（每个样本 3 条音频，即 3*batch 条）。
        # 批处理比逐条转写快得多（GPU 前向可以打包），batch 大小可调。
        chunk = todo[start : start + batch]
        # 展平：本批所有样本的全部待转写路径，保持 (样本0 三条, 样本1 三条, ...) 的顺序。
        paths = [p for _, ps in chunk for p in ps]
        # transcribe_firered_rows 批量转写，返回按输入顺序的
        # [{"text": 归一化文本, "confidence": float}]。
        # 第三个参数 "rp" 是本次调用的 uttid 前缀（replay 的缩写），
        # FireRed 内部用它命名日志里的每条音频，不同调用用不同前缀
        # 便于在日志里分辨来源。
        outs = transcribe_firered_rows(asr, paths, "rp")
        for i, (uid, _) in enumerate(chunk):
            # 每个样本占 3 个输出位：raw、ms0、ms1，按切片取回。
            raw, c0, c1 = outs[3 * i : 3 * i + 3]
            assets[uid] = {
                "raw_text": raw["text"], "raw_confidence": raw["confidence"],
                "cmd0_text": c0["text"], "cmd1_text": c1["text"],
            }
        # 每转写 10 批（或最后一批）报一次进度；条件里同时判"最后一批"，
        # 防止总数不是整 10 批时最后一段进度永远不打印。
        if (start + batch) % (batch * 10) == 0 or start + batch >= len(todo):
            print(f"asr {min(start + batch, len(todo))}/{len(todo)}", flush=True)
    # json.dump 一次性写整个字典；ensure_ascii=False 让中文原样存储
    # （否则每个汉字变成 \uXXXX 转义，文件难读也略大）。
    json.dump(assets, OUT_A.open("w", encoding="utf-8"), ensure_ascii=False)
    print(f"written {OUT_A} ({len(assets)})", flush=True)


def part_valid(asr: FireRedRecognizer, batch: int) -> None:
    # 为 valid 集的两个分离通道（mos0/mos1）生成转写文本。
    # mos0 目录下每个 {id}.wav 对应一条混音，mos1 目录存另一说话人通道。
    # p.stem 取"去掉扩展名的文件名"作为 id；sorted 保证顺序确定，
    # 多次运行产物一致（可复现性习惯）。
    ids = sorted(p.stem for p in (V4SEP / "valid" / "mos0").glob("*.wav"))
    print(f"valid 通道转写 {len(ids)} 条", flush=True)
    out: dict[str, dict] = {}
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        # 路径顺序：先 mos0 全部、再 mos1 全部（两段各 n 条），
        # 取回时靠下面的切片错位 n 拿到对应通道。
        paths = [V4SEP / "valid" / "mos0" / f"{i}.wav" for i in chunk] + \
                [V4SEP / "valid" / "mos1" / f"{i}.wav" for i in chunk]
        outs = transcribe_firered_rows(asr, paths, "rp")
        n = len(chunk)
        for i, uid in enumerate(chunk):
            # outs[i] 是 mos0 通道的转写；outs[n + i] 是 mos1 的
            # （因为输入顺序 = mos0 段在前、mos1 段在后）。
            out[uid] = {"ms0": outs[i]["text"], "ms1": outs[n + i]["text"]}
        if (start + batch) % (batch * 10) == 0 or start + batch >= len(ids):
            print(f"asr {min(start + batch, len(ids))}/{len(ids)}", flush=True)
    # parents=True, exist_ok=True：产物目录可能尚不存在，像 mkdir -p 一样
    # 逐级创建（数据侧脚本先于 metadata 目录存在的常见情况）。
    OUT_B.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, OUT_B.open("w", encoding="utf-8"), ensure_ascii=False)
    print(f"written {OUT_B} ({len(out)})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    # choices 限定取值：传错参数时 argparse 直接在解析期报错并打印合法值，
    # 比跑到一半才发现参数无效好得多。required=True 强制必须选一个 part。
    ap.add_argument("--part", choices=["dataseta", "valid"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    # FireRedRecognizer(模型权重, 设备, 代码仓库, vendor 补丁)：
    # 组装一个可批量转写的 ASR 后端实例。
    asr = FireRedRecognizer(FIRERED_MODEL, args.device, FIRERED_REPO, FIRERED_VENDOR)
    if args.part == "dataseta":
        part_dataseta(asr, args.batch)
    else:
        part_valid(asr, args.batch)


if __name__ == "__main__":
    main()
