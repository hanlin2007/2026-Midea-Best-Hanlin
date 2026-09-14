#!/usr/bin/env python3
"""Move workspace model assets into ../model and leave compatibility links.

（中文扩充）把散落在工作区各处的模型资产搬到 model/workspace/ 统一管理，
并在原位置留下软链接（兼容链接），最后写出 model/MANIFEST.json 资产台账。

为什么要把权重搬走：竞赛项目会积累大量权重文件——预训练模型、各轮微调的
checkpoint、LoRA adapter——散落在 exp_sep/、midea/outputs3/、sv_asr/ 等
十几个目录里。散乱分布的坏处：备份和迁移时要到处找、容易漏；磁盘满了
不知道该删谁；每次实验都要记住"这次用的是哪份权重"。统一搬到 model/
之后：所有模型资产只有一个存放点（单一事实来源），MANIFEST.json 记录
每个资产的路径与指纹，check_models.py 可以一键自检完整性。

为什么留软链接而不是直接改代码路径：历史脚本和配置里写死了旧路径
（如 sv_selector/selector.pt）。软链接让旧路径继续"看起来可用"，
新旧代码都能跑，迁移无需同步改代码。软链接存的是相对路径
（os.path.relpath），因此整个工作区整体挪动时链接依然有效。

设计模式——plan / apply 两阶段：默认只打印搬迁计划（plan），加了
--apply 才真正动文件（apply）。"先看计划、确认无误、再执行"是任何
不可逆操作（移动 GB 级文件）的稳妥姿势，等于内置了一次 dry-run。

用法（仓库根目录）：
  python scripts/relocate_workspace_models.py            # 只看计划，不动文件
  python scripts/relocate_workspace_models.py --apply    # 确认计划后真正搬迁
  python scripts/check_models.py                         # 搬迁后自检

依赖：标准库 argparse / hashlib / json / os / pathlib / typing。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

# 权重文件后缀集合：什么样的文件被认定为"模型资产"。
# 用 set 而不是 list：成员判断（in）是 O(1)，扫描成千上万个文件时更快。
WEIGHT_SUFFIXES = {
    ".bin", ".ckpt", ".h5", ".hdf5", ".joblib", ".model", ".onnx",
    ".pb", ".pt", ".pth", ".safetensors", ".tflite",
}
# 顶层排除目录：这些目录整棵跳过，不扫描、不搬迁（数据集/探针数据/外部
# 数据不是模型资产；midea-best 与 model 是管理目录本身，不能搬自己）。
EXCLUDED_TOP_LEVEL = {"dataset", "dataset_probe", "external_data", "midea-best", "model"}
# 路径段排除：任何一级目录命中这些段就跳过。.git 是版本库；.vendor_transformers
# 是运行时才生成的依赖缓存；.venv-*（下面单独判断）是虚拟环境。
EXCLUDED_PARTS = {".git", ".vendor_transformers"}
# 扫描根清单：历史上权重实际出现过的位置，逐个枚举而不是全盘扫描，
# 一是快，二是避免把无关大目录（如数据集）卷进来。
WEIGHT_SCAN_ROOTS = (
    "exp_sep/moss_anchor_external_v1",
    "midea/outputs",
    "midea/outputs3",
    "midea/tmp",
    "sv_asr",
    "sv_backend",
    "sv_finetune",
    "sv_probe",
    "sv_selector/selector.pt",
    "sv_tse",
    "sv_firered",
)
# 目录型资产（bundle）：整个目录作为一个资产整体搬迁和记账，而不是
# 拆成里面的每个小文件。LoRA adapter 目录内文件数量多且必须整目录
# 原样保留（config.json + adapter_model.safetensors + ...），拆开会破坏结构。
BUNDLE_PATHS = (
    "midea/models",
    "midea/outputs3/qwen_target_lora/checkpoint-1250",
    "midea/outputs3/qwen_target_lora/checkpoint-1275",
    "midea/outputs3/qwen_target_lora/final_adapter",
    "midea/outputs3/qwen_target_lora_smoke/checkpoint-10",
    "midea/outputs3/qwen_target_lora_smoke/final_adapter",
    "sv_asr/out_full/best_adapter",
    "sv_asr/out_full/last_adapter",
    "sv_asr/out_probe/best_adapter",
    "sv_asr/out_probe/last_adapter",
    "sv_firered/out/best",
    "sv_firered/out/last",
)
# 外部依赖软链：key 是 model/ 下的链接位置，value 是真实数据所在。
# 以 "/" 开头的 value 是绝对路径（服务器上另存的模型）；否则是相对
# 工作区根的路径（如 "vendor"）。外部数据不属于本工作区，不搬迁，
# 只留下链接并记录进 MANIFEST。
EXTERNAL_LINKS = {
    "external/FireRedASR2-AED": "/home/hustdun/models/FireRedASR2-AED",
    "external/FireRedASR2S": "/home/hustdun/zhy_workspace/FireRedASR2S",
    "runtime/vendor": "vendor",
}


def workspace_root() -> Path:
    # 定位工作区根的标准手法：
    # __file__ = 本脚本自身的路径（相对当前工作目录，可能是相对路径）；
    # .resolve() 把相对路径变绝对路径、并解开所有软链接（防止脚本本身
    # 被软链引用时定位错乱）；
    # .parents[N] 是"往上数 N 级父目录"的可下标序列：
    #   parents[0] = scripts/，parents[1] = 仓库根 midea_qiji_best_hanlin/，
    #   parents[2] = 工作区根（model/、dataset/ 等资产目录所在处）。
    # 这样无论从哪个目录、以什么方式调用脚本，都能回到工作区根。
    return Path(__file__).resolve().parents[2]


def is_excluded(relative: Path) -> bool:
    # 判断"相对工作区根的路径"是否该排除。两段式判断：
    # 1) 顶层目录命中排除清单 → 直接排除（parts[0] 是路径的第一段）；
    #    parts 为空的情况理论上不会出现（relative 至少有一段的 Path），
    #    这里防御一下，保持函数对任意输入不抛异常。
    if not relative.parts or relative.parts[0] in EXCLUDED_TOP_LEVEL:
        return True
    # 2) 任意一级目录命中 .git / .vendor_transformers，或目录名以
    #    .venv- 开头（虚拟环境目录），也排除。
    #    startswith 前缀匹配而不是相等：.venv-py311 这类带后缀的也覆盖。
    return any(part in EXCLUDED_PARTS or part.startswith(".venv-") for part in relative.parts)


def is_under(path: Path, parent: Path) -> bool:
    # "path 是否在 parent 目录之内"。pathlib 的 relative_to 在两者没有
    # 从属关系时会抛 ValueError，这里把异常当控制流用：抛了就说明不在
    # 里面。这是 pathlib 的惯用写法（相对 path 字符串前缀比较更稳，
    # 不会把 /a/bc 误判成 /a/b 的子路径）。
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def iter_model_files(root: Path) -> Iterable[Path]:
    # 生成器函数（含 yield）：逐个产出"需要搬迁的权重文件"。
    # 用 yield 而不是 return list 的收益：调用方可以边扫描边消费，
    # 不必等全部文件收集完；如果调用方只取前几个（如调试时）就能提前停止，
    # 节省扫描时间。
    for relative_root in WEIGHT_SCAN_ROOTS:
        scan_root = root / relative_root
        if not scan_root.exists():
            continue
        # 扫描根可能是单个文件（如 sv_selector/selector.pt），
        # 也可能是目录——统一处理成"路径集合"。
        paths = (scan_root,) if scan_root.is_file() else scan_root.rglob("*")
        for path in paths:
            # 只收真实文件；软链接跳过（它们要么是别的资产的兼容链接，
            # 要么指向外部数据，都不该被搬动）。
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if is_excluded(relative):
                continue
            if path.suffix.lower() in WEIGHT_SUFFIXES:
                yield path


def directory_stats(path: Path) -> tuple[int, int]:
    # 统计目录型资产：返回 (总字节数, 文件数)。逐文件 st_size 求和，
    # 跳过软链接（链接不占本资产的数据，避免把外部数据的大小算进来）。
    total = 0
    count = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
            count += 1
    return total, count


def sha256(path: Path) -> str:
    # 分块 SHA-256：大文件安全哈希的标准写法。
    # hashlib 的 digest 对象支持流式 update()——喂多少算多少，
    # 全程内存占用只有"当前块"那么大，与文件总大小无关。
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # 两参 iter 技巧：iter(可调用对象, 哨兵值) 会反复调用第一个参数
        # （无参 lambda），直到返回值等于哨兵 b""——也就是读到文件末尾。
        # 于是这行 = "每次读 1MB，直到读完"，等价于手写 while 循环但更简洁。
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    # hexdigest() 把 32 字节二进制摘要渲染成 64 位十六进制字符串，
    # 方便写进 JSON 和肉眼比对。
    return digest.hexdigest()


def build_assets(root: Path) -> list[Path]:
    # 确定搬迁清单 = 目录型 bundle + 散落文件。返回按相对路径排序的列表
    # （排序保证计划输出稳定可读，也便于人眼核对）。
    bundles = [root / relative for relative in BUNDLE_PATHS]
    # 先验证所有声明的 bundle 都存在——bundle 是手工维护的清单，写错
    # 路径时宁可立刻报错，也不要静默搬出"缺了一块"的资产集合。
    missing = [str(path.relative_to(root)) for path in bundles if not path.exists()]
    if missing:
        raise FileNotFoundError(f"declared model bundles missing: {missing}")

    # 散落文件 = 所有权重文件里，不属于任何 bundle 目录的。
    # 必须排除 bundle 内的文件：bundle 已作为目录整体搬迁，
    # 里面的文件再单独搬一遍会造成重复搬迁和台账重复记账。
    files = [
        path for path in iter_model_files(root)
        if not any(is_under(path, bundle) for bundle in bundles)
    ]
    # 按相对路径排序，计划输出与台账顺序都稳定。
    return sorted([*bundles, *files], key=lambda path: str(path.relative_to(root)))


def record_for(source: Path, destination: Path, root: Path) -> dict[str, object]:
    # 为单个资产生成 MANIFEST 台账条目：搬迁后（destination 是数据
    # 新家，source 是留在原地的软链接）按类型记账。
    relative = source.relative_to(root)
    if destination.is_dir():
        # 目录型：记总字节数 + 文件数，不记哈希（理由见 check_models.py）。
        size, file_count = directory_stats(destination)
        digest = None
        kind = "directory"
    else:
        # 单文件：记大小 + sha256 指纹，供日后逐字节校验。
        size = destination.stat().st_size
        file_count = 1
        digest = sha256(destination)
        kind = "file"
    return {
        "original_path": str(relative),
        "canonical_path": str(destination.relative_to(root)),
        "kind": kind,
        "bytes": size,
        "file_count": file_count,
        "sha256": digest,
    }


def ensure_external_links(root: Path) -> list[dict[str, str]]:
    # 在 model/ 下建立外部依赖软链并生成台账记录。
    model_root = root / "model"
    records = []
    for relative, source_value in EXTERNAL_LINKS.items():
        link = model_root / relative
        # 绝对路径（以 / 开头）直接使用；相对路径先相对工作区根 resolve
        # 成绝对路径（resolve 也顺带解开已有软链、规范化 ..）。
        source = (root / source_value).resolve() if not source_value.startswith("/") else Path(source_value)
        # 外部依赖必须先存在——它们不在本工作区，搬迁工具无法修复缺失。
        if not source.exists():
            raise FileNotFoundError(f"external runtime dependency missing: {source}")
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            # 目标位置已有东西：只有"已是软链且指向正确目标"才算没事，
            # 否则是冲突（可能是真文件占了位置），必须报错让人工裁决，
            # 绝不能覆盖删除。
            if not link.is_symlink() or link.resolve() != source.resolve():
                raise FileExistsError(f"external link conflicts with expected target: {link}")
        else:
            # os.path.relpath(source, start=link.parent)：把绝对路径换算成
            # "从链接所在目录出发"的相对路径再创建链接。用相对链接而不是
            # 绝对链接，是为了整个工作区整体搬动时链接依然有效——
            # 绝对链接换台机器/换个挂载点就全部悬空。
            # target_is_directory=source.is_dir()：Windows 建目录链接需要
            # 这个标志（POSIX 上被忽略），带上它跨平台才安全。
            link.symlink_to(os.path.relpath(source, start=link.parent), target_is_directory=source.is_dir())
        records.append({"link": str(link.relative_to(root)), "target": str(source)})
    return records


def plan(root: Path) -> list[dict[str, object]]:
    # 阶段一（只读）：为每个资产计算搬迁方案，不碰任何文件。
    # status 三态：
    #   already_linked：资产本来就是软链接且已指向目的地（幂等重跑的情况）；
    #   conflict：目的地已有东西、或链接指向别处——apply 阶段会整体拒绝；
    #   ready：可以安全搬迁。
    model_root = root / "model" / "workspace"
    result = []
    for source in build_assets(root):
        destination = model_root / source.relative_to(root)
        if source.is_symlink():
            status = "already_linked" if destination.exists() and source.resolve() == destination.resolve() else "conflict"
        elif destination.exists() or destination.is_symlink():
            status = "conflict"
        else:
            status = "ready"
        result.append({
            "source": str(source.relative_to(root)),
            "destination": str(destination.relative_to(root)),
            "kind": "directory" if source.is_dir() else "file",
            "status": status,
        })
    return result


def apply(root: Path) -> dict[str, object]:
    # 阶段二（写操作）：执行搬迁。设计原则：
    # 1) 先整体检查冲突，任何一个 conflict 就拒绝全部搬迁——搬迁要么
    #    完整成功，要么什么都不动（近似"事务"），绝不留半搬半留的状态；
    # 2) rename + 反向 symlink：rename 在同一文件系统上是瞬时元数据操作
    #    （不复制数据，GB 级文件也秒完成），随后在原位创建指向新家的
    #    软链接，旧路径继续可用。
    entries = plan(root)
    conflicts = [entry for entry in entries if entry["status"] == "conflict"]
    if conflicts:
        raise RuntimeError(f"migration conflicts: {conflicts}")
    records = []
    for entry in entries:
        source = root / str(entry["source"])
        destination = root / str(entry["destination"])
        if entry["status"] == "ready":
            # mkdir(parents=True, exist_ok=True)：像 mkdir -p 一样
            # 逐级创建父目录；exist_ok 让并发/重复执行不因目录已存在而报错。
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            # 原位置留下相对路径软链接：搬迁后整个工作区仍可整体搬动。
            source.symlink_to(
                os.path.relpath(destination, start=source.parent),
                target_is_directory=destination.is_dir(),
            )
        records.append(record_for(source, destination, root))

    external = ensure_external_links(root)
    manifest = {
        "schema": "szy_workspace_model_registry/v1",
        "workspace": str(root),
        "asset_count": len(records),
        "assets": records,
        "external_dependencies": external,
    }
    manifest_path = root / "model" / "MANIFEST.json"
    # indent=2 让 JSON 人类可读（diff/审阅友好）；末尾补一个换行符是
    # POSIX 文本文件惯例（很多工具假定文本以换行结尾），也让 git diff
    # 更干净。ensure_ascii=False 保留中文等非 ASCII 字符原样。
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    # action="store_true"：布尔开关，出现即 True，默认 False。
    # 默认（不加 --apply）只打印计划——把"危险操作"做成显式选择。
    parser.add_argument("--apply", action="store_true", help="move assets after printing the exact plan")
    args = parser.parse_args()
    root = workspace_root()
    if args.apply:
        manifest = apply(root)
        print(json.dumps({"asset_count": manifest["asset_count"], "manifest": "model/MANIFEST.json"}, ensure_ascii=False))
    else:
        print(json.dumps(plan(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
